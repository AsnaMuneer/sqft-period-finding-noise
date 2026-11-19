"""Shared noise channels, branching simulator and metrics for the period-finding study.

Used by:
  recyc_exact.py          oracle-noise model (the oracle is decomposed, so its gates are noisy)
  oracle_ideal_study.py   oracle-ideal model (the oracle is one unitary step, so it is noiseless)

The two models differ only in how the circuit is built. build_steps() handles both:
a decomposed oracle contributes sx, x, cx and rz gates, which receive noise, while a
'unitary' oracle step receives none.

Depolarizing convention B (matched to Qiskit Aer 0.15.1 in recyc_exact.py):
  1 qubit: (1 - 3p/4) rho + (p/4) sum_{P in X,Y,Z} P rho P
  2 qubit: (1 - 15p/16) rho + (p/16) sum_{P != I} P rho P
"""
from fractions import Fraction
from math import gcd, sqrt

import numpy as np
from qiskit.quantum_info import DensityMatrix, Kraus, Operator

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P1 = [I2, X, Y, Z]
P2 = [np.kron(a, b) for a in P1 for b in P1]
II = np.eye(4, dtype=complex)

PROJ0 = Kraus([np.array([[1, 0], [0, 0]], dtype=complex)])
PROJ1 = Kraus([np.array([[0, 0], [0, 1]], dtype=complex)])
RESET = Kraus([np.array([[1, 0], [0, 0]], dtype=complex),
               np.array([[0, 1], [0, 0]], dtype=complex)])
NOISY = ("depolarizing", "bitflip")


# ----------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------
def kraus_1q(p, channel, conv="B"):
    if channel == "bitflip":
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]
    if conv == "A":   # (1-p) rho + p/3 sum_P P rho P
        return [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * P for P in (X, Y, Z)]
    return [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * P for P in (X, Y, Z)]


def kraus_2q(p, channel, conv="B"):
    if channel == "bitflip":
        return [np.sqrt(1 - p) * II,
                np.sqrt(p / 2) * np.kron(X, I2),
                np.sqrt(p / 2) * np.kron(I2, X)]
    if conv == "A":   # (1-p) rho + p/15 sum_{P != I} P rho P
        return [np.sqrt(1 - p) * II] + [np.sqrt(p / 15) * P for P in P2[1:]]
    return [np.sqrt(1 - 15 * p / 16) * II] + [np.sqrt(p / 16) * P for P in P2[1:]]


def reset_error_kraus(reset_p):
    """X error with probability reset_p after a reset."""
    return Kraus([np.sqrt(1 - reset_p) * I2, np.sqrt(reset_p) * X])


# ----------------------------------------------------------------------------
# Circuit -> steps
# ----------------------------------------------------------------------------
def build_steps(tq):
    """Turn a transpiled circuit (possibly with control flow) into simulator steps.

    Noise tags: sx, x -> '1q'; cx -> '2q'; rz, p, and 'unitary' (ideal oracle) -> None.
    """
    steps = []
    for inst in tq.data:
        op = inst.operation
        name = op.name
        qargs = [tq.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "reset":
            steps.append(("reset", qargs[0]))
        elif name == "measure":
            steps.append(("measure", qargs[0], tq.find_bit(inst.clbits[0]).index))
        elif name == "if_else":
            cond_bit, val = op.condition
            cl = tq.find_bit(cond_bit).index
            body = op.blocks[0]
            body_steps = []
            for binst in body.data:
                bq = [qargs[body.find_bit(q).index] for q in binst.qubits]
                body_steps.append((Operator(binst.operation), bq))
            steps.append(("if", cl, val, body_steps))
        elif name in ("sx", "x"):
            steps.append(("gate", Operator(op), qargs, "1q"))
        elif name == "cx":
            steps.append(("gate", Operator(op), qargs, "2q"))
        elif name in ("rz", "p", "unitary"):
            steps.append(("gate", Operator(op), qargs, None))
        else:
            raise ValueError(f"unsupported operation: {name}")
    return steps


# ----------------------------------------------------------------------------
# Branching simulator
# ----------------------------------------------------------------------------
def _add(d, key, rho):
    if key in d:
        d[key] = DensityMatrix(d[key].data + rho.data)
    else:
        d[key] = rho


def simulate(steps, nq, ncl, channel, p, readout=0.0, conv1="B", conv2="B", reset_p=0.0):
    """Exact distribution over the classical register.

    channel, p:  gate noise on the '1q' and '2q' gates (depolarizing or bitflip).
    readout:     flip probability applied to each recorded bit at each measurement.
    reset_p:     X error probability after each reset.
    Returns an array of length 2**ncl; index x = sum_i c[i] 2**i.
    """
    gate_noise = channel in NOISY and p > 0
    K1 = Kraus(kraus_1q(p, channel, conv1)) if gate_noise else None
    K2 = Kraus(kraus_2q(p, channel, conv2)) if gate_noise else None
    KR = reset_error_kraus(reset_p) if reset_p > 0 else None

    branches = {tuple([0] * ncl): DensityMatrix.from_label("0" * nq)}
    for step in steps:
        kind = step[0]
        new = {}
        if kind == "gate":
            _, U, qargs, noise = step
            for rec, rho in branches.items():
                r = rho.evolve(U, qargs)
                if gate_noise and noise == "1q":
                    r = r.evolve(K1, [qargs[0]])
                elif gate_noise and noise == "2q":
                    r = r.evolve(K2, qargs)
                _add(new, rec, r)
        elif kind == "reset":
            q = step[1]
            for rec, rho in branches.items():
                r = rho.evolve(RESET, [q])
                if KR is not None:
                    r = r.evolve(KR, [q])
                _add(new, rec, r)
        elif kind == "measure":
            _, q, cl = step
            for rec, rho in branches.items():
                for m, proj in ((0, PROJ0), (1, PROJ1)):
                    rho_m = rho.evolve(proj, [q])
                    for rb in (0, 1):
                        w = (1 - readout) if rb == m else readout
                        if w == 0:
                            continue
                        key = list(rec)
                        key[cl] = rb
                        _add(new, tuple(key), DensityMatrix(rho_m.data * w))
        elif kind == "if":
            _, cl, val, body = step
            for rec, rho in branches.items():
                if rec[cl] == val:
                    r = rho
                    for U, bq in body:
                        r = r.evolve(U, bq)
                    _add(new, rec, r)
                else:
                    _add(new, rec, rho)
        branches = new

    dist = np.zeros(2 ** ncl)
    for rec, rho in branches.items():
        x = sum(bit << i for i, bit in enumerate(rec))
        dist[x] += float(np.real(np.trace(rho.data)))
    return dist


# ----------------------------------------------------------------------------
# Period-finding metrics
# ----------------------------------------------------------------------------
def order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def peaks(N, a, t):
    """Period-consistent outcomes and the useful subset (gcd(j, r) = 1)."""
    M = 2 ** t
    r = order(a, N)
    good = sorted({int(round(j * M / r)) % M for j in range(r)})
    useful = sorted({int(round(j * M / r)) % M for j in range(r) if gcd(j, r) == 1})
    return good, useful


def factor_mask(N, a, t):
    """True for outcomes x whose continued-fraction denominator yields a non-trivial factor."""
    M = 2 ** t
    mask = np.zeros(M, dtype=bool)
    for x in range(1, M):
        r = Fraction(x, M).limit_denominator(N).denominator
        if r % 2 or pow(a, r, N) != 1:
            continue
        h = pow(a, r // 2, N)
        if h in (1, N - 1):
            continue
        f1, f2 = gcd(h - 1, N), gcd(h + 1, N)
        mask[x] = f1 not in (1, N) and f2 not in (1, N) and f1 * f2 == N
    return mask


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def tvd(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# ----------------------------------------------------------------------------
# Qiskit Aer noise models and analytic readout (used by sweep_exact.py and sweep_final.py)
# ----------------------------------------------------------------------------
def aer_noise_model(channel, p, rz=False):
    """Aer noise model for the paper's main results.

    rz=False (default): rz is error-free. rz=True adds bit-flip errors after rz
    gates; use this only for the sensitivity variant.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

    nm = NoiseModel()
    one_q = ["sx", "x"] + (["rz"] if rz else [])
    if channel == "depolarizing":
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), one_q)
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    elif channel == "bitflip":
        nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), one_q)
        nm.add_all_qubit_quantum_error(
            pauli_error([("XI", p / 2), ("IX", p / 2), ("II", 1 - p)]), ["cx"])
    return nm


def readout_flip(dist, t, p):
    """Exact effect of symmetric readout error on a distribution over t bits."""
    return flip(dist, t, p)


def flip(dist, t, p):
    """Symmetric bit-flip with probability p on each of the t bits of a distribution."""
    if p == 0:
        return dist
    d = np.asarray(dist).reshape((2,) * t)
    M = np.array([[1 - p, p], [p, 1 - p]])
    for ax in range(t):
        d = np.moveaxis(np.tensordot(M, d, axes=([1], [ax])), 0, ax)
    return d.reshape(-1)
