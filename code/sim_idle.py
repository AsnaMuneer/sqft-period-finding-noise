"""Exact density-matrix simulation of the inverse-QFT stage with gate noise and
T1/T2 idle decoherence timed from an ASAP schedule with feedback latency L.

Input: the post-oracle counting state for N = 15, a = 7 (oracle ideal; see oracle_ideal_study.py).
Designs: full (deferred CP form + swaps, measured into c[j]) and sqft (hardware form:
H, classically controlled 1q phases, measure with branching on the classical record).
Pure NumPy. Convention: qubit q is tensor axis q; x = sum_i c[i] 2^i.
"""
import math
from math import gcd
from fractions import Fraction
import numpy as np
from latency_model import schedule

D1, D2, DM = 35.0, 300.0, 800.0
N, A = 15, 7
T1 = 100e3

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P1 = [I2, X, Y, Z]
P2 = [np.kron(a, b) for a in P1 for b in P1]
II = np.eye(4, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1M = np.array([[0, 0], [0, 1]], dtype=complex)


def apply_op(rho, K, qs, t):
    """rho -> K rho K^dagger on qubits qs (tensor shape (2,)*2t: rows then columns)."""
    k = len(qs)
    Kr = K.reshape((2,) * (2 * k))
    r = np.tensordot(Kr, rho, axes=(list(range(k, 2 * k)), list(qs)))
    r = np.moveaxis(r, list(range(k)), list(qs))
    Kc = K.conj().reshape((2,) * (2 * k))
    cols = [t + q for q in qs]
    r = np.tensordot(r, Kc, axes=(cols, list(range(k, 2 * k))))
    r = np.moveaxis(r, list(range(r.ndim - k, r.ndim)), cols)
    return r


def apply_channel(rho, kraus, qs, t):
    return sum(apply_op(rho, K, qs, t) for K in kraus)


def depol1(e):
    if e == 0:
        return [I2]
    return [math.sqrt(1 - e) * I2] + [math.sqrt(e / 3) * P for P in (X, Y, Z)]


def depol2(e):
    if e == 0:
        return [II]
    return [math.sqrt(1 - e) * II] + [math.sqrt(e / 15) * P for P in P2[1:]]


def idle_kraus(tau, T2):
    """Amplitude damping (T1) then phase damping so that coherence decays as exp(-tau/T2)."""
    if T2 is None or tau <= 0:
        return None
    g = 1 - math.exp(-tau / T1)
    ad = [np.diag([1.0, math.sqrt(1 - g)]).astype(complex),
          np.array([[0, math.sqrt(g)], [0, 0]], dtype=complex)]
    c = min(1.0, math.exp(-tau / T2) / math.sqrt(1 - g)) if g < 1 else 0.0
    pz = (1 - c) / 2
    pd = [math.sqrt(1 - pz) * I2, math.sqrt(pz) * Z]
    return [b @ a for b in pd for a in ad]


def full_program(t):
    ops = []
    for j in reversed(range(t)):
        ops.append(dict(kind="H", qs=[j], dur=D1, deps=[]))
        for k in range(j + 1, t):
            ops.append(dict(kind="CP", qs=[k, j], angle=-math.pi / 2 ** (k - j),
                            dur=2 * D2 + D1, deps=[]))
    for j in range(t // 2):
        ops.append(dict(kind="SWAP", qs=[j, t - 1 - j], dur=3 * D2, deps=[]))
    for j in range(t):
        ops.append(dict(kind="M", qs=[j], cbit=j, dur=DM, deps=[]))
    return ops


def sqft_program(t):
    ops, meas = [], {}
    for j in reversed(range(t)):
        ops.append(dict(kind="H", qs=[j], dur=D1, deps=[]))
        for k in range(j + 1, t):
            ops.append(dict(kind="P", qs=[j], angle=-math.pi / 2 ** (k - j), cond=t - 1 - k,
                            dur=D1, deps=[meas[k]]))
        ops.append(dict(kind="M", qs=[j], cbit=t - 1 - j, dur=DM, deps=[]))
        meas[j] = len(ops) - 1
    return ops


def input_state(t):
    """Post-oracle counting state (oracle ideal): mixture over target outcomes y."""
    M = 2 ** t
    ys = {}
    for x in range(M):
        ys.setdefault(pow(A, x, N), []).append(x)
    rho = np.zeros((2,) * (2 * t), dtype=complex)
    for xs in ys.values():
        v = np.zeros((2,) * t, dtype=complex)
        for x in xs:
            v[tuple((x >> q) & 1 for q in range(t))] = 1 / math.sqrt(len(xs))
        rho += (len(xs) / M) * np.multiply.outer(v, v.conj())
    return rho


def simulate(ops, t, L, T2, e1, e2):
    sched_ops = [dict(qubits=o["qs"], dur=o["dur"], deps=o["deps"]) for o in ops]
    start, end = schedule(sched_ops, t, L, {})
    branches = {tuple([0] * t): input_state(t)}
    last = [0.0] * t
    d1 = depol1(e1)
    d2 = depol2(e2)
    for i, op in enumerate(ops):
        s = start[i]
        for q in op["qs"]:
            kr = idle_kraus(s - last[q], T2) if T2 is not None else None
            if kr:
                branches = {k: apply_channel(r, kr, [q], t) for k, r in branches.items()}
        kind, qs = op["kind"], op["qs"]
        if kind == "H":
            branches = {k: apply_channel(apply_op(r, H, qs, t), d1, qs, t) for k, r in branches.items()}
        elif kind == "P":
            U = np.diag([1, np.exp(1j * op["angle"])])
            new = {}
            for k, r in branches.items():
                if k[op["cond"]] == 1:
                    r = apply_op(r, U, qs, t)
                new[k] = apply_channel(r, d1, qs, t)
            branches = new
        elif kind == "CP":
            U = np.diag([1, 1, 1, np.exp(1j * op["angle"])]).astype(complex)
            branches = {k: apply_channel(apply_op(r, U, qs, t), d2, qs, t) for k, r in branches.items()}
            branches = {k: apply_channel(r, d2, qs, t) for k, r in branches.items()}
        elif kind == "SWAP":
            S = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
            branches = {k: apply_channel(apply_op(r, S, qs, t), d2, qs, t) for k, r in branches.items()}
            branches = {k: apply_channel(r, d2, qs, t) for k, r in branches.items()}
            branches = {k: apply_channel(r, d2, qs, t) for k, r in branches.items()}
        elif kind == "M":
            new = {}
            for k, r in branches.items():
                for b, Pb in ((0, P0), (1, P1M)):
                    rb = apply_op(r, Pb, qs, t)
                    key = list(k)
                    key[op["cbit"]] = b
                    key = tuple(key)
                    new[key] = new[key] + rb if key in new else rb
            branches = new
        for q in qs:
            last[q] = end[i]
    total_time = max(end)
    M = 2 ** t
    dist = np.zeros(M)
    for k, r in branches.items():
        x = sum(bit << i for i, bit in enumerate(k))
        dist[x] += float(np.real(np.trace(r.reshape(M, M))))
    return dist, total_time


def factor_mask(t):
    M = 2 ** t
    mask = np.zeros(M, dtype=bool)
    for x in range(1, M):
        r = Fraction(x, M).limit_denominator(N).denominator
        if r % 2 or pow(A, r, N) != 1:
            continue
        h = pow(A, r // 2, N)
        if h in (1, N - 1):
            continue
        f1, f2 = gcd(h - 1, N), gcd(h + 1, N)
        mask[x] = f1 not in (1, N) and f2 not in (1, N) and f1 * f2 == N
    return mask


def metrics(dist, t):
    M = 2 ** t
    r = Fraction(1, 1)
    good = sorted({int(round(j * M / 4)) % M for j in range(4)})
    p_peak = sum(dist[x] for x in good)
    p_fact = dist[factor_mask(t)].sum()
    return p_peak, p_fact


if __name__ == "__main__":
    import sys
    t = 8
    d_full, _ = simulate(full_program(t), t, 0.0, None, 0.0, 0.0)
    d_sqft, _ = simulate(sqft_program(t), t, 0.0, None, 0.0, 0.0)
    print("noiseless: sum full", round(d_full.sum(), 9), " sum sqft", round(d_sqft.sum(), 9))
    print("noiseless TVD sqft vs full:", round(0.5 * np.abs(d_full - d_sqft).sum(), 9))
    print("noiseless metrics full:", metrics(d_full, t), " sqft:", metrics(d_sqft, t))
