import csv
import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, HGate, SXGate
from qiskit.quantum_info import DensityMatrix, Kraus, Operator
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from sqft_benchmark import BASIS, N_COUNT, SWEEPS, build_circuit, wilson
from sweep_exact import exact_dist, metrics, tvd_arr

OUT = "results_recyc"
N_SHOTS = 4096

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


def kraus_1q(p, channel, conv):
    if channel == "bitflip":
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]
    if conv == "A":   # (1-p) rho + p/3 sum_P P rho P
        return [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * P for P in (X, Y, Z)]
    return [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * P for P in (X, Y, Z)]


def kraus_2q(p, channel, conv):
    if channel == "bitflip":
        return [np.sqrt(1 - p) * II,
                np.sqrt(p / 2) * np.kron(X, I2),
                np.sqrt(p / 2) * np.kron(I2, X)]
    if conv == "A":   # (1-p) rho + p/15 sum_{P != I} P rho P
        return [np.sqrt(1 - p) * II] + [np.sqrt(p / 15) * P for P in P2[1:]]
    return [np.sqrt(1 - 15 * p / 16) * II] + [np.sqrt(p / 16) * P for P in P2[1:]]


def aer_state(qc, nm):
    qc2 = qc.copy()
    qc2.save_density_matrix()
    res = AerSimulator(method="density_matrix", noise_model=nm).run(qc2).result()
    dm = res.data(0)["density_matrix"]
    return np.asarray(dm.data if hasattr(dm, "data") else dm)


def choose_conventions(p=0.05):
    """Find which depolarizing convention Aer uses, by direct comparison."""
    # 1-qubit check on sx
    qc1 = QuantumCircuit(1)
    qc1.sx(0)
    nm1 = NoiseModel()
    nm1.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["sx"])
    aer1 = aer_state(qc1, nm1)
    conv1 = None
    for conv in ("A", "B"):
        ours = DensityMatrix.from_label("0").evolve(Operator(SXGate()))
        ours = ours.evolve(Kraus(kraus_1q(p, "depolarizing", conv)), [0])
        if np.abs(aer1 - ours.data).max() < 1e-8:
            conv1 = conv
    # 2-qubit check on cx
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.h(1)
    qc2.cx(0, 1)
    nm2 = NoiseModel()
    nm2.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    aer2 = aer_state(qc2, nm2)
    conv2 = None
    for conv in ("A", "B"):
        pre = DensityMatrix.from_label("00").evolve(Operator(HGate()), [0])
        pre = pre.evolve(Operator(HGate()), [1])
        ours = pre.evolve(Operator(CXGate()), [0, 1])
        ours = ours.evolve(Kraus(kraus_2q(p, "depolarizing", conv)), [0, 1])
        if np.abs(aer2 - ours.data).max() < 1e-8:
            conv2 = conv
    if conv1 is None or conv2 is None:
        raise RuntimeError(f"Could not match Aer depolarizing convention: 1q={conv1}, 2q={conv2}")
    print(f"Aer depolarizing convention matched: 1q={conv1}, 2q={conv2}", flush=True)
    return conv1, conv2


def build_steps(tq):
    steps = []
    for inst in tq.data:
        op = inst.operation
        qargs = [tq.find_bit(q).index for q in inst.qubits]
        name = op.name
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
        elif name in ("rz", "p", "sx", "x", "cx"):
            noise = {"sx": "1q", "x": "1q", "cx": "2q"}.get(name)
            steps.append(("gate", Operator(op), qargs, noise))
        else:
            raise ValueError(f"unsupported operation: {name}")
    return steps


def add(d, key, rho):
    if key in d:
        d[key] = DensityMatrix(d[key].data + rho.data)
    else:
        d[key] = rho


def simulate(steps, nq, ncl, channel, p, readout, conv1, conv2):
    gate_noise = channel in ("depolarizing", "bitflip") and p > 0
    K1 = Kraus(kraus_1q(p, channel, conv1)) if gate_noise else None
    K2 = Kraus(kraus_2q(p, channel, conv2)) if gate_noise else None

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
                add(new, rec, r)
        elif kind == "reset":
            q = step[1]
            for rec, rho in branches.items():
                add(new, rec, rho.evolve(RESET, [q]))
        elif kind == "measure":
            _, q, cl = step
            for rec, rho in branches.items():
                for m, proj in ((0, PROJ0), (1, PROJ1)):
                    rho_m = rho.evolve(proj, [q])
                    for r_bit in (0, 1):
                        w = (1 - readout) if r_bit == m else readout
                        if w == 0:
                            continue
                        key = list(rec)
                        key[cl] = r_bit
                        add(new, tuple(key), DensityMatrix(rho_m.data * w))
        elif kind == "if":
            _, cl, val, body = step
            for rec, rho in branches.items():
                if rec[cl] == val:
                    r = rho
                    for U, bq in body:
                        r = r.evolve(U, bq)
                    add(new, rec, r)
                else:
                    add(new, rec, rho)
        branches = new

    dist = np.zeros(2 ** N_COUNT)
    for rec, rho in branches.items():
        x = sum(bit << i for i, bit in enumerate(rec))
        dist[x] += float(np.real(np.trace(rho.data)))
    return dist


def main():
    os.makedirs(OUT, exist_ok=True)
    conv1, conv2 = choose_conventions()

    tq = transpile(build_circuit("recyc"), basis_gates=BASIS,
                   optimization_level=1, seed_transpiler=1)
    steps = build_steps(tq)
    nq, ncl = tq.num_qubits, tq.num_clbits
    print(f"recycled circuit: {nq} qubits, {ncl} classical bits, {len(steps)} steps", flush=True)

    qft_ideal = exact_dist(build_circuit("qft"), False, "none", 0)

    t0 = time.time()
    ideal = simulate(steps, nq, ncl, "none", 0.0, 0.0, conv1, conv2)
    print(f"ideal done ({time.time() - t0:.0f} s), sum={ideal.sum():.6f}, "
          f"TVD vs full QFT (ideal) = {tvd_arr(ideal, qft_ideal):.6f}", flush=True)

    settings = [("readout", p, 0.0, p) for p in SWEEPS["readout"]]
    settings += [("depolarizing", p, 0.0, p) for p in SWEEPS["depolarizing"]]
    settings += [("bitflip", p, 0.0, p) for p in SWEEPS["bitflip"]]

    rows = []
    for channel, rate, _, p in settings:
        t0 = time.time()
        if channel == "readout":
            dist = simulate(steps, nq, ncl, "none", 0.0, p, conv1, conv2)
        else:
            dist = simulate(steps, nq, ncl, channel, p, 0.0, conv1, conv2)
        pk, pu, pf = metrics(dist)
        lo, hi = wilson(round(pf * N_SHOTS), N_SHOTS)
        tv = tvd_arr(dist, ideal)
        rows.append(dict(circuit="recyc", form="mid-circuit exact", channel=channel,
                         p=rate, P_peak=round(pk, 6), P_useful=round(pu, 6),
                         P_fact=round(pf, 6), P_fact_lo=round(lo, 6),
                         P_fact_hi=round(hi, 6), TVD_from_ideal=round(tv, 6),
                         sum=round(float(dist.sum()), 6), shots_assumed=N_SHOTS))
        print(f"{channel:13s} p={rate:<7g} sum={dist.sum():.6f} P_peak={pk:.4f} "
              f"P_fact={pf:.4f} TVD={tv:.4f}  ({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(OUT, "recyc_exact.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()
