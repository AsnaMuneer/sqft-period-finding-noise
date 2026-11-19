"""Exact density-matrix sweep, earlier model (bit-flip on single-qubit gates only).

Superseded by sweep_final.py for the paper. Kept so results_exact/ can be reproduced.
Shared helpers (tvd, wilson, readout flip) now come from noise.py.
"""
import csv
import math
import os
import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from noise import flip, tvd, wilson
from sqft_benchmark import (BASIS, GOOD, N_COUNT, N_TARGET, SWEEPS, USEFUL,
                            build_circuit, controlled_power_gate, factor_found,
                            noise_model, rev_int)

OUT = "results_exact"
N_SHOTS = 4096   # shots assumed when computing sampling intervals


def sqft_deferred():
    t = N_COUNT
    cnt = QuantumRegister(t, "cnt")
    tgt = QuantumRegister(N_TARGET, "tgt")
    c = ClassicalRegister(t, "c")
    qc = QuantumCircuit(cnt, tgt, c)
    qc.x(tgt[0])
    qc.h(cnt)
    for j in range(t):
        g = controlled_power_gate(2**j)
        if g is not None:
            qc.append(g, [cnt[j], *tgt])
    for j in reversed(range(t)):
        qc.h(cnt[j])
        for k in range(j + 1, t):
            qc.cp(-math.pi / 2 ** (k - j), cnt[k], cnt[j])
    for j in range(t):
        qc.measure(cnt[j], c[t - 1 - j])
    return qc


def strip_measurements(circ):
    out = QuantumCircuit(*circ.qregs)
    for inst in circ.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def exact_dist(circ, reverse, channel, p):
    """Exact probability over classical outcome x (c[0] least significant)."""
    body = strip_measurements(circ)
    tq = transpile(body, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    tq.save_probabilities_dict(list(range(N_COUNT)), label="pr")
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(channel, p))
    raw = sim.run(tq, shots=1).result().data(0)["pr"]
    dist = np.zeros(2 ** N_COUNT)
    for k, v in raw.items():
        v_int = k if isinstance(k, int) else int(k, 2)
        x = rev_int(v_int) if reverse else v_int
        dist[x] += v
    return dist


def metrics(dist):
    pk = sum(dist[x] for x in GOOD)
    pu = sum(dist[x] for x in USEFUL)
    pf = sum(dist[x] for x in range(2 ** N_COUNT) if factor_found(x))
    return pk, pu, pf


def main():
    os.makedirs(OUT, exist_ok=True)
    designs = [
        ("qft", "standard", build_circuit("qft"), False),
        ("sqft", "deferred", sqft_deferred(), True),
    ]

    ideal = {}
    for name, form, circ, rev in designs:
        t0 = time.time()
        ideal[name] = exact_dist(circ, rev, "none", 0)
        print(f"{name} ideal done ({time.time() - t0:.0f} s)", flush=True)
    print(f"exact TVD sqft vs qft (ideal): {tvd(ideal['sqft'], ideal['qft']):.6f}",
          flush=True)

    rows = []

    def add(name, form, channel, p, dist):
        pk, pu, pf = metrics(dist)
        lo, hi = wilson(round(pf * N_SHOTS), N_SHOTS)
        tv = tvd(dist, ideal[name])
        rows.append(dict(circuit=name, form=form, channel=channel, p=p,
                         P_peak=round(pk, 6), P_useful=round(pu, 6),
                         P_fact=round(pf, 6), P_fact_lo=round(lo, 6),
                         P_fact_hi=round(hi, 6), TVD_from_ideal=round(tv, 6),
                         shots_assumed=N_SHOTS))
        print(f"{name:5s} {channel:12s} p={p:<7g} P_peak={pk:.4f} "
              f"P_fact={pf:.4f} TVD={tv:.4f}", flush=True)

    for name, form, circ, rev in designs:
        add(name, form, "none", 0.0, ideal[name])
        for p in SWEEPS["readout"]:
            add(name, form, "readout", p, flip(ideal[name], N_COUNT, p))
        for channel in ["depolarizing", "bitflip"]:
            for p in SWEEPS[channel]:
                add(name, form, channel, p, exact_dist(circ, rev, channel, p))

    with open(os.path.join(OUT, "exact_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()
