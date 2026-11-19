#!/usr/bin/env python3
"""Oracle-ideal noise study (Section 3.5 of the manuscript).

Isolates the inverse-QFT stage from the modular-exponentiation oracle:
  - Full QFT & Deferred SQFT: exact t-qubit density matrix obtained by tracing
    out the target register from the ideal post-oracle pure state, followed by
    noisy transpiled inverse-QFT stage in Qiskit Aer.
  - Recycled SQFT: exact classical-branching density-matrix simulation using
    noise.py, where each controlled modular multiplication is a noiseless
    'unitary' step, resets suffer X errors with probability p, and each
    recorded measurement bit flips with probability p.
  - Also evaluates a measurement-only channel (noiseless gates, bit-flip p on
    recorded outcomes) to isolate mid-circuit feedback sensitivity.

Writes results_oracle_ideal/oracle_ideal_summary.csv.
"""
import csv
import math
import os
import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit_aer import AerSimulator

from noise import (
    aer_noise_model,
    build_steps,
    factor_mask,
    flip,
    peaks,
    simulate,
    tvd,
    wilson,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_oracle_ideal")
BASIS = ["rz", "sx", "x", "cx"]
N_SHOTS = 4096

CASES = [(15, 7), (21, 2)]
GATE_RATES = [1e-3, 3e-3, 1e-2]
MEAS_ONLY_RATES = [0.01, 0.02, 0.05]


def rev_int(x, t):
    return int(format(x, f"0{t}b")[::-1], 2)


def controlled_mod_mult(a, e, N, n):
    """Controlled permutation |y> -> |a^e y mod N>, identity when a^e = 1 mod N."""
    mult = pow(a, e, N)
    if mult == 1:
        return None
    dim = 2 ** n
    P = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        z = (mult * y) % N if y < N else y
        P[z, y] = 1.0
    return UnitaryGate(P, label=f"U^{e}").control(1)


def post_oracle_reduced_dm(N, a, n, t):
    """Ideal reduced density matrix on the t counting qubits right after the oracle."""
    cnt = QuantumRegister(t, "cnt")
    tgt = QuantumRegister(n, "tgt")
    qc = QuantumCircuit(cnt, tgt)
    qc.x(tgt[0])
    qc.h(cnt)
    for j in range(t):
        g = controlled_mod_mult(a, 2 ** j, N, n)
        if g is not None:
            qc.append(g, [cnt[j], *tgt])
    sv = Statevector(qc)
    # Trace out target qubits [t, ..., t + n - 1]
    return partial_trace(sv, list(range(t, t + n)))


def build_iqft_stage(t, kind):
    """Inverse-QFT stage only (no oracle), transpiled to BASIS."""
    qc = QuantumCircuit(t)
    if kind == "qft":
        qc.append(QFTGate(t).inverse(), range(t))
    elif kind == "sqft":
        for j in reversed(range(t)):
            qc.h(j)
            for k in range(j + 1, t):
                qc.cp(-math.pi / 2 ** (k - j), k, j)
    else:
        raise ValueError(kind)
    return transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)


def run_stage_dm(rho_in, tq_stage, t, reverse, nm):
    """Evolve t-qubit post-oracle density matrix through noisy inverse-QFT stage."""
    qc = QuantumCircuit(t)
    qc.set_density_matrix(rho_in)
    qc.compose(tq_stage, inplace=True)
    qc.save_probabilities_dict(list(range(t)), label="pr")
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    raw = sim.run(qc, shots=1).result().data(0)["pr"]
    dist = np.zeros(2 ** t)
    for k, v in raw.items():
        idx = k if isinstance(k, int) else int(k, 2)
        dist[rev_int(idx, t) if reverse else idx] += float(v)
    return dist


def build_recyc_oracle_ideal(N, a, n, t):
    """Recycled circuit with opaque 'unitary' oracle steps so only QFT/reset/meas are noisy."""
    anc = QuantumRegister(1, "anc")
    tgt = QuantumRegister(n, "tgt")
    c = ClassicalRegister(t, "c")
    qc = QuantumCircuit(anc, tgt, c)
    qc.x(tgt[0])
    for j in reversed(range(t)):
        qc.reset(anc[0])
        qc.h(anc[0])
        g = controlled_mod_mult(a, 2 ** j, N, n)
        if g is not None:
            qc.append(g, [anc[0], *tgt])
        for k in range(j + 1, t):
            with qc.if_test((c[t - 1 - k], 1)):
                qc.p(-math.pi / 2 ** (k - j), anc[0])
        qc.h(anc[0])
        qc.measure(anc[0], c[t - 1 - j])

    # Transpile only non-unitary gates while preserving 'unitary' oracle steps as opaque
    return transpile(
        qc,
        basis_gates=BASIS + ["unitary", "reset", "measure", "if_else"],
        optimization_level=1,
        seed_transpiler=1,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    for N, a in CASES:
        n = N.bit_length()
        t = 2 * n
        good, useful = peaks(N, a, t)
        fmask = factor_mask(N, a, t)

        def eval_metrics(dist):
            pk = float(sum(dist[x] for x in good))
            pu = float(sum(dist[x] for x in useful))
            pf = float(dist[fmask].sum())
            return pk, pu, pf

        print(f"\n=== Oracle-Ideal Study: N={N}, a={a} (n={n}, t={t}) ===", flush=True)
        rho_cnt = post_oracle_reduced_dm(N, a, n, t)
        tq_full = build_iqft_stage(t, "qft")
        tq_sqft = build_iqft_stage(t, "sqft")

        ideal_full = run_stage_dm(rho_cnt, tq_full, t, False, aer_noise_model("depolarizing", 0.0))
        ideal_sqft = run_stage_dm(rho_cnt, tq_sqft, t, True, aer_noise_model("depolarizing", 0.0))
        _, _, pf0 = eval_metrics(ideal_full)
        print(f"Ideal P_fact={pf0:.6f}, TVD(sqft, full)={tvd(ideal_sqft, ideal_full):.2e}", flush=True)

        tq_rec = build_recyc_oracle_ideal(N, a, n, t)
        rec_steps = build_steps(tq_rec)
        nq_rec, ncl_rec = tq_rec.num_qubits, tq_rec.num_clbits
        ideal_rec = simulate(rec_steps, nq_rec, ncl_rec, "none", 0.0, readout=0.0, reset_p=0.0)
        print(f"Ideal TVD(recyc, full)={tvd(ideal_rec, ideal_full):.2e}", flush=True)

        def record(circ, ch, p, dist):
            pk, pu, pf = eval_metrics(dist)
            lo, hi = wilson(round(pf * N_SHOTS), N_SHOTS)
            rows.append({
                "N": N,
                "a": a,
                "t": t,
                "circuit": circ,
                "channel": ch,
                "p": p,
                "P_peak": round(pk, 6),
                "P_useful": round(pu, 6),
                "P_fact": round(pf, 6),
                "retained": round(pf / pf0, 6) if pf0 > 0 else "",
                "P_fact_lo": round(lo, 6),
                "P_fact_hi": round(hi, 6),
                "TVD_from_ideal": round(tvd(dist, ideal_full), 6),
            })
            print(f"  {circ:6s} {ch:16s} p={p:<7g} P_peak={pk:.4f} P_fact={pf:.4f} "
                  f"ret={pf/pf0:.3f} TVD={tvd(dist, ideal_full):.4f}", flush=True)

        # Gate channels (depolarizing & bitflip; recycled also gets reset_p = p)
        for ch in ("depolarizing", "bitflip"):
            for p in GATE_RATES:
                nm = aer_noise_model(ch, p)
                record("qft", ch, p, run_stage_dm(rho_cnt, tq_full, t, False, nm))
                record("sqft", ch, p, run_stage_dm(rho_cnt, tq_sqft, t, True, nm))
                dist_r = simulate(rec_steps, nq_rec, ncl_rec, ch, p, readout=0.0, reset_p=p)
                record("recyc", ch, p, dist_r)

        # Measurement-only channel (isolates mid-circuit readout + feedback cost)
        for p in MEAS_ONLY_RATES:
            record("qft", "measurement_only", p, flip(ideal_full, t, p))
            record("sqft", "measurement_only", p, flip(ideal_sqft, t, p))
            dist_r = simulate(rec_steps, nq_rec, ncl_rec, "none", 0.0, readout=p, reset_p=0.0)
            record("recyc", "measurement_only", p, dist_r)

    out_csv = os.path.join(OUT, "oracle_ideal_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}", flush=True)


if __name__ == "__main__":
    t_start = time.time()
    main()
    print(f"Total elapsed: {time.time() - t_start:.1f} s", flush=True)
