#!/usr/bin/env python3
"""
sqft_benchmark.py -- noise benchmark for period finding, N = 15, a = 7.

Three counting-register designs share one 4-qubit target register:
  qft    : 8 counting qubits, standard inverse QFT with swaps (baseline)
  sqft   : 8 counting qubits, semiclassical inverse QFT (Griffiths-Niu)
  recyc  : 1 counting qubit, reset and reused for 8 rounds (qubit recycling)

Bit convention for all designs: x = sum_j c[j] * 2^j, c[0] least significant.
The swap-free semiclassical designs store qubit/round j in c[t-1-j], which
restores the same convention. This is checked against the qft baseline by a
total-variation test before any noisy run.

Period-consistent outcomes for r = 4 and t = 8 are x in {0, 64, 128, 192}.

Requirements: qiskit>=1.2, qiskit-aer>=0.15, numpy, matplotlib.
Usage:
  python sqft_benchmark.py --smoke              # quick check, 512 shots, 1 seed
  python sqft_benchmark.py --shots 4096 --seeds 10 --out results
"""
import argparse
import csv
import json
import math
import os
from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

# ----------------------------------------------------------------------------
# Problem instance
# ----------------------------------------------------------------------------
N_MOD = 15
A = 7
N_COUNT = 8                       # counting precision t = 8 (= 2b for b = 4)
N_TARGET = 4                      # 2^4 = 16 > 15
PERIOD = 4                        # order of 7 mod 15
GOOD = [s * 2**N_COUNT // PERIOD for s in range(PERIOD)]                  # 0, 64, 128, 192
USEFUL = [s * 2**N_COUNT // PERIOD for s in range(PERIOD) if gcd(s, PERIOD) == 1]  # 64, 192
N_OTHER = 2**N_COUNT - len(GOOD)

CIRCUITS = ["qft", "sqft", "recyc"]
LABELS = {"qft": "Full QFT", "sqft": "Semiclassical QFT", "recyc": "Recycled SQFT"}

# Transpile target. Noise is applied to these physical gates only.
BASIS = ["rz", "sx", "x", "cx"]
ONE_Q = ["sx", "x"]
TWO_Q = ["cx"]

SWEEPS = {
    "depolarizing": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "bitflip": [1e-3, 3e-3, 5e-3, 1e-2],
    "readout": [0.01, 0.02, 0.05],
}

TVD_TOLERANCE = 0.02


# ----------------------------------------------------------------------------
# Modular multiplication: exact permutation unitaries
# ----------------------------------------------------------------------------
def controlled_power_gate(exponent):
    """Controlled unitary |y> -> |M y mod 15>, with M = a^exponent mod 15.

    Returns None when M == 1 (identity), so no gate is added. For a = 7 this
    is non-trivial only for exponent 1 (M = 7) and exponent 2 (M = 4).
    The unitary is a permutation of 16 basis states: y < 15 maps to M*y mod 15,
    and 15 is fixed.
    """
    M = pow(A, exponent, N_MOD)
    if M == 1:
        return None
    P = np.zeros((2**N_TARGET, 2**N_TARGET))
    for y in range(2**N_TARGET):
        z = (M * y) % N_MOD if y < N_MOD else y
        P[z, y] = 1.0
    return UnitaryGate(P).control(1)


# ----------------------------------------------------------------------------
# Circuits
# ----------------------------------------------------------------------------
def build_circuit(kind):
    t = N_COUNT

    if kind == "recyc":
        anc = QuantumRegister(1, "anc")
        tgt = QuantumRegister(N_TARGET, "tgt")
        c = ClassicalRegister(t, "c")
        qc = QuantumCircuit(anc, tgt, c)
        qc.x(tgt[0])                                   # target = |1>
        for j in reversed(range(t)):                   # round j uses power 2^j, j = t-1 first
            qc.reset(anc[0])
            qc.h(anc[0])
            g = controlled_power_gate(2**j)
            if g is not None:
                qc.append(g, [anc[0], *tgt])
            for k in range(j + 1, t):                  # feedback from earlier rounds
                with qc.if_test((c[t - 1 - k], 1)):
                    qc.p(-math.pi / 2 ** (k - j), anc[0])
            qc.h(anc[0])
            qc.measure(anc[0], c[t - 1 - j])           # reversed label, see module docstring
        return qc

    cnt = QuantumRegister(t, "cnt")
    tgt = QuantumRegister(N_TARGET, "tgt")
    c = ClassicalRegister(t, "c")
    qc = QuantumCircuit(cnt, tgt, c)
    qc.x(tgt[0])                                       # target = |1>
    qc.h(cnt)
    for j in range(t):
        g = controlled_power_gate(2**j)
        if g is not None:
            qc.append(g, [cnt[j], *tgt])

    if kind == "qft":
        qc.append(QFTGate(t).inverse(), cnt)
        qc.measure(cnt, c)
    elif kind == "sqft":
        for j in reversed(range(t)):
            qc.h(cnt[j])
            for k in range(j + 1, t):
                with qc.if_test((c[t - 1 - k], 1)):
                    qc.p(-math.pi / 2 ** (k - j), cnt[j])
            qc.measure(cnt[j], c[t - 1 - j])           # reversed label, see module docstring
    else:
        raise ValueError(kind)
    return qc


# ----------------------------------------------------------------------------
# Noise and simulation
# ----------------------------------------------------------------------------
def noise_model(channel, p):
    if channel == "none" or p == 0:
        return None
    nm = NoiseModel()
    if channel == "depolarizing":
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ONE_Q)
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), TWO_Q)
    elif channel == "bitflip":
        nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ONE_Q)
    elif channel == "readout":
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    else:
        raise ValueError(channel)
    return nm


def run_once(kind, noise, shots, seed):
    qc = build_circuit(kind)
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=seed)
    sim = AerSimulator(noise_model=noise) if noise is not None else AerSimulator()
    result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    return result.get_counts()


# ----------------------------------------------------------------------------
# Post-processing and metrics
# ----------------------------------------------------------------------------
def rev_int(x):
    return int(format(x, f"0{N_COUNT}b")[::-1], 2)


def factor_found(x):
    """Continued-fraction post-processing with the standard checks."""
    if x == 0:
        return False
    r = Fraction(x, 2**N_COUNT).limit_denominator(N_MOD).denominator
    if r % 2 or pow(A, r, N_MOD) != 1:
        return False
    h = pow(A, r // 2, N_MOD)
    if h in (1, N_MOD - 1):
        return False
    f1 = gcd(h - 1, N_MOD)
    f2 = gcd(h + 1, N_MOD)
    p, q = f1, f2
    return p not in (1, N_MOD) and q not in (1, N_MOD) and p * q == N_MOD


def score(counts):
    shots = sum(counts.values())
    k_peak = k_useful = k_fact = k_rev = 0
    for key, n in counts.items():
        x = int(key, 2)                 # Qiskit keys are MSB-first: c[t-1] ... c[0]
        if x in GOOD:
            k_peak += n
        if x in USEFUL:
            k_useful += n
        if factor_found(x):
            k_fact += n
        if rev_int(x) in GOOD:          # diagnostic for a bit-order mismatch
            k_rev += n
    p_peak = k_peak / shots
    signal = p_peak / len(GOOD)
    noise = (1 - p_peak) / N_OTHER
    snr = signal / noise if noise > 0 else float("inf")
    return dict(shots=shots, k_peak=k_peak, k_useful=k_useful,
                k_fact=k_fact, k_rev=k_rev, snr=snr)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def tvd(a, b):
    """Total-variation distance between two count dictionaries."""
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


# ----------------------------------------------------------------------------
# Resource accounting (noise-free transpile of the same circuits)
# ----------------------------------------------------------------------------
def resources():
    rows = []
    for kind in CIRCUITS:
        qc = build_circuit(kind)
        tqc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
        ops = tqc.count_ops()
        counting = 1 if kind == "recyc" else N_COUNT
        rows.append(dict(circuit=kind, counting_qubits=counting,
                         total_qubits=qc.num_qubits,
                         cx=ops.get("cx", 0), depth=tqc.depth(),
                         measurements=ops.get("measure", 0),
                         resets=ops.get("reset", 0)))
    return rows


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_ideal_histogram(ideal_counts, outdir, shots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(ideal_counts.items(), key=lambda kv: -kv[1])[:12]
    labels = [format(int(k, 2), f"0{N_COUNT}b") for k, _ in items]
    values = [v / shots for _, v in items]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Outcome (MSB first)")
    ax.set_title("Ideal full-QFT outcomes", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ideal_histogram.pdf"))
    plt.close(fig)


def plot_sweeps(summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    for ch in SWEEPS:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ideal_rows = [r for r in summary if r["channel"] == "none"]
        for kind in CIRCUITS:
            pts = [r for r in summary if r["circuit"] == kind and r["channel"] == ch]
            pts += [r for r in ideal_rows if r["circuit"] == kind]   # p = 0 anchor
            pts.sort(key=lambda r: r["p"])
            x = [r["p"] for r in pts]
            y = [r["P_fact"] for r in pts]
            lo = [r["P_fact"] - r["P_fact_lo"] for r in pts]
            hi = [r["P_fact_hi"] - r["P_fact"] for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", markersize=3, capsize=2,
                        label=LABELS[kind])
        ax.set_xlabel(f"{ch} error rate $p$")
        ax.set_ylabel("Factoring success probability")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"sweep_factor_{ch}.pdf"))
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, 512 shots, ideal plus one noisy point")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.smoke:
        args.shots, args.seeds = 512, 1
    os.makedirs(args.out, exist_ok=True)

    # 1. Equivalence check: each swap-free design must match the qft baseline.
    print("Equivalence check against the full-QFT baseline (ideal, 20000 shots):")
    ref_counts = run_once("qft", None, 20000, seed=7)
    equivalence = {"qft": 0.0}
    all_ok = True
    for kind in ["sqft", "recyc"]:
        d = tvd(ref_counts, run_once(kind, None, 20000, seed=7))
        equivalence[kind] = d
        status = "OK" if d < TVD_TOLERANCE else "FAIL"
        if d >= TVD_TOLERANCE:
            all_ok = False
        print(f"  {kind:6s} TVD vs qft = {d:.4f}  [{status}]")
    if not all_ok:
        print("WARNING: a design does not match the baseline. Do not trust its results.")

    # 2. Settings
    settings = [("none", 0.0)]
    for ch, grid in SWEEPS.items():
        settings += [(ch, p) for p in grid]
    if args.smoke:
        settings = [("none", 0.0), ("depolarizing", 1e-3)]

    # 3. Runs
    ideal = {}
    run_rows = []
    for kind in CIRCUITS:
        for ch, p in settings:
            noise = noise_model(ch, p)
            for seed in range(args.seeds):
                counts = run_once(kind, noise, args.shots, seed=1000 + seed)
                row = score(counts)
                row.update(circuit=kind, channel=ch, p=p, seed=seed)
                run_rows.append(row)
            sub = [r for r in run_rows if r["circuit"] == kind and r["channel"] == ch and r["p"] == p]
            tot = sum(r["shots"] for r in sub)
            pk = sum(r["k_peak"] for r in sub) / tot
            print(f"{kind:6s} {ch:12s} p={p:<7g} P_peak={pk:.4f}")
            if ch == "none":
                ideal[kind] = dict(P_peak=pk,
                                   P_useful=sum(r["k_useful"] for r in sub) / tot,
                                   P_fact=sum(r["k_fact"] for r in sub) / tot,
                                   P_rev=sum(r["k_rev"] for r in sub) / tot)

    print("\nIdeal diagnostics (expected: P_peak ~ 1 for qft, P_useful ~ 0.5, P_fact ~ 0.5, P_rev ~ 0.25)")
    for kind, d in ideal.items():
        print(f"  {kind:6s} " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))
        if d["P_rev"] > d["P_peak"] + 0.2:
            print(f"  WARNING {kind}: reversed-bit peaks dominate; check the bit-order convention.")

    # 4. Summaries with pooled Wilson intervals and across-seed spread.
    summary = []
    groups = {}
    for r in run_rows:
        groups.setdefault((r["circuit"], r["channel"], r["p"]), []).append(r)
    for (kind, ch, p), rs in groups.items():
        shots = sum(r["shots"] for r in rs)
        kp = sum(r["k_peak"] for r in rs)
        kf = sum(r["k_fact"] for r in rs)
        fracs = [r["k_fact"] / r["shots"] for r in rs]
        lo_f, hi_f = wilson(kf, shots)
        lo_p, hi_p = wilson(kp, shots)
        summary.append(dict(
            circuit=kind, channel=ch, p=p, runs=len(rs), shots=shots,
            P_peak=kp / shots, P_peak_lo=lo_p, P_peak_hi=hi_p,
            P_fact=kf / shots, P_fact_lo=lo_f, P_fact_hi=hi_f,
            P_fact_sd_across_seeds=float(np.std(fracs, ddof=1)) if len(fracs) > 1 else 0.0,
            SNR_mean=float(np.mean([r["snr"] for r in rs])),
        ))

    # 5. Write outputs
    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    res = resources()
    with open(os.path.join(args.out, "resources.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    with open(os.path.join(args.out, "ideal_diagnostics.json"), "w") as f:
        json.dump({"ideal": ideal, "equivalence_tvd": equivalence,
                   "tvd_tolerance": TVD_TOLERANCE,
                   "qiskit_note": "record qiskit and qiskit-aer versions in the paper"},
                  f, indent=2)

    plot_ideal_histogram(ref_counts, args.out, 20000)
    if not args.smoke:
        plot_sweeps(summary, args.out)

    print(f"\nWrote results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
