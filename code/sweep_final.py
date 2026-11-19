"""Final exact sweep for the paper: full QFT and deferred semiclassical QFT, N = 15.

Noise model: sx, x depolarizing or bit-flip; cx depolarizing or X on either qubit;
rz error-free (main results). Readout is applied analytically. Shared channels,
readout and metrics helpers come from noise.py.
"""
import csv
import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit_aer
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from noise import aer_noise_model, flip, tvd, wilson
from sqft_benchmark import (BASIS, GOOD, N_COUNT, SWEEPS, USEFUL,
                            build_circuit, rev_int)
from sweep_exact import metrics, sqft_deferred, strip_measurements

OUT = "results_final"
N_SHOTS = 4096          # shots assumed for sampling intervals
RZ_RATES = [0.001, 0.005, 0.02]


def exact(circ, reverse, nm):
    """Exact probabilities over the classical outcome x (c[0] least significant)."""
    body = strip_measurements(circ)
    tq = transpile(body, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    tq.save_probabilities_dict(list(range(N_COUNT)), label="pr")
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    raw = sim.run(tq, shots=1).result().data(0)["pr"]
    dist = np.zeros(2 ** N_COUNT)
    for k, v in raw.items():
        x = k if isinstance(k, int) else int(k, 2)
        dist[rev_int(x) if reverse else x] += v
    return dist


def plot_series(rows, series, fname, xlabel):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for label, name, ch in series:
        pts = [r for r in rows if r["circuit"] == name and r["channel"] in (ch, "none")]
        pts.sort(key=lambda r: r["p"])
        x = [r["p"] for r in pts]
        y = [r["P_fact"] for r in pts]
        lo = [r["P_fact"] - r["P_fact_lo"] for r in pts]
        hi = [r["P_fact_hi"] - r["P_fact"] for r in pts]
        ax.errorbar(x, y, yerr=[lo, hi], marker="o", markersize=3, capsize=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Factoring success probability")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def plot_ideal(dist, fname):
    order = np.argsort(-dist)[:12]
    labels = [format(int(x), f"0{N_COUNT}b") for x in order]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(range(len(order)), dist[order])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Outcome (MSB first)")
    ax.set_title("Ideal full-QFT outcomes", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    designs = [
        ("qft", "standard", build_circuit("qft"), False),
        ("sqft", "deferred", sqft_deferred(), True),
    ]

    ideal = {}
    for name, form, circ, rev in designs:
        t0 = time.time()
        ideal[name] = exact(circ, rev, NoiseModel())
        print(f"{name} ideal done ({time.time() - t0:.0f} s), sum={ideal[name].sum():.6f}",
              flush=True)
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
                         sum=round(float(dist.sum()), 6), shots_assumed=N_SHOTS))
        print(f"{name:5s} {channel:14s} p={p:<7g} P_peak={pk:.4f} "
              f"P_fact={pf:.4f} TVD={tv:.4f}", flush=True)

    for name, form, circ, rev in designs:
        add(name, form, "none", 0.0, ideal[name])
        for p in SWEEPS["readout"]:
            add(name, form, "readout", p, flip(ideal[name], N_COUNT, p))
        for p in SWEEPS["depolarizing"]:
            add(name, form, "depolarizing", p,
                exact(circ, rev, aer_noise_model("depolarizing", p)))
        for p in SWEEPS["bitflip"]:
            add(name, form, "bitflip", p, exact(circ, rev, aer_noise_model("bitflip", p)))
        # Sensitivity only: bit-flip with rz errors added
        for p in RZ_RATES:
            add(name, form, "bitflip +rz", p,
                exact(circ, rev, aer_noise_model("bitflip", p, rz=True)))

    with open(os.path.join(OUT, "exact_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot_ideal(ideal["qft"], "ideal_histogram.pdf")
    plot_series(rows, [("Full QFT", "qft", "depolarizing"),
                       ("Semiclassical QFT", "sqft", "depolarizing")],
                "sweep_factor_depolarizing.pdf", r"depolarizing error rate $p$")
    plot_series(rows, [("Full QFT", "qft", "bitflip"),
                       ("Semiclassical QFT", "sqft", "bitflip")],
                "sweep_factor_bitflip.pdf", r"bit-flip error rate $p$")
    plot_series(rows, [("Full QFT", "qft", "readout"),
                       ("Semiclassical QFT", "sqft", "readout")],
                "sweep_factor_readout.pdf", r"readout error rate $p$")
    plot_series(rows, [("Full QFT", "qft", "bitflip"),
                       ("Full QFT, +rz", "qft", "bitflip +rz"),
                       ("Semiclassical QFT", "sqft", "bitflip"),
                       ("Semiclassical QFT, +rz", "sqft", "bitflip +rz")],
                "sensitivity_rz.pdf", r"bit-flip error rate $p$")

    meta = dict(
        qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
        python=platform.python_version(),
        method="exact density-matrix probabilities (no sampling, no seeds)",
        shots_assumed_for_intervals=N_SHOTS,
        noise_model=("sx, x: depolarizing or X-flip; cx: two-qubit depolarizing or "
                     "single-qubit X on either qubit; rz error-free (main results). "
                     "Readout: symmetric, applied exactly to the ideal distribution."),
        semiclassical_form="deferred measurement (all measurements at the end)",
        bit_convention="x = sum_j c[j] 2^j, c[0] least significant",
        good_outcomes=GOOD, useful_outcomes=USEFUL,
    )
    with open(os.path.join(OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote results to {os.path.abspath(OUT)}", flush=True)


if __name__ == "__main__":
    main()
