#!/usr/bin/env python3
"""Regenerate all N = 15 figures for the JPDC manuscript from exact results.

Reads:
  - results_final/exact_summary.csv  (Full QFT and Semiclassical QFT)
  - results_recyc/recyc_exact.csv    (Recycled SQFT)

Writes to "figures, charts/" (matching the LaTeX \\includegraphics paths):
  - ideal_histogram.pdf
  - sweep_factor_depolarizing.pdf
  - sweep_factor_bitflip.pdf
  - sweep_factor_readout.pdf
  - sensitivity_rz.pdf
"""
import csv
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_CSV = os.path.join(HERE, "results_final", "exact_summary.csv")
RECYC_CSV = os.path.join(HERE, "results_recyc", "recyc_exact.csv")
FIG_DIR = os.path.join(HERE, "figures, charts")
N_SHOTS = 4096
N_COUNT = 8


def wilson(k, n=N_SHOTS, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return centre - half, centre + half


def load_csv_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({
                "circuit": r["circuit"],
                "channel": r["channel"],
                "p": float(r["p"]),
                "P_peak": float(r["P_peak"]),
                "P_fact": float(r["P_fact"]),
                "P_fact_lo": float(r["P_fact_lo"]),
                "P_fact_hi": float(r["P_fact_hi"]),
            })
        return rows


def fallback_rows():
    """Exact values from Tables 2 and 3 of the manuscript if CSVs are unavailable."""
    raw = [
        # Ideal anchor (p = 0)
        ("qft", "none", 0.0, 1.0, 0.5),
        ("sqft", "none", 0.0, 1.0, 0.5),
        ("recyc", "none", 0.0, 1.0, 0.5),
        # Depolarizing (Table 2 & Table 3)
        ("qft", "depolarizing", 1e-4, 0.9952, 0.4868),
        ("qft", "depolarizing", 3e-4, 0.9857, 0.4695),
        ("qft", "depolarizing", 1e-3, 0.9532, 0.4534),
        ("qft", "depolarizing", 3e-3, 0.8665, 0.4463),
        ("qft", "depolarizing", 1e-2, 0.6241, 0.3600),
        ("sqft", "depolarizing", 1e-4, 0.9966, 0.4873),
        ("sqft", "depolarizing", 3e-4, 0.9897, 0.4708),
        ("sqft", "depolarizing", 1e-3, 0.9661, 0.4573),
        ("sqft", "depolarizing", 3e-3, 0.9020, 0.4582),
        ("sqft", "depolarizing", 1e-2, 0.7116, 0.3913),
        ("recyc", "depolarizing", 1e-4, 0.9995, 0.5025),
        ("recyc", "depolarizing", 3e-4, 0.9985, 0.5061),
        ("recyc", "depolarizing", 1e-3, 0.9950, 0.5089),
        ("recyc", "depolarizing", 3e-3, 0.9851, 0.4982),
        ("recyc", "depolarizing", 1e-2, 0.9512, 0.4797),
        # Bit-flip (Table 2 & Table 3)
        ("qft", "bitflip", 1e-3, 0.9490, 0.4656),
        ("qft", "bitflip", 3e-3, 0.8554, 0.4550),
        ("qft", "bitflip", 5e-3, 0.7720, 0.4295),
        ("qft", "bitflip", 1e-2, 0.6006, 0.3703),
        ("sqft", "bitflip", 1e-3, 0.9589, 0.4651),
        ("sqft", "bitflip", 3e-3, 0.8822, 0.4530),
        ("sqft", "bitflip", 5e-3, 0.8121, 0.4268),
        ("sqft", "bitflip", 1e-2, 0.6622, 0.3668),
        ("recyc", "bitflip", 1e-3, 0.9900, 0.5024),
        ("recyc", "bitflip", 3e-3, 0.9704, 0.4888),
        ("recyc", "bitflip", 5e-3, 0.9512, 0.4798),
        ("recyc", "bitflip", 1e-2, 0.9048, 0.4604),
        # Bit-flip + rotation error (Table 2)
        ("qft", "bitflip +rz", 1e-3, 0.8810, 0.4516),
        ("qft", "bitflip +rz", 5e-3, 0.5420, 0.3469),
        ("qft", "bitflip +rz", 2e-2, 0.1205, 0.1526),
        ("sqft", "bitflip +rz", 1e-3, 0.9108, 0.4544),
        ("sqft", "bitflip +rz", 5e-3, 0.6340, 0.3575),
        ("sqft", "bitflip +rz", 2e-2, 0.1943, 0.1661),
        # Readout (Section 5.2 & Table 3)
        ("qft", "readout", 0.01, 0.9227, 0.4804),
        ("qft", "readout", 0.02, 0.8508, 0.4612),
        ("qft", "readout", 0.05, 0.6634, 0.4068),
        ("sqft", "readout", 0.01, 0.9227, 0.4804),
        ("sqft", "readout", 0.02, 0.8508, 0.4612),
        ("sqft", "readout", 0.05, 0.6634, 0.4068),
        ("recyc", "readout", 0.01, 0.9415, 0.4788),
        ("recyc", "readout", 0.02, 0.8858, 0.4584),
        ("recyc", "readout", 0.05, 0.7351, 0.4022),
    ]
    rows = []
    for circ, ch, p, pk, pf in raw:
        lo, hi = wilson(round(pf * N_SHOTS), N_SHOTS)
        rows.append({
            "circuit": circ,
            "channel": ch,
            "p": p,
            "P_peak": pk,
            "P_fact": pf,
            "P_fact_lo": lo,
            "P_fact_hi": hi,
        })
    return rows


def setup_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
    })


def plot_ideal_histogram(out_path):
    """Exact ideal distribution for N=15, a=7 (peaks at x in {0, 64, 128, 192} = 0.25)."""
    dist = np.zeros(2 ** N_COUNT)
    for x in (0, 64, 128, 192):
        dist[x] = 0.25

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.bar(np.arange(2 ** N_COUNT), dist, width=2.2, color="#1f77b4", edgecolor="#1f77b4")
    ax.set_xlim(-6, 261)
    ax.set_ylim(0, 0.30)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xticklabels(["0\n(00000000)", "64\n(01000000)", "128\n(10000000)", "192\n(11000000)", "255"],
                       fontsize=6)
    ax.set_xlabel("Counting register outcome $x$")
    ax.set_ylabel("Ideal probability")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_sweep(rows, series, out_path, xlabel, ylim=(0.32, 0.53)):
    fig, ax = plt.subplots(figsize=(3.4, 2.45))

    for spec in series:
        label, circ, ch, color, marker, ls = spec
        pts = [r for r in rows if r["circuit"] == circ and r["channel"] == ch]
        # Attach p = 0 ideal anchor
        ideal_pts = [r for r in rows if r["circuit"] == circ and r["channel"] == "none"]
        if ideal_pts:
            pts.append(ideal_pts[0])
        else:
            lo0, hi0 = wilson(round(0.5 * N_SHOTS), N_SHOTS)
            pts.append({"p": 0.0, "P_fact": 0.5, "P_fact_lo": lo0, "P_fact_hi": hi0})

        pts.sort(key=lambda r: r["p"])
        x = [r["p"] for r in pts]
        y = [r["P_fact"] for r in pts]
        yerr_lo = [max(0.0, r["P_fact"] - r["P_fact_lo"]) for r in pts]
        yerr_hi = [max(0.0, r["P_fact_hi"] - r["P_fact"]) for r in pts]

        ax.errorbar(
            x, y, yerr=[yerr_lo, yerr_hi],
            label=label, color=color, marker=marker, linestyle=ls,
            capsize=2.2, elinewidth=0.9, markeredgewidth=0.8
        )

    ax.axhline(0.5, color="0.45", linestyle=":", linewidth=0.9, label="Ideal ceiling (0.50)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Factoring probability $P_{\mathrm{fact}}$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()

    final_rows = load_csv_rows(FINAL_CSV)
    recyc_rows = load_csv_rows(RECYC_CSV)
    if final_rows and recyc_rows:
        rows = final_rows + recyc_rows
        print("Loaded rows from results_final/ and results_recyc/.")
    else:
        rows = fallback_rows()
        print("One or both CSVs missing/empty; using exact manuscript table values.")

    # Figure 1: Ideal distribution
    plot_ideal_histogram(os.path.join(FIG_DIR, "ideal_histogram.pdf"))

    # Figure 2: Depolarizing sweep (3 designs)
    plot_sweep(
        rows,
        [
            ("Full QFT", "qft", "depolarizing", "#1f77b4", "o", "-"),
            ("Semiclassical QFT", "sqft", "depolarizing", "#ff7f0e", "s", "-"),
            ("Recycled SQFT (5q)", "recyc", "depolarizing", "#2ca02c", "^", "-"),
        ],
        os.path.join(FIG_DIR, "sweep_factor_depolarizing.pdf"),
        r"Depolarizing error rate $p$",
        ylim=(0.33, 0.53),
    )

    # Figure 3: Bit-flip sweep (3 designs)
    plot_sweep(
        rows,
        [
            ("Full QFT", "qft", "bitflip", "#1f77b4", "o", "-"),
            ("Semiclassical QFT", "sqft", "bitflip", "#ff7f0e", "s", "-"),
            ("Recycled SQFT (5q)", "recyc", "bitflip", "#2ca02c", "^", "-"),
        ],
        os.path.join(FIG_DIR, "sweep_factor_bitflip.pdf"),
        r"Bit-flip error rate $p$",
        ylim=(0.33, 0.53),
    )

    # Figure 4: Readout sweep (3 designs; QFT and SQFT overlap identically)
    plot_sweep(
        rows,
        [
            ("Full QFT (analytic)", "qft", "readout", "#1f77b4", "o", "-"),
            ("Semiclassical QFT (analytic)", "sqft", "readout", "#ff7f0e", "s", "--"),
            ("Recycled SQFT (per-meas.)", "recyc", "readout", "#2ca02c", "^", "-"),
        ],
        os.path.join(FIG_DIR, "sweep_factor_readout.pdf"),
        r"Readout bit-flip probability $p$",
        ylim=(0.37, 0.53),
    )

    # Sensitivity plot: Bit-flip + Rz error
    plot_sweep(
        rows,
        [
            ("Full QFT", "qft", "bitflip", "#1f77b4", "o", "-"),
            ("Full QFT (+$R_z$)", "qft", "bitflip +rz", "#1f77b4", "o", "--"),
            ("Semiclassical QFT", "sqft", "bitflip", "#ff7f0e", "s", "-"),
            ("Semiclassical QFT (+$R_z$)", "sqft", "bitflip +rz", "#ff7f0e", "s", "--"),
        ],
        os.path.join(FIG_DIR, "sensitivity_rz.pdf"),
        r"Bit-flip error rate $p$",
        ylim=(0.12, 0.53),
    )

    print(f"Saved all PDF figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()