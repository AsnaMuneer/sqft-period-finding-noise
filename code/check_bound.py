"""Check the noise bound (Proposition 3 of the paper) against the exact sweep.

Run from code/ after sweep_final.py has written results_final/exact_summary.csv.
For each depolarizing, bit-flip, bit-flip + rz and readout row, the bound is
TVD <= 1 - w, where w is the probability that no noise event occurs, computed
from the gate counts of the transpiled circuit (rows for the full QFT and SQFT).
Readout uses 1 - (1 - p)^t. Prints each row and whether the exact TVD is below the bound.
"""
import csv

from qiskit import transpile

from sqft_benchmark import BASIS, N_COUNT, build_circuit
from sweep_exact import sqft_deferred, strip_measurements

CSV_PATH = "results_final/exact_summary.csv"


def gate_counts(circ):
    body = strip_measurements(circ)
    tq = transpile(body, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    counts = {"1q": 0, "2q": 0, "rz": 0}
    for inst in tq.data:
        name = inst.operation.name
        if name in ("sx", "x"):
            counts["1q"] += 1
        elif name == "cx":
            counts["2q"] += 1
        elif name == "rz":
            counts["rz"] += 1
    return counts


def bound(channel, p, c):
    """Return 1 - w for the channel, or None if the row is not a noise channel."""
    if channel == "depolarizing":
        e1, e2, er = 3 * p / 4, 15 * p / 16, 0.0
    elif channel == "bitflip":
        e1, e2, er = p, p, 0.0
    elif channel == "bitflip +rz":
        e1, e2, er = p, p, p
    elif channel == "readout":
        return 1 - (1 - p) ** N_COUNT
    else:
        return None
    w = (1 - e1) ** c["1q"] * (1 - e2) ** c["2q"] * (1 - er) ** c["rz"]
    return 1 - w


def main():
    designs = {"qft": build_circuit("qft"), "sqft": sqft_deferred()}
    counts = {k: gate_counts(v) for k, v in designs.items()}
    for k, c in counts.items():
        print(f"{k}: noisy 1q = {c['1q']}, 2q = {c['2q']}, rz = {c['rz']}")

    all_ok = True
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row["channel"] == "none":
                continue
            p = float(row["p"])
            b = bound(row["channel"], p, counts[row["circuit"]]) if row["circuit"] in counts else None
            if b is None:
                continue
            tvd = float(row["TVD_from_ideal"])
            ok = tvd <= b + 1e-6
            all_ok &= ok
            print(f"{row['circuit']:5s} {row['channel']:14s} p={p:<7g} "
                  f"TVD={tvd:.6f} bound={b:.6f} {'ok' if ok else 'VIOLATED'}")
    print("all rows within bound" if all_ok else "SOME ROWS VIOLATE THE BOUND")


if __name__ == "__main__":
    main()
