#!/usr/bin/env python3
"""Resource scaling for the three period-finding counting designs.

Designs compared (all for the same N and base a):
  - full:     standard inverse QFT on t = 2n counting qubits (with swaps)
  - deferred: semiclassical inverse QFT in deferred-measurement form (no swaps)
  - recycled: one counting qubit reused for all t rounds (mid-circuit reset)

Modular exponentiation is excluded from the counts. All three designs make
the same t = 2n controlled-modular-exponentiation calls, so the oracle cost
does not change the ranking. The oracle is an opaque unitary in this script.

Writes results_resources/resources.csv next to this script and prints a table.
Run with:  python resource_scaling.py   (inside the pinned .venv_old)
"""
import csv
import math
import os

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate

CASES = [(15, 7), (21, 2), (35, 2), (39, 2), (51, 2)]   # (N, a) with gcd(a, N) = 1
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_resources")
BASIS = ["rz", "sx", "x", "cx"]


def order(a, N):
    """Multiplicative order r of a mod N (smallest r with a^r = 1 mod N)."""
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


def full_inverse_qft(t):
    """Standard inverse QFT on t qubits, as a circuit (includes swaps)."""
    qc = QuantumCircuit(t)
    qc.append(QFTGate(t).inverse(), range(t))
    return qc


def deferred_inverse_qft(t):
    """Semiclassical inverse QFT in deferred-measurement form (no swaps).

    Matches sqft_benchmark.py / sweep_exact.py: for j in reversed(range(t)):
    H on qubit j, then CP(-pi / 2^(k-j)) from control k onto qubit j.
    """
    qc = QuantumCircuit(t)
    for j in reversed(range(t)):
        qc.h(j)
        for k in range(j + 1, t):
            qc.cp(-math.pi / 2 ** (k - j), k, j)
    return qc


def transpiled_counts(qc):
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
    ops = tqc.count_ops()
    return ops.get("cx", 0), tqc.depth()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f"{'N':>4} {'a':>3} {'r':>3} {'n':>2} {'design':<9} {'cnt':>4} {'qubits':>6} "
          f"{'modexp':>6} {'H':>4} {'CP':>4} {'swap':>4} {'reset':>5} {'meas':>4} "
          f"{'CX':>5} {'depth':>6}")
    for N, a in CASES:
        r = order(a, N)
        n = N.bit_length()          # target register size
        t = 2 * n                   # counting qubits
        cp_count = t * (t - 1) // 2
        swap_count = t // 2

        # Full QFT: t counting + n target qubits, swaps included
        cx_full, depth_full = transpiled_counts(full_inverse_qft(t))
        # Deferred semiclassical: same register sizes, no swaps
        cx_def, depth_def = transpiled_counts(deferred_inverse_qft(t))

        designs = [
            # name, counting qubits, total qubits, H, CP, swap, reset, measure, CX, depth
            ("full", t, t + n, t, cp_count, swap_count, 0, t, cx_full, depth_full),
            ("deferred", t, t + n, t, cp_count, 0, 0, t, cx_def, depth_def),
            # Recycled: CX and depth not transpiled (mid-circuit if_test ops);
            # the logical counts are exact.
            ("recycled", 1, 1 + n, t, cp_count, 0, t - 1, t, None, None),
        ]
        for name, cnt, total, h, cp, sw, rs, ms, cx, depth in designs:
            rows.append({
                "N": N, "a": a, "order_r": r, "n_target": n, "t_counting": t,
                "design": name, "counting_qubits": cnt, "total_qubits": total,
                "controlled_modexp_calls": t, "H_gates": h, "controlled_phase_gates": cp,
                "swap_gates": sw, "mid_circuit_resets": rs, "measurements": ms,
                "transpiled_cx": cx if cx is not None else "",
                "transpiled_depth": depth if depth is not None else "",
            })
            print(f"{N:>4} {a:>3} {r:>3} {n:>2} {name:<9} {cnt:>4} {total:>6} "
                  f"{t:>6} {h:>4} {cp:>4} {sw:>4} {rs:>5} {ms:>4} "
                  f"{(cx if cx is not None else '-'):>5} {(depth if depth is not None else '-'):>6}")

    path = os.path.join(OUT, "resources.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
