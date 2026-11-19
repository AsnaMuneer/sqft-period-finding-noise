#!/usr/bin/env python3
"""Per-peak breakdown of the N = 15 recycled design (exact).

Shows where probability goes when noise pushes P_fact above 0.5.
Run in ~/Desktop/RSA with .venv_old, in a second Terminal window while the
N = 21 run is going. It takes about three minutes and writes nothing to disk.
"""
from qiskit import transpile

from sqft_benchmark import BASIS, GOOD, USEFUL, build_circuit, factor_found
from recyc_exact import build_steps, choose_conventions, simulate

FACTOR_SET = [x for x in range(256) if factor_found(x)]


def report(label, dist, ideal):
    print(f"\n{label}")
    for x in GOOD:
        tag = "  (useful)" if x in USEFUL else ""
        print(f"  x={x:3d}{tag:11s} ideal={ideal[x]:.5f}  noisy={dist[x]:.5f}  "
              f"change={dist[x] - ideal[x]:+.5f}")
    p_peak = sum(dist[x] for x in GOOD)
    p_useful = sum(dist[x] for x in USEFUL)
    p_fact = sum(dist[x] for x in FACTOR_SET)
    print(f"  P_peak={p_peak:.5f}  P_useful={p_useful:.5f}  P_fact={p_fact:.5f}")
    extra = [(x, round(dist[x], 5)) for x in FACTOR_SET if x not in GOOD and dist[x] > 1e-6]
    print(f"  factorable outcomes outside the peaks with mass: {extra if extra else 'none'}")
    gained = [(x, round(dist[x] - ideal[x], 5)) for x in range(256)
              if x not in GOOD and dist[x] - ideal[x] > 1e-4]
    print(f"  non-peak outcomes that gained more than 1e-4: {gained if gained else 'none'}")


def main():
    conv1, conv2 = choose_conventions()
    tq = transpile(build_circuit("recyc"), basis_gates=BASIS,
                   optimization_level=1, seed_transpiler=1)
    steps = build_steps(tq)
    nq, ncl = tq.num_qubits, tq.num_clbits

    ideal = simulate(steps, nq, ncl, "none", 0.0, 0.0, conv1, conv2)
    print(f"ideal P_fact = {sum(ideal[x] for x in FACTOR_SET):.5f}")
    report("ideal (for reference)", ideal, ideal)

    for p in (1e-4, 1e-3, 1e-2):
        dist = simulate(steps, nq, ncl, "depolarizing", p, 0.0, conv1, conv2)
        report(f"depolarizing p={p:g}", dist, ideal)


if __name__ == "__main__":
    main()
