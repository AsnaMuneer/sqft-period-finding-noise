#!/usr/bin/env python3
"""Recycled-design scaling across N, exact density-matrix simulation.

Generalises sqft_benchmark.py (build_circuit "recyc" and "qft") and
recyc_exact.py (exact simulator, noise models, metrics) to any (N, a).

Conventions are the same as the N = 15 study:
  - x = sum_j c[j] 2^j, c[0] least significant
  - round order, feedback rule, and reversed bit labels as in sqft_benchmark.py
  - Aer depolarizing convention chosen by choose_conventions() (1q=B, 2q=B)
  - readout applied per measurement inside the recycled circuit

Safety gate: for every case, the ideal recycled distribution must match the
ideal full-QFT distribution (TVD < 1e-6), computed from the same oracle.
If it does not, the script stops before any noisy run.

Scope: exact density-matrix noise is run for CASES. Full-QFT noisy runs are
NOT included for N > 15 (about 15 qubits: too large for an exact density
matrix on a laptop). Their resource counts come from resource_scaling.py.

Writes results_recyc_scaling/recyc_scaling.csv next to this script.
Run with:  python recyc_scaling.py   (inside the pinned .venv_old)
"""
import csv
import math
import os
import time
from fractions import Fraction
from math import gcd

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import DensityMatrix, Kraus, Statevector

from recyc_exact import (PROJ0, PROJ1, RESET, add, build_steps, choose_conventions,
                         kraus_1q, kraus_2q)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_recyc_scaling")
BASIS = ["rz", "sx", "x", "cx"]
N_SHOTS = 4096
# Reduced set for the first N = 21 run (timing check). Restore the full sweep
# once the per-setting time is known:
#   readout [0.01, 0.02, 0.05]; depolarizing [1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
#   bitflip [1e-3, 3e-3, 5e-3, 1e-2]
SWEEPS = {
    "readout": [],
    "depolarizing": [1e-3, 1e-2],
    "bitflip": [1e-2],
}
CASES = [(15, 7), (21, 2)]     # exact noisy sweeps; (15, 7) is the regression check
TVD_GATE = 1e-6


def order(a, N):
    r, x = 1, a % N
    while x != 1:
        x = (x * a) % N
        r += 1
    return r


class Instance:
    def __init__(self, N, a):
        self.N, self.a = N, a
        self.n = N.bit_length()          # target register size
        self.t = 2 * self.n              # counting precision
        self.r = order(a, N)
        self.M = 2 ** self.t
        pts = {int(round(j * self.M / self.r)) % self.M for j in range(self.r)}
        self.good = sorted(pts)
        self.useful = sorted({int(round(j * self.M / self.r)) % self.M
                              for j in range(self.r) if gcd(j, self.r) == 1})

    def cpow(self, e):
        """Controlled permutation |y> -> |a^e y mod N>, identity when a^e = 1 mod N."""
        mult = pow(self.a, e, self.N)
        if mult == 1:
            return None
        dim = 2 ** self.n
        P = np.zeros((dim, dim))
        for y in range(dim):
            z = (mult * y) % self.N if y < self.N else y
            P[z, y] = 1.0
        return UnitaryGate(P).control(1)

    def recyc(self):
        """Mirror of sqft_benchmark.py build_circuit('recyc')."""
        t = self.t
        anc = QuantumRegister(1, "anc")
        tgt = QuantumRegister(self.n, "tgt")
        c = ClassicalRegister(t, "c")
        qc = QuantumCircuit(anc, tgt, c)
        qc.x(tgt[0])
        for j in reversed(range(t)):
            qc.reset(anc[0])
            qc.h(anc[0])
            g = self.cpow(2 ** j)
            if g is not None:
                qc.append(g, [anc[0], *tgt])
            for k in range(j + 1, t):
                with qc.if_test((c[t - 1 - k], 1)):
                    qc.p(-math.pi / 2 ** (k - j), anc[0])
            qc.h(anc[0])
            qc.measure(anc[0], c[t - 1 - j])
        return qc

    def full(self, measure=True):
        """Mirror of sqft_benchmark.py build_circuit('qft')."""
        t = self.t
        cnt = QuantumRegister(t, "cnt")
        tgt = QuantumRegister(self.n, "tgt")
        c = ClassicalRegister(t, "c")
        qc = QuantumCircuit(cnt, tgt, c)
        qc.x(tgt[0])
        qc.h(cnt)
        for j in range(t):
            g = self.cpow(2 ** j)
            if g is not None:
                qc.append(g, [cnt[j], *tgt])
        qc.append(QFTGate(t).inverse(), cnt)
        if measure:
            qc.measure(cnt, c)
        return qc

    def full_ideal(self):
        """Exact ideal distribution of the full QFT over the counting register."""
        sv = Statevector(self.full(measure=False))
        return np.asarray(sv.probabilities([*range(self.t)]), dtype=float)

    def factor_found(self, x):
        if x == 0:
            return False
        r = Fraction(x, self.M).limit_denominator(self.N).denominator
        if r % 2 or pow(self.a, r, self.N) != 1:
            return False
        h = pow(self.a, r // 2, self.N)
        if h in (1, self.N - 1):
            return False
        p, q = gcd(h - 1, self.N), gcd(h + 1, self.N)
        return p not in (1, self.N) and q not in (1, self.N) and p * q == self.N

    def metrics(self, dist):
        pk = float(sum(dist[x] for x in self.good))
        pu = float(sum(dist[x] for x in self.useful))
        pf = float(sum(dist[x] for x in range(len(dist)) if self.factor_found(x)))
        return pk, pu, pf


def simulate(steps, nq, ncl, channel, p, readout, conv1, conv2):
    """Mirror of recyc_exact.py simulate, with the counting size taken from ncl."""
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

    dist = np.zeros(2 ** ncl)
    for rec, rho in branches.items():
        x = sum(bit << i for i, bit in enumerate(rec))
        dist[x] += float(np.real(np.trace(rho.data)))
    return dist


def tvd(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def main():
    os.makedirs(OUT, exist_ok=True)
    conv1, conv2 = choose_conventions()
    rows = []
    for N, a in CASES:
        inst = Instance(N, a)
        print(f"\n=== N={N}, a={a}, r={inst.r}, n={inst.n}, t={inst.t}, "
              f"good={inst.good}, useful={inst.useful}", flush=True)

        tq = transpile(inst.recyc(), basis_gates=BASIS, optimization_level=1, seed_transpiler=1)
        steps = build_steps(tq)
        nq, ncl = tq.num_qubits, tq.num_clbits
        print(f"recycled circuit: {nq} qubits, {ncl} classical bits, {len(steps)} steps", flush=True)

        full_ideal = inst.full_ideal()
        t0 = time.time()
        ideal = simulate(steps, nq, ncl, "none", 0.0, 0.0, conv1, conv2)
        gate_tv = tvd(ideal, full_ideal)
        print(f"ideal ({time.time() - t0:.0f} s), sum={ideal.sum():.6f}, "
              f"TVD vs full QFT = {gate_tv:.2e}", flush=True)
        if gate_tv > TVD_GATE:
            raise RuntimeError(f"N={N}: recycled ideal does not match full QFT "
                               f"(TVD {gate_tv:.2e}). Stop; do not use these numbers.")

        pk0, pu0, pf0 = inst.metrics(full_ideal)
        rows.append(dict(N=N, a=a, order_r=inst.r, n_target=inst.n, t_counting=inst.t,
                         circuit="full_ideal", channel="none", p=0.0,
                         P_peak=round(pk0, 6), P_useful=round(pu0, 6), P_fact=round(pf0, 6),
                         P_fact_ideal=round(pf0, 6), retained=1.0,
                         P_fact_lo="", P_fact_hi="", TVD_from_ideal=0.0,
                         sum=round(float(full_ideal.sum()), 6), shots_assumed=N_SHOTS,
                         time_s=""))

        settings = [("readout", p) for p in SWEEPS["readout"]]
        settings += [("depolarizing", p) for p in SWEEPS["depolarizing"]]
        settings += [("bitflip", p) for p in SWEEPS["bitflip"]]
        for channel, p in settings:
            t0 = time.time()
            if channel == "readout":
                dist = simulate(steps, nq, ncl, "none", 0.0, p, conv1, conv2)
            else:
                dist = simulate(steps, nq, ncl, channel, p, 0.0, conv1, conv2)
            pk, pu, pf = inst.metrics(dist)
            lo, hi = wilson(round(pf * N_SHOTS), N_SHOTS)
            tv = tvd(dist, ideal)
            dt = time.time() - t0
            rows.append(dict(N=N, a=a, order_r=inst.r, n_target=inst.n, t_counting=inst.t,
                             circuit="recyc", channel=channel, p=p,
                             P_peak=round(pk, 6), P_useful=round(pu, 6), P_fact=round(pf, 6),
                             P_fact_ideal=round(pf0, 6),
                             retained=round(pf / pf0, 6) if pf0 else "",
                             P_fact_lo=round(lo, 6), P_fact_hi=round(hi, 6),
                             TVD_from_ideal=round(tv, 6), sum=round(float(dist.sum()), 6),
                             shots_assumed=N_SHOTS, time_s=round(dt)))
            print(f"  {channel:12s} p={p:<7g} sum={dist.sum():.6f} P_peak={pk:.4f} "
                  f"P_useful={pu:.4f} P_fact={pf:.4f} TVD={tv:.4f} ({dt:.0f} s)", flush=True)

    path = os.path.join(OUT, "recyc_scaling.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {path}", flush=True)


if __name__ == "__main__":
    main()
