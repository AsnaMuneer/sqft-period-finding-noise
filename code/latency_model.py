"""Makespan and idle-exposure model for the inverse-QFT stage (oracle excluded).

Pure Python, no Qiskit. Three designs on t counting qubits:
  full : deferred-form controlled phases (CP = 2 CX + 1q), then swap layer (3 CX per swap)
  sqft : hardware semiclassical form: H, classically controlled 1q phases, measure;
         feedback latency L between a measurement and any gate it conditions
  rec  : one ancilla, per round reset -> H -> conditioned phases -> H -> measure
Durations (ns) and the feedback latency L are parameters.
ASAP schedule. Idle exposure per qubit = (last end - first start) - busy time.
Proxy fidelity w = prod over gates (1 - eps) * prod over qubits (1 - (1 - exp(-idle/T2))/2).
"""
import math

def schedule(ops, nq, L, d):
    """ops: list of dicts with keys name, qubits, dur, deps (indices of ops whose end + L
    must precede this op; feedback=True marks classical conditioning). Returns per-op start/end."""
    ready = [0.0] * nq
    start, end = [], []
    for i, op in enumerate(ops):
        s = max(ready[q] for q in op["qubits"])
        for dep in op.get("deps", []):
            s = max(s, end[dep] + L)
        e = s + op["dur"]
        for q in op["qubits"]:
            ready[q] = e
        start.append(s)
        end.append(e)
    return start, end

def full_ops(t, d):
    ops = []
    for j in reversed(range(t)):
        ops.append(dict(name="H", qubits=[j], dur=d["1q"]))
        for k in range(j + 1, t):
            ops.append(dict(name="CP", qubits=[k, j], dur=2 * d["2q"] + d["1q"]))
    for j in range(t // 2):
        ops.append(dict(name="SWAP", qubits=[j, t - 1 - j], dur=3 * d["2q"]))
    for j in range(t):
        ops.append(dict(name="M", qubits=[j], dur=d["meas"]))
    return ops

def sqft_ops(t, d):
    ops, meas = [], {}
    for j in reversed(range(t)):
        ops.append(dict(name="H", qubits=[j], dur=d["1q"]))
        for k in range(j + 1, t):
            ops.append(dict(name="P_cond", qubits=[j], dur=d["1q"], deps=[meas[k]], feedback=True))
        ops.append(dict(name="M", qubits=[j], dur=d["meas"]))
        meas[j] = len(ops) - 1
    return ops

def rec_ops(t, d):
    # one ancilla = qubit 0
    ops, meas = [], {}
    for j in reversed(range(t)):
        ops.append(dict(name="RESET", qubits=[0], dur=d["reset"]))
        ops.append(dict(name="H", qubits=[0], dur=d["1q"]))
        for k in range(j + 1, t):
            ops.append(dict(name="P_cond", qubits=[0], dur=d["1q"], deps=[meas[k]], feedback=True))
        ops.append(dict(name="H", qubits=[0], dur=d["1q"]))
        ops.append(dict(name="M", qubits=[0], dur=d["meas"]))
        meas[j] = len(ops) - 1
    return ops

def gate_fidelity(ops, eps1, eps2):
    w = 1.0
    for op in ops:
        if op["name"] in ("H", "P_cond", "RESET"):
            w *= 1 - eps1
        elif op["name"] == "CP":
            w *= (1 - eps2) ** 2
        elif op["name"] == "SWAP":
            w *= (1 - eps2) ** 3
    return w

def analyse(name, ops, nq, L, d, T2, eps1, eps2):
    start, end = schedule(ops, nq, L, d)
    makespan = max(end)
    w_gate = gate_fidelity(ops, eps1, eps2)
    w_idle = 1.0
    idle_total = 0.0
    for q in range(nq):
        idx = [i for i, op in enumerate(ops) if q in op["qubits"]]
        if not idx:
            continue
        window = max(end[i] for i in idx) - min(start[i] for i in idx)
        busy = sum(ops[i]["dur"] for i in idx)
        idle = window - busy
        idle_total += idle
        w_idle *= 1 - (1 - math.exp(-idle / T2)) / 2
    return dict(design=name, makespan_ns=makespan, idle_ns=idle_total,
                w_gate=w_gate, w_idle=w_idle, w=w_gate * w_idle,
                two_q=sum(2 if o["name"] == "CP" else 3 if o["name"] == "SWAP" else 0 for o in ops))

def layers(ops, start):
    """Number of distinct start times and the largest number of gates starting together."""
    from collections import Counter
    c = Counter(round(s, 6) for s in start)
    return len(c), max(c.values())

if __name__ == "__main__":
    d = dict(**{"1q": 35.0, "2q": 300.0, "meas": 800.0, "reset": 800.0})
    for t in (8, 10):
        print(f"\n== t = {t} ==")
        for L in (0.0, 500.0, 1000.0, 2000.0):
            for name, ops, nq in (("full", full_ops(t, d), t), ("sqft", sqft_ops(t, d), t), ("rec", rec_ops(t, d), 1)):
                r = analyse(name, ops, nq, L, d, T2=100e3, eps1=1e-3, eps2=1e-2)
                print(f"L={L:6.0f} {name:5s} makespan={r['makespan_ns']/1e3:7.2f} us  CX-equiv={r['two_q']:3d}  "
                      f"w={r['w']:.4f} (gate {r['w_gate']:.3f}, idle {r['w_idle']:.3f})")
