import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from sweep_exact import sqft_deferred, build_circuit, strip_measurements, metrics, tvd_arr
from sqft_benchmark import BASIS, N_COUNT, rev_int

RATES = [0.001, 0.005, 0.02, 0.05]


def model(p):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["sx", "x"])
    nm.add_all_qubit_quantum_error(
        pauli_error([("XI", p / 2), ("IX", p / 2), ("II", 1 - p)]), ["cx"])
    return nm


def exact(circ, reverse, nm):
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


designs = [("qft", build_circuit("qft"), False), ("sqft", sqft_deferred(), True)]
for name, circ, rev in designs:
    ideal = exact(circ, rev, NoiseModel())
    for p in RATES:
        dist = exact(circ, rev, model(p))
        pk, pu, pf = metrics(dist)
        print(f"{name:5s} bitflip (sx/x + cx) p={p:<6g} sum={dist.sum():.6f} "
              f"P_peak={pk:.4f} P_fact={pf:.4f} TVD={tvd_arr(dist, ideal):.4f}", flush=True)
