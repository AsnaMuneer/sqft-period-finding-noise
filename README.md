# Noise Sensitivity of Semiclassical and Qubit-Recycled QFT in Period Finding

Exact density-matrix simulation code, analytical validation scripts, plotting routines, and numerical results for a noise study of three counting-register designs used in period finding (the quantum stage of Shor's algorithm):

- **Full QFT (baseline):** Standard inverse QFT with swaps.
- **Semiclassical QFT (SQFT):** Griffiths–Niu inverse QFT in deferred-measurement form (and hardware classically conditioned form for the latency/idle study).
- **Recycled SQFT:** One counting qubit reused across rounds, with mid-circuit measurement, reset, and classical feedback.

The instances are $N = 15$ ($a = 7$) and $N = 21$ ($a = 2$), with transpiled resource scaling evaluated up to $N = 51$. The manuscript source is in `main_jpdc.tex` (submitted to *Journal of Parallel and Distributed Computing*)[cite: 1].

---

## Repository layout

```text
code/                       scripts (run from inside this folder)
  noise.py                  shared noise channels, branching simulator, metrics[cite: 17]
  sqft_benchmark.py         N = 15 circuit builders, bit convention, Monte Carlo benchmark[cite: 23]
  recyc_exact.py            exact recycled-design simulator (oracle-noise model), N = 15[cite: 18]
  sweep_exact.py            earlier exact sweep (bit-flip on single-qubit gates only)[cite: 24]
  sweep_final.py            final exact sweep for the paper, N = 15 (full QFT, SQFT)[cite: 25]
  attach_final.py           noise-attachment checks used during development[cite: 12]
  resource_scaling.py       transpiled resource counts for N = 15, 21, 35, 39, 51[cite: 20]
  recyc_scaling.py          recycled design at N = 15 and 21, oracle-noise model[cite: 19]
  oracle_ideal_study.py     oracle-ideal model: noise on the QFT stage, measurements, resets
  check_bound.py            numerical check of Proposition 3 (analytical TVD bound)[cite: 13]
  check_peaks.py            per-peak breakdown of the recycled design at N = 15[cite: 14]
  latency_model.py          ASAP schedule makespan and idle-exposure model[cite: 16]
  sim_idle.py               exact density-matrix simulation with gate noise + T1/T2 idle decoherence[cite: 22]
  run_grid2.py              feedback-latency L and T1/T2 sweep runner (Table 5)[cite: 21]
  make_plots.py             generates publication PDFs from results_final/ and results_recyc/
  grid2_results.txt         saved output from run_grid2.py (Table 5)[cite: 15, 21]
  requirements.txt          pinned Python packages (Python 3.12, Qiskit 1.4.6, Aer 0.15.1)
  figures, charts/          generated vector PDF figures included in main_jpdc.tex[cite: 1]
results_final/              final exact sweep (N = 15, full QFT and SQFT)[cite: 25]
results_exact/              earlier exact sweep (superseded)[cite: 24]
results_recyc/              recycled design, exact, N = 15 (oracle-noise model)[cite: 18]
results_resources/          resource counts for N = 15, 21, 35, 39, 51[cite: 20]
results_recyc_scaling/      recycled design at N = 15 and N = 21 (oracle-noise model, reduced sweep)[cite: 19]
results_oracle_ideal/       oracle-ideal study at N = 15 and N = 21
