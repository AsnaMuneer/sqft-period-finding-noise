Noise sensitivity of semiclassical and qubit-recycled QFT in period finding

Exact density-matrix simulation code, analytical validation scripts, plotting routines, and numerical results for a noise study of three counting-register designs used in period finding (the quantum stage of Shor's algorithm):
- Full QFT (baseline): standard inverse QFT with swaps.
- Semiclassical QFT (SQFT): Griffiths–Niu inverse QFT in deferred-measurement form (and hardware classically conditioned form for the latency/idle study).
- Recycled SQFT: one counting qubit reused across rounds, with mid-circuit measurement, reset and classical feedback.

The instances are N = 15 (a = 7) and N = 21 (a = 2), with transpiled resource scaling evaluated up to N = 51. The paper is in main_jpdc.tex (submitted to Journal of Parallel and Distributed Computing).

Repository layout

code/                       scripts (run from inside this folder)
  noise.py                  shared noise channels, branching simulator, metrics
  sqft_benchmark.py         N = 15 circuit builders, bit convention, Monte Carlo benchmark
  recyc_exact.py            exact recycled-design simulator (oracle-noise model), N = 15
  sweep_exact.py            earlier exact sweep (bit-flip on single-qubit gates only)
  sweep_final.py            final exact sweep for the paper, N = 15 (full QFT, SQFT)
  attach_final.py           noise-attachment checks used during development
  resource_scaling.py       transpiled resource counts for N = 15, 21, 35, 39, 51
  recyc_scaling.py          recycled design at N = 15 and 21, oracle-noise model
  oracle_ideal_study.py     oracle-ideal model: noise on the QFT stage, measurements, resets
  check_bound.py            numerical check of Proposition 3 (analytical TVD bound)
  check_peaks.py            per-peak breakdown of the recycled design at N = 15
  latency_model.py          ASAP schedule makespan and idle-exposure model
  sim_idle.py               exact density-matrix simulation with gate noise + T1/T2 idle decoherence
  run_grid2.py              feedback-latency L and T1/T2 sweep runner (Table 5)
  make_plots.py             generates publication PDFs from results_final/ and results_recyc/
  grid2_results.txt         saved output from run_grid2.py (Table 5)
  requirements.txt          pinned Python packages
  figures, charts/          generated vector PDF figures included in main_jpdc.tex
results_final/              final exact sweep (N = 15, full QFT and SQFT)
results_exact/              earlier exact sweep (superseded)
results_recyc/              recycled design, exact, N = 15 (oracle-noise model)
results_resources/          resource counts for N = 15, 21, 35, 39, 51
results_recyc_scaling/      recycled design at N = 15 and 21 (oracle-noise model, reduced sweep)
results_oracle_ideal/       oracle-ideal study at N = 15 and 21

Environment

Python 3.12 with the versions in code/requirements.txt:
- qiskit 1.4.6
- qiskit-aer 0.15.1
- numpy, matplotlib

Set up:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt

Results were produced on macOS (Apple silicon). Exact density-matrix runs are CPU-bound and memory-light (< 1 GB RAM) for N = 15 and N = 21, though multi-round classical branching at N = 21 (2^10 = 1024 branches) takes ~1–3 hours depending on circuit decomposition.

Reproducing the results

Run every script from inside code/, because the scripts import each other by file name. Outputs are written relative to the folder the script is run from.

Result                                      Script                 Expected output                                  Approximate time
Final N = 15 sweep (full QFT, SQFT)         sweep_final.py         results_final/exact_summary.csv                  30–45 min
Earlier N = 15 sweep                        sweep_exact.py         results_exact/exact_summary.csv                  30–45 min
Recycled design, N = 15                     recyc_exact.py         results_recyc/recyc_exact.csv                    15 min
Resource counts                             resource_scaling.py    results_resources/resources.csv                  < 1 min
Recycled design, N = 15 & 21 (reduced)      recyc_scaling.py       results_recyc_scaling/recyc_scaling.csv          ~3 hours
Oracle-ideal study                          oracle_ideal_study.py  results_oracle_ideal/oracle_ideal_summary.csv    ~1.5 hours
Feedback latency & T1/T2 idle sweep         run_grid2.py           stdout / grid2_results.txt                       ~3 min
Proposition 3 analytical bound check        check_bound.py         stdout verification                              < 1 min
Paper figures (Figures 1–4 + Rz)            make_plots.py          figures, charts/*.pdf                            < 5 sec

Example:
cd code
python sweep_final.py

Noise models

The repository contains three noise and timing models. The paper reports them separately.

1. Oracle-noise model (recyc_exact.py, recyc_scaling.py, sweep_final.py). The modular-multiplication oracle is compiled to the basis gates {rz, sx, x, cx}, so its gates receive noise like every other gate. This is the primary model behind the N = 15 results (Tables 2–3).
2. Oracle-ideal model (oracle_ideal_study.py). The oracle is a single ideal unitary step. Noise acts only on the inverse-QFT stage gates, measurements, and resets. This model was introduced because, at N = 21, the oracle's compiled CX count is large enough that the oracle-noise model drives the output close to uniform at depolarizing rate 1e-3.
3. Feedback-latency and idle-decoherence model (latency_model.py, sim_idle.py, run_grid2.py). Evaluates the inverse-QFT stage on an ASAP schedule (1q: 35 ns, CX: 300 ns, measurement: 800 ns) with depolarizing gate errors (eps1 = 1e-3, eps2 = 1e-2) and T1 = T2 amplitude/phase damping during idle periods across classical feedback latencies L (Section 6, Table 5).

Gate and readout conventions (models 1 and 2):
- Depolarizing, convention B (checked against Qiskit Aer 0.15.1): single-qubit error probability 3p/4, uniform over X, Y, Z; two-qubit error probability 15p/16.
- Bit-flip: X with probability p on single-qubit gates; on CX, X on the control or the target with probability p/2 each.
- rz gates are error-free in the main results. A sensitivity variant adds bit-flip errors after rz.
- Readout is symmetric. For the full QFT and SQFT it is applied analytically to the final distribution. For the recycled design it is applied at each mid-circuit measurement.

Validation

- Ideal equivalence. For N = 15 the ideal distributions of all three designs agree to total-variation distance 0.000000 (2.01e-13 in the recycled branching simulator). For N = 21 the recycled design agrees with the full QFT to 4.53e-12.
- Analytical bound check. check_bound.py verifies that every noisy distribution in results_final/exact_summary.csv satisfies Proposition 3 (TVD <= 1 - w).
- Per-peak breakdown. check_peaks.py shows that the small excess of the recycled design's factoring probability above 0.5 at low noise comes from probability moving out of the non-factoring peak at x = 128 into the factoring peaks at x = 64 and x = 192.

Bit convention

x = sum_j c[j] 2^j, with c[0] the least significant bit. The swap-free designs store round j in classical bit c[t-1-j], which restores the same convention as the full QFT. The ideal equivalence test above checks this.

Metrics

- P_peak: probability of a period-consistent outcome (x such that x/2^t is near s/r).
- P_useful: probability of the outcomes with gcd(s, r) = 1, i.e. the ones that yield factors in the ideal case.
- P_fact: probability that continued-fraction post-processing of x/2^t yields a non-trivial factor of N.
- Intervals are 95% Wilson intervals at an assumed budget of 4096 shots. The reported probabilities are exact, not sampled.

Limitations

- Full three-design noisy comparisons are limited to N = 15 (a = 7); N = 21 (a = 2) evaluates the recycled design under the oracle-noise model and all three designs under the oracle-ideal model.
- The noise models are idealised. They do not include calibrated gate fidelities, crosstalk, or readout asymmetry.
- The modular-exponentiation oracle is a permutation unitary for the given modulus, not a general reversible modular-exponentiation circuit.
- No hardware results are included.

Citation

Citation details will be added on acceptance.

Licence

To be added before public release.
