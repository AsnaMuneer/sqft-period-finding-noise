import time
import sim_idle
from sim_idle import simulate, full_program, sqft_program, metrics

t, e1, e2 = 8, 1e-3, 1e-2
T1s = [20e3, 50e3, 100e3]
Ls = [0.0, 500.0, 1000.0, 2000.0, 5000.0]
t0 = time.time()
for T1 in T1s:
    sim_idle.T1 = T1
    T2 = T1
    d_f, T_f = simulate(full_program(t), t, 0.0, T2, e1, e2)
    pf, pff = metrics(d_f, t)
    print(f"\nT1 = T2 = {T1/1e3:.0f} us   full: makespan {T_f/1e3:.2f} us  P_peak={pf:.4f}  P_fact={pff:.4f}", flush=True)
    for L in Ls:
        d_s, T_s = simulate(sqft_program(t), t, L, T2, e1, e2)
        ps, pss = metrics(d_s, t)
        print(f"  L={L/1e3:5.2f} us  sqft: makespan {T_s/1e3:6.2f} us  P_peak={ps:.4f}  P_fact={pss:.4f}  "
              f"(full - sqft = {pff - pss:+.4f})", flush=True)
print(f"\nelapsed {time.time()-t0:.0f} s", flush=True)
