# Read counters (sampled every 20 ms)
sm_util, mem_bw_util, ipc, tensor_occ, temp = read_perf_counters()
# Classify workload quickly
if mem_bw_util > 0.85:           # memory-bound
    target_mem_f = mem_lut.high   # keep memory fast
    target_sm_f  = sm_lut.low     # slow SM to save energy
elif tensor_occ > 0.6 and ipc>threshold:
    target_sm_f  = sm_lut.high   # favor tensor throughput
    target_mem_f = mem_lut.med
else:                            # compute-bound
    target_sm_f  = sm_lut.med
    target_mem_f = mem_lut.med
# Enforce thermal headroom and apply with hysteresis
apply_freqs_if_stable(target_sm_f, target_mem_f, temp)