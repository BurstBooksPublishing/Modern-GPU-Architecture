def roofline(C_peak, B, intensity):
    # C_peak: GFLOP/s, B: GB/s, intensity: FLOP/byte
    return min(C_peak, B * intensity)  # GFLOP/s

# Example: tensor core pool and HBM channel scaling
C_peak = 40000.0   # GFLOP/s (aggregate)
B = 1500.0         # GB/s (sustainable)
kernels = {'conv':8.0, 'gemm_small':64.0, 'spmv':0.5}
for name, I in kernels.items():
    print(name, roofline(C_peak, B, I))  # shows bottleneck per kernel