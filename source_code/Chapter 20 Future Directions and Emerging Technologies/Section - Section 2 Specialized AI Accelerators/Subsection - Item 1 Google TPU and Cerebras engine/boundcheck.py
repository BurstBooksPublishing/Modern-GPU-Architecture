def bottleneck(P_peak, B_mem, arithmetic_intensity):
    # P_peak in FLOPS, B_mem in bytes/s, arithmetic_intensity in FLOP/byte
    sustained = min(P_peak, B_mem * arithmetic_intensity)
    return ("compute_bound" if sustained == P_peak else "memory_bound", sustained)

# example usage: TPU-like and Cerebras-like parameters (illustrative)
print(bottleneck(1e15, 1e12, 800))  # high intensity -> compute bound
print(bottleneck(1e15, 2e11, 100))  # lower BW or intensity -> memory bound