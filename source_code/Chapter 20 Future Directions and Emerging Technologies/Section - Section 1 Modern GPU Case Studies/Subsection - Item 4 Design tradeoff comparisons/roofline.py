def roofline_bound(P_peak, bandwidth, intensity):
    """Return achievable FLOP/s (min of compute and memory roof)."""
    # P_peak: FLOP/s, bandwidth: bytes/s, intensity: FLOP/byte
    return min(P_peak, bandwidth * intensity)

# Example compare: Hopper-like vs RDNA3-like (numbers illustrative)
hopper = roofline_bound(1.2e14, 1.2e12, 30)   # 120 TFLOPS, 1.2 TB/s, intensity 30
rdna3  = roofline_bound(8.0e13, 0.7e12, 30)   # 80 TFLOPS, 0.7 TB/s, intensity 30
print(hopper, rdna3)  # numeric bound outputs