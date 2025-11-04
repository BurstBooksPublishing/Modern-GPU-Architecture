# Simple HBM bandwidth/power model -- change params for scenarios.
def hbm_stats(channels=8, width_bits=128, rate_hz=3.2e9, eta=0.9, eb_j_per_bit=2e-12):
    # B: bytes/sec from Eq. (1)
    B = eta * channels * width_bits * rate_hz / 8.0
    # P: power for data transfers
    power_W = eb_j_per_bit * channels * width_bits * rate_hz
    return B, power_W

# Example: evaluate a stack
bw, p = hbm_stats(channels=8, width_bits=128, rate_hz=4.0e9, eta=0.88, eb_j_per_bit=1.5e-12)
print(f"Bandwidth (GB/s): {bw/1e9:.2f}, Data power (W): {p:.2f}")  # quick check