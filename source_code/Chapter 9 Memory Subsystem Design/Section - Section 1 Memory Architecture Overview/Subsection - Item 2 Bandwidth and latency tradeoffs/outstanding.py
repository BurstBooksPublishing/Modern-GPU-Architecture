def required_outstanding(bw_bytes_per_s, latency_s, line_size_bytes):
    # compute N = B * L / S
    return (bw_bytes_per_s * latency_s) / line_size_bytes

# Example: 256 GB/s, 100 ns, 128 B line
print(required_outstanding(256e9, 100e-9, 128))  # -> ~200