# Simple credit calculator for link tuning
def required_credits(bw_bytes_s, rtt_s, flit_size=256, margin=1.2):
    bdp = bw_bytes_s * rtt_s                      # bytes in flight
    credits = int((bdp / flit_size) * margin + 0.999)  # ceil with margin
    buf_bytes = credits * flit_size
    return credits, buf_bytes

# Example: 50 GB/s, 1us RTT
credits, buf = required_credits(50e9, 1e-6, flit_size=256, margin=1.25)
print(f"credits={credits} flits, buffer={buf/1024:.1f} KiB")  # debug output