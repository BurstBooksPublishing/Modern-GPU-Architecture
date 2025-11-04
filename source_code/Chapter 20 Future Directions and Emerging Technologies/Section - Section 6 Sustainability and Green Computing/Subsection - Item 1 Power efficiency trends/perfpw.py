# Simple perf/watt model; adjust params for design exploration
def perf_per_watt(flops_per_cycle, freq_hz, C, V, alpha, static_W):
    P_dyn = alpha * C * (V**2) * freq_hz
    flops = flops_per_cycle * freq_hz
    return (flops/1e9) / (P_dyn + static_W)  # GFLOPS/W

# example points: tensor core (mixed) vs SM (FP32)
points = [
    {'name':'tensor_fp16','flops_per_cycle':512,'freq':1.5e9,'C':1e-9,'V':0.8,'alpha':0.2,'static':20},
    {'name':'sm_fp32','flops_per_cycle':256,'freq':1.5e9,'C':1.2e-9,'V':1.0,'alpha':0.3,'static':25},
]
for p in points:
    print(p['name'], perf_per_watt(p['flops_per_cycle'],p['freq'],p['C'],p['V'],p['alpha'],p['static']))