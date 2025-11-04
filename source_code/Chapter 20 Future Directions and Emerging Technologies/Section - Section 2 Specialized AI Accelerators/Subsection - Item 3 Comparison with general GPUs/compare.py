# compute achieved FLOPS using roofline model (toy values)
def achieved_flops(F_peak, bandwidth, intensity):
    return min(F_peak, intensity * bandwidth)

# GPU: flexible SIMT with tensor cores (values in TFLOPS, TB/s)
gpu = {'F_peak': 100.0, 'bandwidth': 1.0}   # 100 TFLOPS, 1 TB/s
tpu = {'F_peak': 250.0, 'bandwidth': 0.8}   # 250 TFLOPS, 0.8 TB/s

# transformer attention intensity ~ 40 FLOP/byte; GEMM ~ 200 FLOP/byte
print(achieved_flops(gpu['F_peak'], gpu['bandwidth'], 40))  # GPU attention
print(achieved_flops(tpu['F_peak'], tpu['bandwidth'], 40))  # TPU attention