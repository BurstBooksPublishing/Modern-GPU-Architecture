import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, \
                   nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)  # single-GPU test

def sample(duration_s=10, interval_s=0.05):
    t0 = time.time()
    rows=[]
    while time.time() - t0 < duration_s:
        p = nvmlDeviceGetPowerUsage(h) / 1000.0  # mW->W
        u = nvmlDeviceGetUtilizationRates(h).gpu  # percent
        m = nvmlDeviceGetUtilizationRates(h).memory  # percent
        rows.append((time.time(), p, u, m))
        time.sleep(interval_s)
    return rows

data = sample(30, 0.02)  # 30s trace at 20ms intervals
# serialize data for offline regression and counter fusion