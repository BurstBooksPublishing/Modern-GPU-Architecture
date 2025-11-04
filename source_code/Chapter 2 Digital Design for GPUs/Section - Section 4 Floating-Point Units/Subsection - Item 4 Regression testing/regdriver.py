#!/usr/bin/env python3
import subprocess, concurrent.futures, os, sys
SEEDS = range(100,110)            # 10 randomized runs
SIM_CMD = "./simulator --top gpu_tb --seed {seed}"  # placeholder simulator
ARTIFACT_DIR = "reg_results"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run(seed):
    cmd = SIM_CMD.format(seed=seed).split()
    env = os.environ.copy(); env['SEED']=str(seed)
    out = open(f"{ARTIFACT_DIR}/run_{seed}.log","wb")
    rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
    out.close()
    return seed, rc

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as e:
    futures = [e.submit(run,s) for s in SEEDS]
    failed = []
    for f in concurrent.futures.as_completed(futures):
        seed, rc = f.result()
        if rc != 0:
            failed.append(seed)

if failed:
    print("Regression failed for seeds:", failed); sys.exit(1)
print("All runs passed.")