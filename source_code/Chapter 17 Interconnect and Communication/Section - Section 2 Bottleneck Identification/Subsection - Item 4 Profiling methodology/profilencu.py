import subprocess,csv
# run ncu to collect metrics; replace 'my_app' and kernel selector as needed
subprocess.run(["ncu","--target-processes","all","--metrics",
                "sm__sass_thread_inst_executed_op_fma_sum,dram__bytes_read.sum",
                "--csv","--export","report.csv","./my_app"])
# parse CSV to extract metrics (very small parser)
with open("report.csv") as f:
    for row in csv.DictReader(f):
        # compute simple ratios; fields vary by GPU/tool
        insts = float(row.get("sm__sass_thread_inst_executed_op_fma_sum",0))
        bytes_read = float(row.get("dram__bytes_read.sum",1))
        ai = insts/bytes_read
        print(f"AI={ai:.3f}, FMA={insts:.0f}, DRAM_bytes={bytes_read:.0f}")