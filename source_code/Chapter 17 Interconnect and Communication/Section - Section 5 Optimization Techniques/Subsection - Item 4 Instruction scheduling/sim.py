# Simple warp scheduler: ready_queue holds (warp_id, instr_latency)
ready_queue = [(0, 1), (1, 400), (2, 2)]  # (id, latency)
cycle = 0
while ready_queue:
    # pick first ready warp (round-robin) -- real design uses age + priorities
    warp_id, lat = ready_queue.pop(0)
    # issue instruction: if long latency, reinsert after latency cycles
    if lat > 16:
        # simulate long-latency stall before next ready instruction
        ready_queue.append((warp_id, 1))  # next instruction will be short
        cycle += 1  # issue cost
    else:
        cycle += 1  # complete short op quickly
    # basic bookkeeping -- real hardware tracks many more states
print("Simulated cycles:", cycle)  # // small model output