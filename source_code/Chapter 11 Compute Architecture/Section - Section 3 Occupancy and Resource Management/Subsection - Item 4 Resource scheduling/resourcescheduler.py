# Simple resource-aware scheduler emulator
def admit_blocks(R_SM, S_SM, B_max, blocks): 
    # blocks: list of (R_blk, S_blk, T_blk)
    admitted = []
    r_avail, s_avail, b_avail = R_SM, S_SM, B_max
    # sort by smaller combined resource to reduce fragmentation
    blocks_sorted = sorted(blocks, key=lambda b: (b[0]+b[1]))
    for R_blk, S_blk, T_blk in blocks_sorted:
        if R_blk <= r_avail and S_blk <= s_avail and b_avail > 0:
            admitted.append((R_blk, S_blk, T_blk))
            r_avail -= R_blk; s_avail -= S_blk; b_avail -= 1
    return admitted

def dispatch_warps(admitted, W_size=32):
    # round-robin warp issue with simple backpressure
    warps = []
    for R_blk, S_blk, T_blk in admitted:
        num_warps = (T_blk + W_size - 1) // W_size
        warps.extend([num_warps]*1)  # one entry per block (placeholder)
    return warps  # emulator uses warp counts for latency hiding