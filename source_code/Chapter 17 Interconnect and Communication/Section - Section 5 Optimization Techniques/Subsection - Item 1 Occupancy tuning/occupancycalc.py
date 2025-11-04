def compute_occupancy(R_SM, S_SM, T_SM, B_SM, warp_size,
                      regs_per_thread, shared_per_block, threads_per_block):
    # blocks constrained by each resource
    b_regs = R_SM // (regs_per_thread * threads_per_block)  # register limit
    b_sh   = S_SM // shared_per_block                       # shared mem limit
    b_thr  = T_SM // threads_per_block                      # thread limit
    b_res  = min(b_regs, b_sh, b_thr, B_SM)                 # resident blocks
    warps_res = b_res * (threads_per_block // warp_size)    # resident warps
    warps_max = T_SM // warp_size
    return warps_res / warps_max
# Example: R_SM=65536 regs, S_SM=98304 bytes, T_SM=2048 threads, B_SM=16 blocks, warp_size=32