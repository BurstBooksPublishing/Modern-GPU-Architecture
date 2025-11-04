def required_warps(latency_cycles, inst_between_memops, warp_issue_rate=1.0):
    # compute minimal resident warps to hide latency (Equation 1)
    return max(1, latency_cycles * warp_issue_rate / inst_between_memops)

# example: 300-cycle latency, one mem op per 40 instructions
print(required_warps(300, 40))  # -> ~7.5 warps