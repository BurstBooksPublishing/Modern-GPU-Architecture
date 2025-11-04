#include 
// compute occupancy fraction (0..1)
double occupancy(int R_SM, int r_warp, int S_SM, int s_block,
                 int W_per_block, int W_max) {
  int max_warps_by_regs = R_SM / r_warp;            // integer division
  int max_blocks_by_shm = S_SM / s_block;
  int max_warps_by_shm = max_blocks_by_shm * W_per_block;
  int active_warps = std::min(max_warps_by_regs, max_warps_by_shm);
  if (active_warps > W_max) active_warps = W_max;
  return double(active_warps) / double(W_max);
}
// Example: use real device constants when available.