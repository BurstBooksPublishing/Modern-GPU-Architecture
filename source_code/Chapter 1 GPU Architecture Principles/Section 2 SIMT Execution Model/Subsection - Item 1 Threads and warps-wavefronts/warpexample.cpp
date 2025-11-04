__global__ void filter_kernel(float *data, int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // global thread id
  if (gid >= N) return;
  float v = data[gid];
  // Divergent predicate: different lanes may take different paths
  if (v > 0.0f) { v = sqrtf(v); } else { v = 0.0f; }
  // Warp-aware prefix (uses warp-synchronous logic) for reduction
  unsigned int lane = threadIdx.x & 31;
  for (int offset=1; offset<32; offset<<=1) {
    float n = __shfl_up_sync(0xFFFFFFFF, v, offset); // warp shuffle
    if (lane >= offset) v += n;
  }
  if (lane == 31) atomicAdd(&data[0], v); // accumulate per-warp sum
  data[gid] = v;
}