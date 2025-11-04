__global__ void binPrimitives(const Primitive *prims, int n, Bin *bins, AABB centroidBounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // compute bin index from centroid (simple linear mapping)
  float t = (prims[i].centroid.x - centroidBounds.min.x) / centroidBounds.extent.x;
  int b = min(max(int(t * NUM_BINS), 0), NUM_BINS-1);
  // atomic updates to bin AABB and count (reduced with warp-level ops in tuned code)
  atomicMin(&bins[b].aabb.min.x, prims[i].aabb.min.x);
  atomicMax(&bins[b].aabb.max.x, prims[i].aabb.max.x);
  atomicAdd(&bins[b].count, 1);
}