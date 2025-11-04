__global__ void tile_accumulate(float* globalOut, const float* globalIn, int N) {
  extern __shared__ float sdata[]; // dynamic shared memory (per-block)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int lane = threadIdx.x % warpSize;

  // Load into shared memory tile (coalesced from global)
  if (tid < N) sdata[threadIdx.x] = globalIn[tid]; else sdata[threadIdx.x] = 0.0f;
  __syncthreads(); // ensure tile is ready for all threads

  // Parallel reduce within block (simple tree)
  for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    __syncthreads(); // synchronize at each reduction step
  }

  // Thread 0 atomically updates global accumulator
  if (threadIdx.x == 0) {
    atomicAdd(&globalOut[0], sdata[0]); // global atomic; may need __threadfence() when mixing with other scopes
  }
}