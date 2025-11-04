__global__ void persistent_kernel(int N, int chunk, int *tasks, /* ... */) {
  extern __shared__ int sdata[]; // per-block scratch (optional)
  // Local worker id (warp or thread-level work mapping).
  int tid = threadIdx.x;
  // Single atomic counter in global memory (address passed as tasks[0]).
  int base;
  while (true) {
    // Each atomicAdd returns previous value; fetch a chunk of work.
    base = atomicAdd(&tasks[0], chunk); // amortize atomic per chunk
    if (base >= N) break;                // no more work
    int upper = min(base + chunk, N);
    for (int i = base + tid; i < upper; i += blockDim.x) {
      // process task i (compute, memory ops, etc.) -- keep independent.
      process_task(i); // inline compute or call device function.
    }
    // optional: __syncthreads() if using block-local accumulators.
  }
}