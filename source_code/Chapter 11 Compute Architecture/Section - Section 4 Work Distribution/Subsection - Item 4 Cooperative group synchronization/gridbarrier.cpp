#include 
using namespace cooperative_groups;

__global__ void kernel_with_sync(int *global_bar, int num_blocks) {
  auto tb = this_thread_block();
  // block-local work uses shared memory synchronization
  __syncthreads(); // SM-level barrier
  // try hardware grid sync if launched cooperatively
  #ifdef __CUDA_ARCH__
  grid_group g = this_grid(); // may require cooperative launch
  g.sync(); // blocks all threads in grid (heap semantics)
  #else
  // Fallback: scalable two-phase atomic barrier (simplified)
  if (threadIdx.x==0) atomicAdd(global_bar, 1); // leader increments
  __syncthreads();
  while (*global_bar < num_blocks) { /* spin-wait*/ } // poll with backoff
  __syncthreads(); // ensure all threads see release
  #endif
}