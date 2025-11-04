__global__ void producer_consumer(int *device_buf){
  __shared__ int sflag;            // SM-local synchronization
  if(threadIdx.x==0){
    // produce data in shared memory (fast, L1)
    sflag = 1;                     // release to block
    __threadfence_block();         // ensure block visibility (cheap)
    atomicExch(&device_buf[0], 42);// device-visible store
    __threadfence();               // ensure device-wide visibility (expensive)
  }
  __syncthreads();                 // SM-local barrier (fast)
  // consumer on same SM can read sflag; remote SMs require device fence semantics.
}