__global__ void prod_cons(int *data, int *flag) {
  if (threadIdx.x == 0) {
    data[0] = 42;                   // write data (may be buffered)
    __threadfence();               // ensure data visible to device
    atomicExch(flag, 1);           // release: publish by atomic flag
  } else if (threadIdx.x == 1) {
    while (atomicAdd(flag, 0) == 0) {} // spin on flag (atomic read)
    __threadfence();               // ensure subsequent loads see writes
    int v = data[0];               // now observe published data
  }
}