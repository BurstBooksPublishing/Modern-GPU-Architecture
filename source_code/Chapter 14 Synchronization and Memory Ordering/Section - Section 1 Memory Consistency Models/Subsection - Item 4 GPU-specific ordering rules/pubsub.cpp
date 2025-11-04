__global__ void publish_kernel(int *data, int *flag) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  // producer: write data then publish
  if (tid == 0) {
    data[0] = 42;                   // write payload to global memory
    __threadfence();               // ensure device visibility to L2
    atomicExch(flag, 1);           // publish (release) via atomic
  }
  // consumer: spin on flag then read
  if (tid == 1) {
    while (atomicAdd(flag, 0) == 0) { } // acquire via atomic read
    __threadfence();               // optional: ensure L1 sees updates
    int v = data[0];               // now guaranteed to observe 42
    // ... use v ...
  }
}