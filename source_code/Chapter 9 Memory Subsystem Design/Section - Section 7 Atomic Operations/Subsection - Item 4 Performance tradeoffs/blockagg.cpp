__global__ void block_sum(const float* A, float* out, int N){
  extern __shared__ float s[];               // per-block shared memory
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int lane = threadIdx.x;
  float acc = 0.0f;
  // local accumulate (strided)
  for(int i=tid; i0; offset>>=1){
    if(lane < offset) s[lane] += s[lane+offset];
    __syncthreads();
  }
  if(lane==0) atomicAdd(out, s[0]); // single global atomic per block
}