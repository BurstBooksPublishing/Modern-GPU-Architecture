extern "C" __global__ void warpStochAvg(float *out, uint64_t seed, int Nbits){
  int wid = (blockIdx.x*blockDim.x + threadIdx.x) / 32; // warp id
  uint32_t lane = threadIdx.x & 31;
  uint64_t state = seed ^ (wid<<32) ^ lane; // simple per-lane seed
  unsigned int pop = 0;
  for(int b=0;b> 7; state ^= state << 17;
    unsigned int bit = (state & 1u);
    pop += bit;
  }
  float p_hat = float(pop)/float(Nbits); // decode
  if(lane==0) out[wid]=p_hat; // one value per warp
}