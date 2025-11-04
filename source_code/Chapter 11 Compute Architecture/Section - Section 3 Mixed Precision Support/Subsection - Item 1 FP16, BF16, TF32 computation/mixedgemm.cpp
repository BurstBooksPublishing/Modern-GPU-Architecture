__global__ void mixedGEMM(const __half *A, const __nv_bfloat16 *B, float *C, int M,int N,int K){
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  if(row>=M||col>=N) return;
  float acc = 0.0f;                      // accumulator in FP32
  for(int k=0;k float
    float b = __bfloat162float(B[k*N + col]);      // convert bfloat16 -> float
    acc += a * b;                                  // FP32 FMA on SM ALU or tensor core
  }
  C[row*N + col] = acc;
}