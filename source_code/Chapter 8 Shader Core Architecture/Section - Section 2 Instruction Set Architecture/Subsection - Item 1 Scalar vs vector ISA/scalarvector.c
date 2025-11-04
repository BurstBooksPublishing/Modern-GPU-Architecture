/* scalar kernel: each thread operates on one element (maps well to scalar ISA) */
__kernel void kernel_scalar(float *A, float *B, float *C, int N) {
  int i = get_global_id(0); if (i>=N) return; // per-thread control
  float a = A[i]; float b = B[i];
  C[i] = a*b + a; // scalar ALU ops
}

/* vector-packed kernel: groups 4 elements per work-item (maps to vector ISA) */
__kernel void kernel_vector(float4 *A4, float4 *B4, float4 *C4, int M) {
  int gid = get_global_id(0); if (gid>=M) return;
  float4 a = A4[gid]; float4 b = B4[gid];
  C4[gid] = a*b + a; // single vector instruction operates on 4 lanes
}