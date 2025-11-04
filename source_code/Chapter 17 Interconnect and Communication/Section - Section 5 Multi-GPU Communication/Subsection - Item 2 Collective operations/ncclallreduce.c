#include 
#include 
#include 

int main(int argc,char** argv){
  int rank=0, nranks=4; // assign ranks/size beforehand
  size_t N = 1<<20; // elements per GPU
  float *d_buf; cudaMalloc(&d_buf,N*sizeof(float)); // device buffer
  cudaStream_t s; cudaStreamCreate(&s); // per-rank stream

  ncclComm_t comm;
  ncclCommInitRank(&comm,nranks,/*ncclId*/*(ncclUniqueId*)argv,rank);

  // launch collective on stream; NCCL handles NVLink/PCIe/GPUDirect internally
  ncclAllReduce((const void*)d_buf, (void*)d_buf, N, ncclFloat, ncclSum, comm, s);

  cudaStreamSynchronize(s); // ensure completion before using results
  ncclCommDestroy(comm);
  cudaFree(d_buf); cudaStreamDestroy(s);
  return 0;
}