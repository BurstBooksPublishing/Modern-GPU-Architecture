extern "C" __global__ void refitLeaves(AABB* leafAabbs, const Tri* tris, int leafCount){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=leafCount) return;
  // recompute leaf AABB from triangle vertices
  Tri t = tris[i]; // triangle assigned to leaf i
  float3 v0=t.v0, v1=t.v1, v2=t.v2;
  AABB a; a.min = fminf(fminf(v0,v1), v2); a.max = fmaxf(fmaxf(v0,v1), v2);
  leafAabbs[i] = a; // write updated leaf bbox
}
__global__ void refitInternal(AABB* nodes, int levelStart, int nodeCount){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>=nodeCount) return;
  int idx = levelStart + i;
  AABB left = nodes[2*i + levelStart]; // layout-dependent index calc
  AABB right = nodes[2*i+1 + levelStart];
  nodes[idx] = unionAABB(left,right); // parent bbox = union of children
}