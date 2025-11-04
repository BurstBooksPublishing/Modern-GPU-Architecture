struct Ray { float3 o, d, invd; }; // ray origin, direction, inverse dir
struct Node { float3 min, max; int left, right; int triOffset, triCount; };

__device__ bool intersect_aabb(const Ray& r, const Node& n, float& tmin) {
  // branch-minimized slab test; returns hit distance in tmin
  float t0 = (n.min.x - r.o.x) * r.invd.x;
  float t1 = (n.max.x - r.o.x) * r.invd.x;
  float tmin_x = fminf(t0,t1), tmax_x = fmaxf(t0,t1);
  // repeat for y,z and combine (omitted for brevity)
  // final tmin assignment and test against tmax
  return /*true if intersects*/ true;
}

__global__ void traverse_kernel(const Node* bvh, const Triangle* tris, Ray* rays, Hit* hits, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  Ray r = rays[idx];
  int stack[64]; int sp = 0; stack[sp++] = 0; // root index
  float tmin;
  while (sp) {
    int nodeIdx = stack[--sp];
    const Node& node = bvh[nodeIdx];
    if (!intersect_aabb(r,node,tmin)) continue; // reject
    if (node.triCount) {
      // test triangles (triangle intersection code omitted)
    } else {
      // push children; push order can be near-first for early-out
      stack[sp++] = node.right; stack[sp++] = node.left;
    }
  }
  hits[idx] = /*closest hit record*/ {};
}