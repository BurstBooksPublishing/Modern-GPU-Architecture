float4 sampleTexture(const MipMap& mip, float u, float v) {
  float lambda = computeLOD(u,v,mip);             // per-quad derivative estimator
  int level = floor(lambda);
  float f = lambda - level;                       // blend factor between levels
  // fetch and bilinear on level
  float4 B0 = bilinearFetch(mip[level], u, v);    // fetch 4 texels, interpolate
  float4 B1 = bilinearFetch(mip[level+1], u, v);  // next level
  return lerp(B0, B1, f);                         // trilinear blend
}