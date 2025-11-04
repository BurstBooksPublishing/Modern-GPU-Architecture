const float sampleOffsets[4][2] = {{0.25f,0.25f},{0.75f,0.25f},{0.25f,0.75f},{0.75f,0.75f}}; // 4x rotated-grid
for (int px = xmin; px <= xmax; ++px) {
  vec3 accumColor = vec3(0.0f);
  int covered = 0;
  for (int s = 0; s < 4; ++s) {
    float sx = px + sampleOffsets[s][0];
    float sy = py + sampleOffsets[s][1];
    if (edgeA.eval(sx,sy) >= 0 && edgeB.eval(sx,sy) >= 0 && edgeC.eval(sx,sy) >= 0) {
      float sampleDepth = interpDepth(sx,sy);                 // perspective-correct interpolation
      if (sampleDepth < depthBuffer.sample(px,s)) {           // per-sample depth test
        depthBuffer.writeSample(px,s,sampleDepth);            // update per-sample Z
        vec3 sampleColor = shadeSample(sx,sy);               // per-sample shading if needed
        accumColor += sampleColor;
        ++covered;
      }
    }
  }
  if (covered > 0) framebuffer.write(px,py, accumColor / float(covered)); // resolve
}