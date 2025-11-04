[numthreads(8,8,1)]
void CSMain(uint3 tid : SV_DispatchThreadID) {
    // inputs bound: gX,gY (pixels), screenW,screenH, tileSize, sigma, alpha
    uint2 tile = tid.xy;                          // tile coord
    float2 p = (tile + 0.5) * tileSize;          // tile center in pixels
    float dx = p.x - gX; float dy = p.y - gY;
    float d2 = dx*dx + dy*dy;
    float fall = 1.0f - exp(-d2 / (2.0f * sigma*sigma)); // 0..1
    // map fall to discrete rates: 0->1x1, 1->4x4 for example
    int step = (int)(fall * float(maxSteps));    // quantize
    step = clamp(step, 0, maxSteps);
    uint rate = rateTable[step];                 // e.g., {1,1,2,4}
    OutputRateMap[tile] = rate;                  // write to UAV texture
}