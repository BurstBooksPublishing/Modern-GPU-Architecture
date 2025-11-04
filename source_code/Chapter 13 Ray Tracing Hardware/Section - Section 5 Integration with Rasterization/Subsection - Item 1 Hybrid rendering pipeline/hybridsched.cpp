struct TileStats { int triCount; float depthVariance; float rayCoherence; };
void scheduleFrame(vector& tiles) {
  for (int i=0;i lambda) {
      enqueueRTQueue(i);   // send tile's ray-gen work to RT-core queue
    } else {
      enqueueRasterQueue(i); // keep shading on SMs
    }
  }
}