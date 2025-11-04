// warpWidth lanes; regs[warpWidth][R] per-thread regs
uint32_t activeMask = initialMask; // 1 bit per lane
while (activeMask) {
  uint32_t nextMask = 0;
  for (int lane = 0; lane < warpWidth; ++lane) {
    if (!(activeMask & (1u << lane))) continue; // lane idle
    // execute instruction for this lane using regs[lane]
    // branching example: decide which path this lane takes
    bool take = (regs[lane][R0] > threshold);
    if (take) { regs[lane][R1] = computeA(regs[lane]); nextMask |= (1u << lane); }
    else     { regs[lane][R1] = computeB(regs[lane]); nextMask |= (1u << lane); }
  }
  activeMask = nextMask; // update active set for next instruction or path
}