struct RaySoA { // arrays for N rays; N is wavefront size
  float* ox; float* oy; float* oz;          // origin arrays
  float* dx; float* dy; float* dz;          // direction arrays
  float* dir_rcpx; float* dir_rcpy; float* dir_rcpz; // reciprocals
  uint8_t* dir_sign;                        // 3-bit sign per ray
  float* tmin; float* tmax;                 // interval arrays
  uint32_t* payload;                        // shader payload index
};
// Example init inner loop (host or kernel): compute reciprocals and sign
// for (i = 0; i < N; ++i) { dir_rcpx[i] = 1.0f / dx[i]; dir_sign[i] = dx[i] < 0; ... }