#include 
// interleave bits of x and y to form Morton index; used to compute tile base.
static inline uint32_t morton2(uint16_t x, uint16_t y) {
    uint32_t z = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        z |= ((x >> i) & 1u) << (2*i);   // x bit -> even positions
        z |= ((y >> i) & 1u) << (2*i+1); // y bit -> odd positions
    }
    return z; // use with tile_size to form physical block address
}