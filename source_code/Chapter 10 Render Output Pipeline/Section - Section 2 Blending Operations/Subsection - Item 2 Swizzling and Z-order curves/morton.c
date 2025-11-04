#include 
// Expand 16-bit to 32-bit by inserting zeros between bits.
static inline uint32_t expandBits(uint16_t v){
    uint32_t x = v;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
}
static inline uint32_t mortonEncode(uint16_t x, uint16_t y){
    return (expandBits(x) | (expandBits(y) << 1)); // interleaved index
}
// Simple decode (extract every other bit).
static inline uint16_t compactBits(uint32_t v){
    v &= 0x55555555;
    v = (v | (v >> 1)) & 0x33333333;
    v = (v | (v >> 2)) & 0x0F0F0F0F;
    v = (v | (v >> 4)) & 0x00FF00FF;
    v = (v | (v >> 8)) & 0x0000FFFF;
    return (uint16_t)v;
}
static inline void mortonDecode(uint32_t code, uint16_t *x, uint16_t *y){
    *x = compactBits(code);
    *y = compactBits(code >> 1);
}