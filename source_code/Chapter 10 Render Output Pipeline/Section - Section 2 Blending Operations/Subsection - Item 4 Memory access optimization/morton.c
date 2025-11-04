static inline uint32_t morton32(uint16_t x, uint16_t y) {
    uint32_t xx = x; uint32_t yy = y;
    xx = (xx | (xx << 8)) & 0x00FF00FFu;    // spread bits
    xx = (xx | (xx << 4)) & 0x0F0F0F0Fu;
    xx = (xx | (xx << 2)) & 0x33333333u;
    xx = (xx | (xx << 1)) & 0x55555555u;
    yy = (yy | (yy << 8)) & 0x00FF00FFu;
    yy = (yy | (yy << 4)) & 0x0F0F0F0Fu;
    yy = (yy | (yy << 2)) & 0x33333333u;
    yy = (yy | (yy << 1)) & 0x55555555u;
    return (xx | (yy << 1)); // interleaved address
}