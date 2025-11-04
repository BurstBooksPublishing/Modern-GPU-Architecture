// framebuffer coords (x,y), SRI buffer preloaded; Bx,By block sizes
int u = x / Bx; int v = y / By;           // SRI index (fast integer div/shr)
uint8_t rate = SRI[v * SRI_stride + u];   // e.g., encoded: 0=1x1,1=2x2,2=4x4
int tile_w = blockWidthFromRate(rate);    // decode to Bx,B y multiples
int tile_h = blockHeightFromRate(rate);
emit_shading_tile(tile_center(x,y,tile_w,tile_h), tile_w, tile_h); // single shader dispatch