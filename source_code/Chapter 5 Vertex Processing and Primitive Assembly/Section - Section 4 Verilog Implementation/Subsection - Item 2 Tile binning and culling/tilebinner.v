module tile_binner #(
  parameter TILE_SHIFT = 4,             // tile width = 2^TILE_SHIFT
  parameter SCREEN_W = 1920,
  parameter SCREEN_H = 1080,
  parameter PIXEL_BITS = 16,
  parameter TILE_IDX_BITS = 10
)(
  input  wire [PIXEL_BITS-1:0] xmin,
  input  wire [PIXEL_BITS-1:0] ymin,
  input  wire [PIXEL_BITS-1:0] xmax,
  input  wire [PIXEL_BITS-1:0] ymax,
  output wire [TILE_IDX_BITS-1:0] tx0, // inclusive min tile x
  output wire [TILE_IDX_BITS-1:0] ty0, // inclusive min tile y
  output wire [TILE_IDX_BITS-1:0] tx1, // inclusive max tile x
  output wire [TILE_IDX_BITS-1:0] ty1  // inclusive max tile y
);
  localparam TILE_W = (1 << TILE_SHIFT);
  localparam MAX_TX = ( (SCREEN_W + TILE_W - 1) >> TILE_SHIFT ) - 1;
  localparam MAX_TY = ( (SCREEN_H + TILE_W - 1) >> TILE_SHIFT ) - 1;

  // clamp coordinates to framebuffer
  wire [PIXEL_BITS-1:0] cxmin = (xmin < 0) ? 0 : ((xmin > SCREEN_W-1) ? SCREEN_W-1 : xmin);
  wire [PIXEL_BITS-1:0] cymin = (ymin < 0) ? 0 : ((ymin > SCREEN_H-1) ? SCREEN_H-1 : ymin);
  wire [PIXEL_BITS-1:0] cxmax = (xmax < 0) ? 0 : ((xmax > SCREEN_W-1) ? SCREEN_W-1 : xmax);
  wire [PIXEL_BITS-1:0] cymax = (ymax < 0) ? 0 : ((ymax > SCREEN_H-1) ? SCREEN_H-1 : ymax);

  // compute tile indices via shift
  assign tx0 = (cxmin >> TILE_SHIFT) > MAX_TX ? MAX_TX : (cxmin >> TILE_SHIFT);
  assign ty0 = (cymin >> TILE_SHIFT) > MAX_TY ? MAX_TY : (cymin >> TILE_SHIFT);
  assign tx1 = (cxmax >> TILE_SHIFT) > MAX_TX ? MAX_TX : (cxmax >> TILE_SHIFT);
  assign ty1 = (cymax >> TILE_SHIFT) > MAX_TY ? MAX_TY : (cymax >> TILE_SHIFT);

endmodule