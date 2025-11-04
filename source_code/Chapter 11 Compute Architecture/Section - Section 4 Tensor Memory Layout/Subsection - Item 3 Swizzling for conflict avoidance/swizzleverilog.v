module swizzle_addr_gen #(
  parameter integer ADDR_WIDTH = 64,
  parameter integer TILE_SHIFT = 6, // tile size = 64 bytes
  parameter integer LANE_SHIFT = 5  // lane granularity
) (
  input  wire [ADDR_WIDTH-1:0] base_addr, // base tile address
  input  wire [31:0]            tile_idx,  // linear tile index
  input  wire [31:0]            lane,      // warp lane id
  output wire [ADDR_WIDTH-1:0]  swizzled_addr
);
  // extend and shift to ADDR_WIDTH then XOR and add
  wire [ADDR_WIDTH-1:0] tile_off = {{(ADDR_WIDTH-32){1'b0}}, tile_idx} << TILE_SHIFT;
  wire [ADDR_WIDTH-1:0] lane_off = {{(ADDR_WIDTH-32){1'b0}}, lane}     << LANE_SHIFT;
  assign swizzled_addr = base_addr + (tile_off ^ lane_off); // combinational
endmodule