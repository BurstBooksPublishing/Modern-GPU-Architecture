module tile_light_binner #(
  parameter SCREEN_W = 1920,
  parameter SCREEN_H = 1080,
  parameter TILE_W = 16, // power of two
  parameter TILE_H = 16, // power of two
  parameter MAX_LIGHTS = 32,
  parameter NTX = (SCREEN_W+TILE_W-1)/TILE_W,
  parameter NTY = (SCREEN_H+TILE_H-1)/TILE_H
)(
  input  wire                clk,
  input  wire                rst_n,
  // light input: id [0,MAX_LIGHTS-1], x/y bounds in pixels
  input  wire                in_valid,
  input  wire [4:0]          in_light_id, // supports 0..31
  input  wire [15:0]         in_x0, in_y0, in_x1, in_y1,
  output reg                 ready
);
  // per-tile 32-bit mask RAM (synthesizable)
  reg [MAX_LIGHTS-1:0] tile_mask [0:NTX*NTY-1];
  integer tx0, tx1, ty0, ty1;
  integer tx, ty, idx;
  wire [31:0] bit = 32'h1 << in_light_id;

  // reset RAM (synchronous clear)
  integer i;
  always @(posedge clk) begin
    if (!rst_n) begin
      for (i=0; i> $clog2(TILE_W);
        ty0 = in_y0 >> $clog2(TILE_H);
        tx1 = in_x1 >> $clog2(TILE_W);
        ty1 = in_y1 >> $clog2(TILE_H);
        // clamp
        if (tx0 < 0) tx0 = 0; if (ty0 < 0) ty0 = 0;
        if (tx1 >= NTX) tx1 = NTX-1; if (ty1 >= NTY) ty1 = NTY-1;
        // update tiles in bounding box (constant-bounded loops)
        for (ty = ty0; ty <= ty1; ty = ty + 1) begin
          for (tx = tx0; tx <= tx1; tx = tx + 1) begin
            idx = ty*NTX + tx;
            tile_mask[idx] <= tile_mask[idx] | bit; // set bit
          end
        end
        ready <= 1'b1;
      end
    end
  end
endmodule