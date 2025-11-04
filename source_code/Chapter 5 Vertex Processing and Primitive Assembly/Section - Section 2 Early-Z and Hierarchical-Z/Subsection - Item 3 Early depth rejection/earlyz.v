module tile_z_reject #(
  parameter PIXELS = 16,            // pixels per tile
  parameter DEPTH_W = 24            // depth fixed-point width
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire [DEPTH_W-1:0]       z_tile_min,         // tile min depth
  input  wire [PIXELS*DEPTH_W-1:0] z_frag_vec,         // concatenated fragment depths
  output reg  [PIXELS-1:0]        reject_mask         // 1 => reject (occluded)
);
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      reject_mask <= {PIXELS{1'b0}};
    end else begin
      for (i=0; i= z_tile_min)
          reject_mask[i] <= 1'b1; // occluded
        else
          reject_mask[i] <= 1'b0; // potentially visible
      end
    end
  end
endmodule