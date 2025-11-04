module hi_z_tile #
  (parameter DEPTH_BITS=24)
  (
   input  wire                     clk,
   input  wire                     rstn,
   input  wire [DEPTH_BITS-1:0]    frag_z,   // incoming fragment depth (fixed-point)
   input  wire                     valid,    // fragment valid
   input  wire                     write_en, // commit depth to leaf happened
   output reg                      reject    // asserted when conservatively occluded
  );
  reg [DEPTH_BITS-1:0] tile_max_z; // stored tile maximum depth (farthest)
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      tile_max_z <= {DEPTH_BITS{1'b0}}; // clear: near (safe initial far value can be set differently)
      reject <= 1'b0;
    end else begin
      if (valid) begin
        // conservative reject if fragment is behind farthest sample in tile
        reject <= (frag_z >= tile_max_z);
      end
      if (valid && write_en) begin
        // atomic-max update: only monotonic increases handled correctly here
        if (frag_z > tile_max_z)
          tile_max_z <= frag_z;
      end
    end
  end
endmodule