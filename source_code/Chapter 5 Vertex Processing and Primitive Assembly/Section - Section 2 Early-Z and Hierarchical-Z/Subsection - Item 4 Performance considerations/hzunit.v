module hz_tile_unit #(
  parameter ADDR_WIDTH = 10,               // number of tiles = 2^ADDR_WIDTH
  parameter DEPTH_W = 16                    // fixed-point depth width
)(
  input  wire                     clk,
  input  wire                     rstn,
  // query interface
  input  wire [ADDR_WIDTH-1:0]    tile_addr,
  input  wire [DEPTH_W-1:0]       in_depth,
  input  wire                     query_valid,
  output reg  [1:0]               verdict,    // 00=reject,01=pass,10=unknown
  output reg                      verdict_valid,
  // update interface (from ROP commit)
  input  wire                     update_valid,
  input  wire [ADDR_WIDTH-1:0]    upd_addr,
  input  wire [DEPTH_W-1:0]       upd_depth
);

  localparam NUM_TILES = (1< reject; nearer-than-closest => pass
        if (in_depth >= tmax) begin
          verdict <= 2'b00; // reject (occluded)
        end else if (in_depth < tmin) begin
          verdict <= 2'b01; // pass (definitely visible)
        end else begin
          verdict <= 2'b10; // unknown -> need per-sample test
        end
        verdict_valid <= 1'b1;
      end
      // handle update from ROP: update min/max
      if (update_valid) begin
        // update min (closest) and max (farthest) conservatively
        if (upd_depth < min_depth[upd_addr]) min_depth[upd_addr] <= upd_depth;
        if (upd_depth > max_depth[upd_addr]) max_depth[upd_addr] <= upd_depth;
      end
    end
  end
endmodule