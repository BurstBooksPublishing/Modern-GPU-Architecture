module tile_binner #(parameter SCREEN_W=1920, SCREEN_H=1080, TILE_W=32, TILE_H=32, MAX_BINS=65536)
(
  input clk, input rst,
  input prim_valid,
  input [15:0] x0, y0, x1, y1, // primitive bbox (inclusive)
  output reg bin_we,
  output reg [15:0] bin_addr,
  output reg [31:0] bin_data
);
  localparam NX = (SCREEN_W + TILE_W - 1)/TILE_W;
  localparam NY = (SCREEN_H + TILE_H - 1)/TILE_H;
  reg [15:0] tx0, ty0, tx1, ty1;
  reg [15:0] tx, ty;
  reg state;
  always @(*) begin
    tx0 = x0 / TILE_W; ty0 = y0 / TILE_H;
    tx1 = x1 / TILE_W; ty1 = y1 / TILE_H;
    if (tx1 >= NX) tx1 = NX-1;
    if (ty1 >= NY) ty1 = NY-1;
  end
  always @(posedge clk) begin
    if (rst) begin
      state <= 0; bin_we <= 0; tx <= 0; ty <= 0;
    end else begin
      case (state)
        0: if (prim_valid) begin tx <= tx0; ty <= ty0; state <= 1; end
        1: begin
             bin_addr <= ty*NX + tx; bin_data <= {y0,x0}; bin_we <= 1; // write bin entry
             if (tx==tx1 && ty==ty1) begin state <= 0; bin_we <= 0; end
             else begin bin_we <= 0;
               if (tx==tx1) begin tx <= tx0; ty <= ty+1; end
               else tx <= tx+1;
             end
           end
      endcase
    end
  end
endmodule