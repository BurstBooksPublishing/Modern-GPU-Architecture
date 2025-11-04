module tess_ctrl #(
  parameter COORD_W = 16,          // Q12.4 fixed-point
  parameter TF_W    = 8,           // tess factor width
  parameter ALPHA   = 1,           // scale numerator
  parameter TARGET  = 8,           // target pixels per edge
  parameter TMAX    = 64
) (
  input  wire clk, rst,
  input  wire valid,               // input coords valid
  input  wire [COORD_W-1:0] x0,y0, // control points (window space)
  input  wire [COORD_W-1:0] x1,y1,
  input  wire [COORD_W-1:0] x2,y2,
  input  wire [COORD_W-1:0] x3,y3,
  output reg  [TF_W-1:0] outer0, outer1, outer2, outer3,
  output reg  [TF_W-1:0] inner0, inner1
);
  // compute abs diff and approx length for an edge
  function [COORD_W-1:0] approx_len;
    input signed [COORD_W-1:0] dx, dy;
    reg [COORD_W-1:0] adx, ady, ma, mi;
    begin
      adx = (dx[COORD_W-1]) ? -dx : dx;
      ady = (dy[COORD_W-1]) ? -dy : dy;
      ma = (adx > ady) ? adx : ady;
      mi = (adx > ady) ? ady : adx;
      approx_len = ma + (mi >> 1); // eq (1) integer form
    end
  endfunction

  // map to tess factor (integer)
  function [TF_W-1:0] map_tess;
    input [COORD_W-1:0] len;
    reg [COORD_W+7:0] scaled;
    reg [TF_W-1:0] t;
    begin
      scaled = (ALPHA * len) + (TARGET>>1);
      t = scaled / TARGET; // integer division -> ceil approx via bias above
      if (t < 1) t = 1;
      if (t > TMAX) t = TMAX;
      map_tess = t;
    end
  endfunction

  // pipeline registers
  reg [COORD_W-1:0] ax0,ay0,ax1,ay1,ax2,ay2,ax3,ay3;
  always @(posedge clk) begin
    if (rst) begin
      outer0<=0; outer1<=0; outer2<=0; outer3<=0; inner0<=0; inner1<=0;
    end else if (valid) begin
      ax0<=x0; ay0<=y0; ax1<=x1; ay1<=y1; ax2<=x2; ay2<=y2; ax3<=x3; ay3<=y3;
      // compute outer edges
      outer0 <= map_tess( approx_len(ax0-ax1, ay0-ay1) );
      outer1 <= map_tess( approx_len(ax1-ax2, ay1-ay2) );
      outer2 <= map_tess( approx_len(ax2-ax3, ay2-ay3) );
      outer3 <= map_tess( approx_len(ax3-ax0, ay3-ay0) );
      // compute two inner edges (diagonals) for quad domain
      inner0 <= map_tess( approx_len(ax0-ax2, ay0-ay2) );
      inner1 <= map_tess( approx_len(ax1-ax3, ay1-ay3) );
    end
  end
endmodule