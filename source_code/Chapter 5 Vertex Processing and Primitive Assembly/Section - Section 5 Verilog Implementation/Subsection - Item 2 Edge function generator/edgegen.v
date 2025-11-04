module edge_gen #(
  parameter INT_W = 20,         // integer bits
  parameter SUBPIX = 4,         // fractional bits
  parameter W = INT_W+SUBPIX+1  // sign bit
)(
  input  wire                  clk,
  input  wire                  rst,
  input  wire                  valid_in,
  input  wire signed [W-1:0]   vx0, vy0, // vertex 0 (Q format)
  input  wire signed [W-1:0]   vx1, vy1, // vertex 1
  input  wire signed [W-1:0]   vx2, vy2, // vertex 2
  input  wire signed [W-1:0]   tile_x, tile_y, // tile origin (Q format)
  output reg                   valid_out,
  output reg signed [W+W-1:0]  A0B0C0, // concatenated A,B,C for edge0
  output reg signed [W+W-1:0]  A1B1C1, // edge1
  output reg signed [W+W-1:0]  A2B2C2, // edge2
  output reg signed [W-1:0]    E0_init, E1_init, E2_init // initial E at tile origin
);

  // stage 0: compute edge deltas and raw C
  reg signed [W-1:0] sx0, sy0, sx1, sy1, sx2, sy2;
  reg signed [W-1:0] A0, B0, C0, A1, B1, C1, A2, B2, C2;
  reg signed [W-1:0] bias0, bias1, bias2;
  reg v0;

  // top-left test: true when edge is top-left per raster rules
  function top_left;
    input signed [W-1:0] x0,y0,x1,y1;
    begin
      top_left = (y0 > y1) || ((y0 == y1) && (x1 > x0));
    end
  endfunction

  always @(posedge clk) begin
    if (rst) begin
      v0 <= 0;
    end else begin
      v0 <= valid_in;
      if (valid_in) begin
        sx0 <= vx0; sy0 <= vy0; sx1 <= vx1; sy1 <= vy1; sx2 <= vx2; sy2 <= vy2;
        // A = y1 - y0; B = -(x1 - x0); C = x1*y0 - y1*x0 (Q math fits in double width)
        A0 <= sy1 - sy0;
        B0 <= -(sx1 - sx0);
        C0 <= (sx1*sy0 - sy1*sx0) >>> SUBPIX; // normalize product back to Q
        bias0 <= top_left(sx0,sy0,sx1,sy1) ? 0 : 1; // bias in LSB units
        A1 <= sy2 - sy1;
        B1 <= -(sx2 - sx1);
        C1 <= (sx2*sy1 - sy2*sx1) >>> SUBPIX;
        bias1 <= top_left(sx1,sy1,sx2,sy2) ? 0 : 1;
        A2 <= sy0 - sy2;
        B2 <= -(sx0 - sx2);
        C2 <= (sx0*sy2 - sy0*sx2) >>> SUBPIX;
        bias2 <= top_left(sx2,sy2,sx0,sy0) ? 0 : 1;
      end
    end
  end

  // stage 1: finalize coefficients and compute initial E at tile origin
  reg v1;
  always @(posedge clk) begin
    if (rst) begin
      valid_out <= 0; v1 <= 0;
    end else begin
      v1 <= v0;
      valid_out <= v1;
      if (v0) begin
        // apply bias: bias represented in 1 LSB of Q format
        C0 <= C0 - bias0; C1 <= C1 - bias1; C2 <= C2 - bias2;
        A0B0C0 <= {A0,B0,C0}; A1B1C1 <= {A1,B1,C1}; A2B2C2 <= {A2,B2,C2};
        // initial E = A*tile_x + B*tile_y + C
        E0_init <= (A0*tile_x + B0*tile_y) >>> SUBPIX; E0_init <= E0_init + C0;
        E1_init <= (A1*tile_x + B1*tile_y) >>> SUBPIX; E1_init <= E1_init + C1;
        E2_init <= (A2*tile_x + B2*tile_y) >>> SUBPIX; E2_init <= E2_init + C2;
      end
    end
  end

endmodule