module edge_eq #(
  parameter IW = 16, // integer bits
  parameter FRAC = 8, // fractional bits
  parameter W = IW+FRAC // total bitwidth
)(
  input  wire signed [W-1:0] vx0, vy0, vx1, vy1, // fixed-point vertices
  input  wire signed [W-1:0] px, py,             // fixed-point sample position
  output wire signed [W-1:0] A, B,               // coefficients
  output wire signed [2*W-1:0] C,                // C needs double width
  output wire signed [2*W-1:0] eval             // E(px,py)
);
  // compute coefficients
  assign A = vy0 - vy1;                            // (y0 - y1)
  assign B = vx1 - vx0;                            // (x1 - x0)
  assign C = $signed(vx0) * $signed(vy1) - $signed(vx1) * $signed(vy0); // x0*y1 - x1*y0
  // evaluate edge function: promote A,B to double width for multiplication
  wire signed [2*W-1:0] A_ext = $signed(A);
  wire signed [2*W-1:0] B_ext = $signed(B);
  assign eval = A_ext * $signed(px) + B_ext * $signed(py) + C;
endmodule