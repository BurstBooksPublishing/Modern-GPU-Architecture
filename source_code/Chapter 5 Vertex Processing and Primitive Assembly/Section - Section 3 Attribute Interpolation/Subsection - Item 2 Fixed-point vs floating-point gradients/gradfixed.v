module grad_fixed #(
  parameter INT=16, FRAC=16 // Q(INT).Q(FRAC) format
)(
  input  signed [INT+FRAC-1:0] x1, y1, p1, // vertex1 (Q)
  input  signed [INT+FRAC-1:0] x2, y2, p2, // vertex2 (Q)
  input  signed [INT+FRAC-1:0] x3, y3, p3, // vertex3 (Q)
  output signed [INT+FRAC-1:0] dpdx, dpdy, // gradients (Q)
  output reg                     valid
);
  // Intermediate wider width to avoid overflow
  localparam W = INT+FRAC+INT; // conservative
  wire signed [W-1:0] dy23 = y2 - y3;
  wire signed [W-1:0] dy31 = y3 - y1;
  wire signed [W-1:0] dy12 = y1 - y2;
  wire signed [W-1:0] dx23 = x2 - x3;
  wire signed [W-1:0] dx31 = x3 - x1;
  wire signed [W-1:0] dx12 = x1 - x2;

  // numerator and denominator for ∂p/∂x
  wire signed [W-1:0] num_x = p1*dy23 + p2*dy31 + p3*dy12;
  wire signed [W-1:0] den   = x1*dy23 + x2*dy31 + x3*dy12; // area*2

  // numerator for ∂p/∂y (swap roles)
  wire signed [W-1:0] num_y = p1*dx32 + p2*dx13 + p3*dx21;
  // create dx32 etc. (explicit to avoid psuedo-names)
  wire signed [W-1:0] dx32 = x3 - x2;
  wire signed [W-1:0] dx13 = x1 - x3;
  wire signed [W-1:0] dx21 = x2 - x1;

  // Avoid divide-by-zero: small-area clamp
  wire signed [W-1:0] den_safe = (den == 0) ? {1'b0, {W-1{1'b1}}} : den;

  // integer division yields Q-format result; divide synthesizes to hardware divider
  assign dpdx = $signed(num_x / den_safe); // result in same Q
  assign dpdy = $signed(num_y / den_safe);

  always @(*) begin
    // Valid if area not tiny (simple threshold)
    valid = (den != 0);
  end
endmodule