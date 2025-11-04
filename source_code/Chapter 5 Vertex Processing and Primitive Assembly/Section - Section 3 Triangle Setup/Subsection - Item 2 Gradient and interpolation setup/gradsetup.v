module grad_setup
  #(parameter W=32) // fixed-point 16.16 convention
  (
   input  signed [W-1:0] x0,y0,x1,y1,x2,y2, // screen-space positions
   input  signed [W-1:0] a0,a1,a2,          // attribute (or a*winv)
   output signed [W-1:0] dax, day            // gradients in same fixed-point
  );
  // intermediate signed extended to avoid overflow
  wire signed [2*W-1:0] dx10 = x1 - x0;
  wire signed [2*W-1:0] dy20 = y2 - y0;
  wire signed [2*W-1:0] dx20 = x2 - x0;
  wire signed [2*W-1:0] dy10 = y1 - y0;
  wire signed [2*W-1:0] D = dx10*dy20 - dx20*dy10; // area determinant (scaled)
  wire signed [2*W-1:0] da10 = a1 - a0;
  wire signed [2*W-1:0] da20 = a2 - a0;
  // Numerators for gradients (scaled)
  wire signed [2*W-1:0] nx = da10*dy20 - da20*dy10;
  wire signed [2*W-1:0] ny = da20*dx10 - da10*dx20;
  // Direct division; synthesis will instantiate a divider (or be replaced by reciprocal+mul)
  assign dax = $signed(nx / D); // result in fixed-point 16.16
  assign day = $signed(ny / D);
endmodule