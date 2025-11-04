module perspective_interpolator #(
  parameter W = 32,               // fixed-point width
  parameter FRAC = 16,            // fractional bits
  parameter OUTW = 32
)(
  input  wire                   clk,
  input  wire                   rst,
  // precomputed plane coefficients (fixed-point)
  input  wire signed [W-1:0]    base_a_over_w, // A0 / w0 at origin
  input  wire signed [W-1:0]    dadx, dady,    // gradients for A_over_w
  input  wire signed [W-1:0]    base_one_over_w,// 1/w at origin
  input  wire signed [W-1:0]    dwdx, dwdy,    // gradients for 1/w
  input  wire signed [W-1:0]    px, py,        // pixel coordinates (fixed-point)
  output reg  signed [OUTW-1:0] attr_out       // final attribute (fixed-point)
);
  // Stage 1: compute numerator and denominator planes
  wire signed [W+W-1:0] mul_dx_x = dadx * px;   // widened multiply
  wire signed [W+W-1:0] mul_dy_y = dady * py;
  wire signed [W+W-1:0] num_acc = mul_dx_x + mul_dy_y + (base_a_over_w <<< FRAC);
  wire signed [W+W-1:0] mul_wdx_x = dwdx * px;
  wire signed [W+W-1:0] mul_wdy_y = dwdy * py;
  wire signed [W+W-1:0] den_acc = mul_wdx_x + mul_wdy_y + (base_one_over_w <<< FRAC);

  // Stage 2: divide numerator by denominator to recover attribute
  // Simple synchronous divide (synthesisable but area-consuming).
  // Scale down to output fractional format.
  always @(posedge clk) begin
    if (rst) begin
      attr_out <= 0;
    end else begin
      if (den_acc != 0)
        attr_out <= $signed(num_acc) / $signed(den_acc); // fixed-point division
      else
        attr_out <= {OUTW{1'b0}};
    end
  end
endmodule