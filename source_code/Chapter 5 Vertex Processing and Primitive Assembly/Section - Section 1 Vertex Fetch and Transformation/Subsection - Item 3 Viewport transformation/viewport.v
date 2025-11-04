module viewport_transform_fixed #(
  parameter WIDTH=32, parameter FRACT=16
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    valid_in,
  input  wire signed [WIDTH-1:0] x_ndc, // Q16.16
  input  wire signed [WIDTH-1:0] y_ndc, // Q16.16
  input  wire signed [WIDTH-1:0] z_ndc, // Q16.16
  input  wire signed [WIDTH-1:0] vx, vy, vw, vh, // viewport params Q16.16
  input  wire signed [WIDTH-1:0] zmin, zmax, // depth range Q16.16
  output reg                     valid_out,
  output reg signed [WIDTH-1:0]  x_win, y_win, z_win // Q16.16
);
  // constants
  localparam signed [WIDTH-1:0] ONE = (1 << FRACT);
  // pipeline stage registers
  reg signed [WIDTH-1:0] x_tmp, y_tmp, z_tmp;
  reg signed [2*WIDTH-1:0] mul_x, mul_y, mul_z; // full mult product
  // stage 0: add/subtract
  always @(posedge clk) begin
    if (rst) begin
      valid_out <= 0;
      x_tmp <= 0; y_tmp <= 0; z_tmp <= 0;
      mul_x <= 0; mul_y <= 0; mul_z <= 0;
      x_win <= 0; y_win <= 0; z_win <= 0;
    end else begin
      // compute (ndc + 1) / 2  in Q16.16: (ndc + ONE) >> 1
      x_tmp <= (x_ndc + ONE) >>> 1;
      y_tmp <= (y_ndc + ONE) >>> 1;
      z_tmp <= (z_ndc + ONE) >>> 1;
      // multiply by viewport scale: (x_tmp * w_v) >> FRACT
      mul_x <= x_tmp * vw;
      mul_y <= x_tmp * 0; // placeholder to avoid latches (unused)
      // compute depth multiplier (zmax - zmin)
      mul_z <= z_tmp * (zmax - zmin);
      // advance valid (simple flow control)
      valid_out <= valid_in;
      // finalize outputs one cycle later (simple pipeline)
      x_win <= (mul_x >>> FRACT) + vx;
      // y: flip for top-left origin: y_win = (1 - y_mapped)*h + vy 
      // = h - (y_tmp*h) + vy
      y_win <= vh - ( (y_tmp * vh) >>> FRACT ) + vy;
      z_win <= zmin + (mul_z >>> FRACT);
    end
  end
endmodule