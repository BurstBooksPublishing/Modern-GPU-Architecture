module perspective_interpolator #(
  parameter IW = 32,              // input fixed width
  parameter FW = 32,              // fraction bits
  parameter OW = 32               // output width
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 valid_in,
  input  wire signed [IW-1:0] af, // a_f (fixed-point)
  input  wire signed [IW-1:0] bf, // b_f
  input  wire signed [IW-1:0] cf, // c_f
  input  wire signed [IW-1:0] ag, // a_g
  input  wire signed [IW-1:0] bg, // b_g
  input  wire signed [IW-1:0] cg, // c_g
  input  wire signed [IW-1:0] px, // pixel x (fixed-point)
  input  wire signed [IW-1:0] py, // pixel y
  output reg                  valid_out,
  output reg  signed [OW-1:0] attr   // A_p (fixed-point)
);
  // stage 1: evaluate planes (64-bit intermediates)
  reg signed [IW*2-1:0] f_val, g_val;
  always @(posedge clk) begin
    if (!rst_n) begin
      f_val <= 0; g_val <= 0; valid_out <= 0; attr <= 0;
    end else begin
      // multiply accumulate: af*px + bf*py + cf
      f_val <= af * px + bf * py + cf;
      g_val <= ag * px + bg * py + cg;
      // stage 2 performed next cycle: divide (synthesizable /)
      if (g_val != 0) attr <= (f_val <<< 0) / g_val; // integer divide, fixed-point alignment implied
      else attr <= 0;
      valid_out <= valid_in;
    end
  end
endmodule