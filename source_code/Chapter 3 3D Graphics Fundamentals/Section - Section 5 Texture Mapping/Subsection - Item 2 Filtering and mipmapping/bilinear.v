module bilinear_interp(
  input  wire        clk,
  input  wire        en,
  input  wire [31:0] tex00, // RGBA8
  input  wire [31:0] tex10,
  input  wire [31:0] tex01,
  input  wire [31:0] tex11,
  input  wire [7:0]  fracu, // Q8
  input  wire [7:0]  fracv, // Q8
  output reg  [31:0] color  // RGBA8 result
);
  // stage 1: compute weights (Q16)
  wire [15:0] one_minus_u = 8'd255 - fracu;
  wire [15:0] one_minus_v = 8'd255 - fracv;
  wire [15:0] w00 = one_minus_u * one_minus_v; // 8x8 -> 16
  wire [15:0] w10 = fracu * one_minus_v;
  wire [15:0] w01 = one_minus_u * fracv;
  wire [15:0] w11 = fracu * fracv;
  // extract channels and compute weighted sums (combinational)
  function [23:0] mul8x16(input [7:0] c, input [15:0] w);
    mul8x16 = c * w; // 8*16 -> 24 bits
  endfunction
  wire [23:0] r_sum = mul8x16(tex00[31:24],w00) + mul8x16(tex10[31:24],w10)
                    + mul8x16(tex01[31:24],w01) + mul8x16(tex11[31:24],w11);
  wire [23:0] g_sum = mul8x16(tex00[23:16],w00) + mul8x16(tex10[23:16],w10)
                    + mul8x16(tex01[23:16],w01) + mul8x16(tex11[23:16],w11);
  wire [23:0] b_sum = mul8x16(tex00[15:8],w00)  + mul8x16(tex10[15:8],w10)
                    + mul8x16(tex01[15:8],w01)  + mul8x16(tex11[15:8],w11);
  wire [23:0] a_sum = mul8x16(tex00[7:0],w00)   + mul8x16(tex10[7:0],w10)
                    + mul8x16(tex01[7:0],w01)   + mul8x16(tex11[7:0],w11);
  // stage 2: normalize by Q16 (>>16) and register output
  always @(posedge clk) begin
    if (en) begin
      color[31:24] <= r_sum[23:16];
      color[23:16] <= g_sum[23:16];
      color[15:8]  <= b_sum[23:16];
      color[7:0]   <= a_sum[23:16];
    end
  end
endmodule