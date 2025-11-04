module ycbcr_to_rgb #(
  parameter BW = 8,               // input component bits
  parameter COEFW = 16            // fixed-point coefficient width Q2.14
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 valid_in,
  input  wire [BW-1:0]        Y_in,
  input  wire [BW-1:0]        Cb_in,
  input  wire [BW-1:0]        Cr_in,
  output reg                  valid_out,
  output reg  [BW-1:0]        R_out,
  output reg  [BW-1:0]        G_out,
  output reg  [BW-1:0]        B_out
);
  // Coefs in Q2.14: 1.402 -> int(1.402*2^14)=22981 etc.
  localparam signed [COEFW-1:0] C_r_cr = 16'sd22981; // 1.402
  localparam signed [COEFW-1:0] C_g_cb = -16'sd5638; // -0.344136
  localparam signed [COEFW-1:0] C_g_cr = -16'sd11698; // -0.714136
  localparam signed [COEFW-1:0] C_b_cb = 16'sd29032; // 1.772

  // Stage 1 registers
  reg signed [BW-1:0] Ys, Cbs, Crs;
  always @(posedge clk) begin
    if (!rstn) begin valid_out <= 0; Ys <= 0; Cbs <= 0; Crs <= 0; end
    else begin
      Ys  <= $signed(Y_in);
      Cbs <= $signed(Cb_in) - 8'sd128; // center chroma
      Crs <= $signed(Cr_in) - 8'sd128;
      valid_out <= valid_in; // one-cycle pipeline flag
    end
  end

  // Stage 2: multiply-accumulate and downshift Q2.14
  always @(posedge clk) begin
    if (!rstn) begin R_out <= 0; G_out <= 0; B_out <= 0; end
    else begin
      // wide products
      wire signed [31:0] r_acc = ($signed(Ys) <<< 14) + C_r_cr * Crs;
      wire signed [31:0] g_acc = ($signed(Ys) <<< 14) + C_g_cb * Cbs + C_g_cr * Crs;
      wire signed [31:0] b_acc = ($signed(Ys) <<< 14) + C_b_cb * Cbs;
      // convert back from Q2.14 to integer with rounding, clamp 0..255
      function [BW-1:0] q2_14_to_8(input signed [31:0] v);
        reg signed [31:0] t;
        begin
          t = (v + 32'sd8192) >>> 14; // rounding
          if (t < 0) q2_14_to_8 = 0;
          else if (t > 255) q2_14_to_8 = 8'hFF;
          else q2_14_to_8 = t[7:0];
        end
      endfunction
      R_out <= q2_14_to_8(r_acc);
      G_out <= q2_14_to_8(g_acc);
      B_out <= q2_14_to_8(b_acc);
    end
  end
endmodule