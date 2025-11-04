module fp32_to_fp16
(
  input  wire [31:0] in,    // IEEE-754 binary32
  output reg  [15:0] out
);
  parameter FTZ = 1'b1;      // flush-to-zero for denormals

  wire sign = in[31];
  wire [7:0] exp32 = in[30:23];
  wire [22:0] frac32 = in[22:0];

  // handle special cases
  wire is_inf_nan = (exp32 == 8'hFF);
  wire is_zero = (exp32 == 8'd0) && (frac32 == 23'd0);

  // extended significand with room for shifting and rounding bits
  wire [36:0] ext = {1'b1, frac32, 13'b0}; // 1+23+13=37

  // compute target exponent as signed to detect underflow/overflow
  integer e32_i = 0;
  integer e16_i = 0;
  always @* e32_i = exp32;
  always @* e16_i = e32_i - 127 + 15;

  // denorm shift (if any)
  integer denorm_shift = 0;
  always @* denorm_shift = (e16_i <= 0) ? (1 - e16_i) : 0;

  // shifted value for extracting mantissa, guard, round, sticky
  wire [36:0] shifted = ext >> (13 + denorm_shift);
  wire [9:0] mant10   = shifted[23:14];
  wire guard = shifted[13];
  wire roundb = shifted[12];
  wire sticky = |shifted[11:0];

  // rounding decision (round-to-nearest-even)
  wire round_add = guard & (roundb | sticky | mant10[0]);

  reg [9:0] mant_rounded;
  reg [4:0] exp_out; // 5-bit exponent for FP16

  always @* begin
    if (is_inf_nan) begin
      exp_out = 5'h1F;                     // all ones
      mant_rounded = (frac32 == 0) ? 10'd0 : 10'h200; // NaN payload -> qNaN
    end else if (is_zero) begin
      exp_out = 5'd0;
      mant_rounded = 10'd0;
    end else begin
      // normalized or denorm path
      reg [10:0] mant_plus;
      mant_plus = {1'b0, mant10} + round_add;
      exp_out = e16_i[4:0];
      // handle mantissa overflow from rounding
      if (mant_plus[10]) begin
        mant_rounded = 10'd0;
        exp_out = exp_out + 1;
      end else begin
        mant_rounded = mant_plus[9:0];
      end
      // exponent overflow -> Inf
      if (e16_i >= 31) begin
        exp_out = 5'h1F;
        mant_rounded = 10'd0;
      end
      // underflow handling: if still <=0 after rounding, FTZ or denorm
      if (e16_i <= 0 && FTZ) begin
        exp_out = 5'd0;
        mant_rounded = 10'd0;
      end
    end
    out = {sign, exp_out, mant_rounded};
  end
endmodule