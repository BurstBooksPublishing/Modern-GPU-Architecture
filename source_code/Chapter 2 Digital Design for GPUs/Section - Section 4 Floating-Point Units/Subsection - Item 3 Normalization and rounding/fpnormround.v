module fp_norm_round
#(parameter EXP = 8, MANT = 23)
(
  input  wire                 sign_in,
  input  wire [EXP-1:0]       exp_in,            // biased exponent
  input  wire [MANT+4:0]      mant_in,           // [carry|MANT-1:0|g|r|s]
  input  wire [1:0]           rnd_mode,          // 00:RN(even),01:RZ,10:RP,11:RM
  output reg                  sign_out,
  output reg [EXP-1:0]        exp_out,
  output reg [MANT-1:0]       mant_out,
  output reg                  overflow,
  output reg                  underflow,
  output reg                  inexact
);
  // internal wires
  reg carry;
  reg [MANT:0] sig_shifted; // includes implicit 1 for normalized mantissa
  reg [MANT+4:0] working;   // working shifted value
  integer i;
  integer lzc; // leading zero count

  // combinational normalization + rounding
  always @* begin
    sign_out = sign_in;
    overflow = 0;
    underflow = 0;
    inexact = 0;
    exp_out = exp_in;
    carry = mant_in[MANT+4]; // top bit is carry
    working = mant_in;

    if (carry) begin
      // shift-right one (handle adder carry) -> top becomes 1.x
      // form: [carry|rest] >> 1
      working = {1'b0, mant_in[MANT+4:1]}; // new carry position cleared
      exp_out = exp_in + 1'b1;
    end else begin
      // count leading zeros on normalized field (width MANT+1)
      lzc = 0;
      for (i = MANT+3; i >= 0; i = i - 1) begin
        if (working[i] == 0) lzc = lzc + 1;
        else break;
      end
      // limit shift by exponent (prevent negative exponent)
      if (lzc > exp_in) lzc = exp_in;
      // shift-left by lzc
      working = (working << lzc);
      exp_out = exp_in - lzc;
    end

    // extract candidate significand and GRS
    // after normalization top bit should be at position MANT+3 or MANT+4 depending
    // Use bits [MANT+3:...] for mantissa selection (safe indexing)
    sig_shifted = working[MANT+3 -: MANT+1]; // MANT+1 bits including MSB
    // G,R,S are next three bits
    // guard: bit at position MANT+3-(MANT+1) = index 2
    // compute indices
    // For robustness, check bounds
    reg g, r, s;
    g = working[MANT+3-(MANT+1)];
    r = working[MANT+3-(MANT+1)-1];
    s = |working[MANT+3-(MANT+1)-2:0]; // sticky OR

    // set inexact if any GRS are non-zero
    inexact = g | r | s;

    // compute round increment
    reg round_inc;
    case (rnd_mode)
      2'b00: begin // round-to-nearest-even
        round_inc = (g & (r | s | sig_shifted[0]));
      end
      2'b01: begin // toward zero
        round_inc = 1'b0;
      end
      2'b10: begin // toward +inf
        round_inc = (~sign_out) & (g | r | s);
      end
      2'b11: begin // toward -inf
        round_inc = sign_out & (g | r | s);
      end
      default: round_inc = 1'b0;
    endcase

    // add rounding increment
    reg [MANT:0] rounded;
    rounded = sig_shifted + round_inc;

    // handle rounding carry-out
    if (rounded[MANT] == 1'b1 && sig_shifted[MANT] == 1'b1 && round_inc) begin
      // carry caused extra MSB, perform right shift and exponent increment
      rounded = rounded >> 1;
      exp_out = exp_out + 1'b1;
    end

    // extract final mantissa (drop implicit 1)
    mant_out = rounded[MANT-1:0];

    // handle overflow/underflow flags
    if (exp_out == {EXP{1'b1}}) overflow = 1;
    if (exp_out == {EXP{1'b0}}) underflow = inexact; // approximate subnormal handling
  end
endmodule