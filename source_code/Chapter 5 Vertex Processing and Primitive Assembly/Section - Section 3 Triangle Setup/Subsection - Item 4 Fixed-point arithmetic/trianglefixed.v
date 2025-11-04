module triangle_setup_fixed (
  input  wire signed [31:0] x1, y1, x2, y2, x3, y3, // Q16.16 coords
  output reg  signed [47:0] A1, B1, C1,             // widened edge coeffs
  output reg  signed [31:0] recip_twice_area_q16   // Q16.16 reciprocal
);
  parameter integer F = 16;
  // intermediate wide signed products and accumulators
  reg signed [63:0] t1, t2, t3;
  reg signed [95:0] S; // integer representation of 2*area * 2^{2F}
  reg signed [127:0] numer; // numerator for reciprocal (2^{3F})
  reg signed [127:0] recip_tmp;

  always @* begin
    // edge coefficients: A = y2 - y1, B = x1 - x2, C = x2*y1 - x1*y2
    A1 = $signed(y2) - $signed(y1);                   // still Q16.16
    B1 = $signed(x1) - $signed(x2);
    t1 = $signed(x2) * $signed(y1);
    t2 = $signed(x1) * $signed(y2);
    C1 = t1 - t2;                                      // widen product diff

    // signed integer S = x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2)
    t1 = $signed(x1) * ($signed(y2) - $signed(y3));
    t2 = $signed(x2) * ($signed(y3) - $signed(y1));
    t3 = $signed(x3) * ($signed(y1) - $signed(y2));
    S  = $signed(t1) + $signed(t2) + $signed(t3);     // S is integer scaled by 2^{2F}

    // compute reciprocal R_int = round(2^{3F} / S) to encode 1/(2A) in QF
    if (S == 0) begin
      recip_twice_area_q16 = 32'sd0; // degenerate triangle -> zero reciprocal
    end else begin
      numer = (128'sd1 << (3*F));                     // 2^{3F}
      recip_tmp = numer / S;                          // integer division
      recip_twice_area_q16 = recip_tmp[31:0];        // truncate to Q16.16
    end
  end
endmodule