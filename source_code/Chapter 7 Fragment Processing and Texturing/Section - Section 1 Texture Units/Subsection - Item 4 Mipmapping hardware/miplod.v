module mip_lod_calc
#(parameter FP = 16, LEVELS = 12) // FP = Q1.(FP-1)
(
  input  signed [FP-1:0] dudx, dvdx, dudy, dvdy,
  output reg [$clog2(LEVELS)-1:0] lod_int,
  output reg [7:0] lod_frac // 8-bit fractional for trilinear
);
  // absolute values
  wire [FP-1:0] ax_dx = dudx[FP-1] ? -dudx : dudx;
  wire [FP-1:0] ay_dx = dvdx[FP-1] ? -dvdx : dvdx;
  wire [FP-1:0] ax_dy = dudy[FP-1] ? -dudy : dudy;
  wire [FP-1:0] ay_dy = dvdy[FP-1] ? -dvdy : dvdy;

  // squares (unsigned)
  localparam SW = 2*FP;
  wire [SW-1:0] ex = ax_dx * ax_dx + ay_dx * ay_dx;
  wire [SW-1:0] ey = ax_dy * ax_dy + ay_dy * ay_dy;
  wire [SW-1:0] emax = (ex >= ey) ? ex : ey;

  // leading-one detection (find exponent)
  integer i;
  reg [$clog2(SW+1)-1:0] lz;
  always @(*) begin
    lz = 0;
    for (i = SW-1; i >= 0; i = i - 1)
      if (emax[i]) begin lz = SW-1-i; disable for; end
    // exponent = position of MSB
    // exponent fits in small range for practical FP widths
    // normalize mantissa: shift left to align MSB at bit (SW-1)
    // take top M bits as mantissa approximation
    reg [SW-1:0] norm = emax << lz;
    reg [7:0] mant_top = norm[SW-1 -: 8]; // top 8 bits of mantissa
    // approximate log2(mantissa) by linear mapping around [1,2)
    // mant_top ranges [0x80,0xFF] for normalized values; map to 0..255
    reg [15:0] mant_frac = (mant_top - 8'h80) << 8; // scaled fraction
    // compute lambda_fixed = 0.5*(exp + log2(mantissa))
    // exp = (SW-1 - lz), scaled by 256
    reg [23:0] exp_scaled = (SW-1 - lz) << 8;
    reg [23:0] lambda_scaled = (exp_scaled + mant_frac) >> 1; // divide by 2
    // output integer and fractional parts
    integer lod_full = lambda_scaled >> 8;
    if (lod_full < 0) lod_int = 0;
    else if (lod_full >= LEVELS) lod_int = LEVELS-1;
    else lod_int = lod_full[$clog2(LEVELS)-1:0];
    lod_frac = lambda_scaled[7:0];
  end
endmodule