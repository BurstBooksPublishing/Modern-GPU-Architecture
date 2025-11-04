module bilinear_interp #(
  parameter CH = 4, parameter CHW = 8, parameter FRAC = 8
)(
  input  wire              clk,
  input  wire              valid_in,
  input  wire [31:0]       tex00, // RGBA packed 8b per channel
  input  wire [31:0]       tex10,
  input  wire [31:0]       tex01,
  input  wire [31:0]       tex11,
  input  wire [FRAC-1:0]   u_frac,
  input  wire [FRAC-1:0]   v_frac,
  output reg  [31:0]       color_out,
  output reg               valid_out
);
  // local widths
  localparam W = CHW + FRAC + 1; // accumulator width
  // stage 1 registers
  reg [31:0] t00, t10, t01, t11;
  reg [FRAC-1:0] uf, vf;
  always @(posedge clk) begin
    t00 <= tex00; t10 <= tex10; t01 <= tex01; t11 <= tex11;
    uf  <= u_frac; vf  <= v_frac;
    valid_out <= valid_in; // single-cycle pipeline latency for simplicity
  end
  // compute weights (expanded to 16-bit)
  wire [FRAC:0] one = {1'b0, {FRAC{1'b1}}}; // value 2^FRAC-1
  wire [FRAC*2+1:0] w00 = (one - uf) * (one - vf);
  wire [FRAC*2+1:0] w10 = (uf)       * (one - vf);
  wire [FRAC*2+1:0] w01 = (one - uf) * (vf);
  wire [FRAC*2+1:0] w11 = (uf)       * (vf);
  // per-channel accumulation
  integer i;
  reg [W-1:0] acc[0:3];
  always @(posedge clk) begin
    // unpack channels and multiply-accumulate
    for (i=0;i<4;i=i+1) begin
      acc[i] <= ( (t00 >> (i*8)) & 8'hFF ) * w00 +
                ( (t10 >> (i*8)) & 8'hFF ) * w10 +
                ( (t01 >> (i*8)) & 8'hFF ) * w01 +
                ( (t11 >> (i*8)) & 8'hFF ) * w11;
    end
    // normalize and pack (shift by 2*FRAC) with rounding
    color_out <= { 
      ((acc[3] + (1 << (2*FRAC-1))) >> (2*FRAC)) & 8'hFF,
      ((acc[2] + (1 << (2*FRAC-1))) >> (2*FRAC)) & 8'hFF,
      ((acc[1] + (1 << (2*FRAC-1))) >> (2*FRAC)) & 8'hFF,
      ((acc[0] + (1 << (2*FRAC-1))) >> (2*FRAC)) & 8'hFF
    };
  end
endmodule