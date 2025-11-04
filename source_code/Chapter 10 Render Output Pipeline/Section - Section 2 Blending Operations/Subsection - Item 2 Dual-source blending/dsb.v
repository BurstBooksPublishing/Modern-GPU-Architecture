module dual_src_blend(
  input  wire         clk,
  input  wire         valid_in,              // input handshake
  input  wire [31:0]  src0_in,               // {R,G,B,A} each 8-bit
  input  wire [31:0]  src1_in,               // second color
  input  wire [31:0]  dst_in,                // framebuffer readback
  input  wire [3:0]   factor_sel,            // 4-bit selector
  output reg  [31:0]  out_color,
  output reg          valid_out
);
  // Factor selector encodings
  localparam F_ZERO = 4'd0, F_ONE = 4'd1, F_SRC1 = 4'd2, F_ONE_MINUS_SRC1 = 4'd3;
  // helper unpack
  wire [7:0] s0 [3:0]; assign {s0[3],s0[2],s0[1],s0[0]} = src0_in;
  wire [7:0] s1 [3:0]; assign {s1[3],s1[2],s1[1],s1[0]} = src1_in;
  wire [7:0] d  [3:0]; assign {d[3],d[2],d[1],d[0]}    = dst_in;
  integer i;
  always @(posedge clk) begin
    valid_out <= valid_in; // simple pipelining of valid
    if (valid_in) begin
      for (i=0;i<4;i=i+1) begin
        // select factor per-channel (use src1 channel as factor if selected)
        wire [7:0] f = (factor_sel==F_ZERO) ? 8'd0 :
                       (factor_sel==F_ONE)  ? 8'd255 :
                       (factor_sel==F_SRC1) ? s1[i] :
                       (factor_sel==F_ONE_MINUS_SRC1) ? (8'd255 - s1[i]) : 8'd0;
        // multiply and accumulate in 16-bit domain, then normalize by 255 with rounding
        wire [15:0] term_s = s0[i] * f;
        wire [15:0] term_d = d[i] * (8'd255 - f); // complementary for demo
        reg [15:0] sum = term_s + term_d;
        // normalize and saturate to 8 bits (add 127 for rounding)
        reg [15:0] scaled = (sum + 16'd127) / 16'd255;
        out_color[8*i +: 8] <= (scaled[7:0] > 8'd255) ? 8'd255 : scaled[7:0];
      end
    end
  end
endmodule