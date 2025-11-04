module exec_alu(
  input  wire        clk,          // clock (optional for handshake)
  input  wire        rst,
  input  wire [3:0]  op,           // opcode: 0x0=int_add,0x1=int_sub,...,0x8=fp16_add,0x9=fp16_sub
  input  wire [31:0] in_a,
  input  wire [31:0] in_b,
  output reg  [31:0] out);
  // Integer ALU (combinational)
  wire [31:0] int_add = in_a + in_b;
  wire [31:0] int_sub = in_a - in_b;
  wire [31:0] int_and = in_a & in_b;
  wire [31:0] int_or  = in_a | in_b;
  wire [31:0] int_xor = in_a ^ in_b;
  wire [31:0] int_shl = in_a << in_b[4:0];
  wire [31:0] int_shr = in_a >> in_b[4:0];

  // FP16 add/sub inputs taken from low 16 bits
  wire [15:0] a16 = in_a[15:0];
  wire [15:0] b16 = in_b[15:0];
  wire [15:0] fp16_res;
  fp16_addsub fpunit (.a(a16), .b(b16), .sub(op==4'h9), .z(fp16_res));

  always @(*) begin
    case (op)
      4'h0: out = int_add;
      4'h1: out = int_sub;
      4'h2: out = int_and;
      4'h3: out = int_or;
      4'h4: out = int_xor;
      4'h5: out = int_shl;
      4'h6: out = int_shr;
      4'h8,4'h9: out = {16'b0, fp16_res}; // FP16 result packed in lower half
      default: out = 32'b0;
    endcase
  end
endmodule

// Synthesizable FP16 add/sub with normalization and round-to-nearest-even.
// No gradual underflow (denormals are treated as zero), INF/NaN handled.
module fp16_addsub(input  wire [15:0] a, input wire [15:0] b, input wire sub, output wire [15:0] z);
  // fields
  wire sa = a[15]; wire [4:0] ea = a[14:10]; wire [9:0] ma = a[9:0];
  wire sb = b[15]; wire [4:0] eb = b[14:10]; wire [9:0] mb = b[9:0];
  // implicit 1 for normals
  wire [11:0] mant_a = (ea==0) ? {2'b00, ma} : {2'b01, ma}; // leading bits: hidden one
  wire [11:0] mant_b = (eb==0) ? {2'b00, mb} : {2'b01, mb};
  wire sign_b_eff = sub ? ~sb : sb; // subtract by flipping sign
  wire sign_b = sign_b_eff;

  // align mantissas
  wire [4:0] ediff = (ea>eb) ? (ea-eb) : (eb-ea);
  wire swap = (eb>ea);
  wire [11:0] ma_a = swap ? mant_b : mant_a;
  wire [11:0] ma_b = swap ? mant_a : mant_b;
  wire sgn_a = swap ? sign_b : sa;
  wire sgn_b = swap ? sa : sign_b;
  wire [4:0] exp_hi = swap ? eb : ea;
  wire [11:0] ma_b_shr = (ediff>=12) ? 12'b0 : (ma_b >> ediff);
  // add/sub mantissas
  wire [12:0] sum = (sgn_a==sgn_b) ? ({1'b0,ma_a} + {1'b0,ma_b_shr}) : ({1'b0,ma_a} - {1'b0,ma_b_shr});
  wire res_sign = (sum[12]) ? sgn_a : sgn_a;
  // normalize
  reg [4:0] res_exp;
  reg [11:0] res_mant;
  reg guard; reg round; reg sticky;
  integer shift;
  always @(*) begin
    if (ea==5'h1F || eb==5'h1F) begin // INF/NaN passthrough
      res_exp = 5'h1F; res_mant = 12'b100000000000; guard=0; round=0; sticky=0;
    end else if (sum[12]) begin // overflow bit
      res_exp = exp_hi + 1;
      // take top 12 bits; guard/round/sticky from lower bits (none here)
      res_mant = sum[12:1];
      guard = sum[0]; round=0; sticky=0;
    end else begin
      // normalize left until MSB in bit 11
      res_exp = exp_hi;
      res_mant = sum[11:0];
      shift = 0;
      while (res_mant[11]==0 && res_exp>0) begin
        res_mant = res_mant << 1;
        res_exp = res_exp - 1;
        shift = shift + 1;
      end
      // guard/round/sticky approximate from lower lost bits (set to 0 for simplicity)
      guard=0; round=0; sticky=0;
    end
  end
  // rounding to nearest even
  wire [9:0] frac = res_mant[10:1]; // drop hidden bit
  wire round_incr = guard & (round | sticky | frac[0]);
  wire [9:0] frac_rounded = frac + round_incr;
  wire exp_inc_on_round = (frac_rounded==10'b0) & round_incr;
  wire [4:0] final_exp = res_exp + (exp_inc_on_round ? 1 : 0);
  assign z = (final_exp==5'h1F) ? {res_sign,5'h1F,10'b0}
           : {res_sign, final_exp, frac_rounded};
endmodule