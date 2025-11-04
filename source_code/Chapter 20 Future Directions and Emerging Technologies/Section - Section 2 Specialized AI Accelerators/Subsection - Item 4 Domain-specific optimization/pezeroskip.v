module pe_zero_skip #(
  parameter WIDTH = 16
)(
  input  wire                 clk,       // clock
  input  wire                 rst_n,     // active-low reset
  input  wire                 valid_in,  // input valid
  input  wire signed [WIDTH-1:0] a,       // left operand
  input  wire signed [WIDTH-1:0] b,       // top operand
  input  wire                 mask,      // 1=valid MAC, 0=skip
  input  wire signed [31:0]   acc_in,    // incoming accumulator
  output reg  signed [31:0]   acc_out,   // outgoing accumulator
  output reg                  valid_out  // output valid
);
  // simple pipeline: register inputs, perform conditional MAC
  reg signed [WIDTH-1:0] a_r, b_r;
  reg mask_r, v_r;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_r <= 0; b_r <= 0; mask_r <= 0; v_r <= 0;
      acc_out <= 0; valid_out <= 0;
    end else begin
      a_r <= a; b_r <= b; mask_r <= mask; v_r <= valid_in;
      // conditional multiply-accumulate (quantized MAC)
      if (v_r) begin
        if (mask_r)
          acc_out <= acc_in + $signed(a_r) * $signed(b_r);
        else
          acc_out <= acc_in; // skip multiply, forward accumulator
        valid_out <= 1'b1;
      end else begin
        acc_out <= acc_out;
        valid_out <= 1'b0;
      end
    end
  end
endmodule