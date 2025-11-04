module decompressor_2to4 #(parameter W=16) (
  input  wire               clk,
  input  wire               rst_n,
  input  wire               valid_in,                 // new block valid
  input  wire [2*W-1:0]     packed_vals,              // packed nonzeros (MSB first)
  input  wire [3:0]         mask,                     // presence mask bit0->pos0
  output reg                valid_out,
  output reg  [W-1:0]       out0, out1, out2, out3    // expanded outputs
);
  // combinational expansion
  integer i; integer j; reg [W-1:0] tmp_out [0:3];
  always @* begin
    j = 0;
    for (i=0;i<4;i=i+1) begin
      if (mask[i]) begin
        tmp_out[i] = packed_vals[(2-j-1)*W +: W]; // consume next packed (MSB-first)
        j = j + 1;
      end else tmp_out[i] = {W{1'b0}};
    end
  end
  // register outputs
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin valid_out <= 1'b0; out0<=0; out1<=0; out2<=0; out3<=0; end
    else begin
      if (valid_in) begin
        out0 <= tmp_out[0]; out1 <= tmp_out[1];
        out2 <= tmp_out[2]; out3 <= tmp_out[3];
        valid_out <= 1'b1;
      end else valid_out <= 1'b0;
    end
  end
endmodule