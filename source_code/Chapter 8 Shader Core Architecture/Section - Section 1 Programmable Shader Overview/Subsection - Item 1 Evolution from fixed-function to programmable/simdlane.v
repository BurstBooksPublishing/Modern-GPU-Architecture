module simd_lane #(
  parameter W = 32
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 active,       // lane active mask
  input  wire [1:0]           opcode,       // 00:add 01:mul 10:and 11:nop
  input  wire [W-1:0]         in_a,
  input  wire [W-1:0]         in_b,
  output reg  [W-1:0]         out           // registered output
);
  wire [W-1:0] add_r = in_a + in_b;         // combinational add
  wire [W-1:0] mul_r = in_a * in_b;         // combinational multiply
  wire [W-1:0] and_r = in_a & in_b;         // bitwise and

  wire [W-1:0] result = (opcode==2'b00) ? add_r :
                        (opcode==2'b01) ? mul_r :
                        (opcode==2'b10) ? and_r : {W{1'b0}};

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) out <= {W{1'b0}};
    else if (active) out <= result;         // mask controls write
    else out <= out;                        // inactive lanes keep state
  end
endmodule