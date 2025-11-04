module comb_alu #(
  parameter WIDTH = 32
)(
  input  wire [1:0]  opcode,        // 2-bit operation select
  input  wire [WIDTH-1:0] a, b,     // operands
  output reg  [WIDTH-1:0] y,        // combinational result
  output reg           zero        // zero flag
);
  // combinational logic: assign all outputs on every path
  always @* begin
    case (opcode)
      2'b00: y = a + b;   // add
      2'b01: y = a - b;   // sub
      2'b10: y = a & b;   // and
      2'b11: y = a | b;   // or
      default: y = {WIDTH{1'b0}}; // defensive default
    endcase
    zero = (y == {WIDTH{1'b0}}); // always assigned
  end
endmodule