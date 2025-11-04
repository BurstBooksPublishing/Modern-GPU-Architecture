module fixed_fma #(
  parameter integer INTW = 16,    // integer bits
  parameter integer FRACW = 16    // fractional bits
)(
  input  wire                  clk,
  input  wire                  rstn,
  input  wire                  in_valid,
  output wire                  in_ready,
  input  wire signed [(INTW+FRACW)-1:0] a, // fixed-point a
  input  wire signed [(INTW+FRACW)-1:0] b, // fixed-point b
  input  wire signed [(INTW+FRACW+FRACW)-1:0] c, // pre-shifted addend (wider)
  output reg                   out_valid,
  output reg signed [(INTW+FRACW+FRACW)-1:0] out // wide result
);

localparam PRODW = 2*(INTW+FRACW);
localparam ADDW  = PRODW+1; // carry

// simple two-stage pipeline: mul -> add
reg signed [PRODW-1:0] mul_reg;
reg mul_valid;

assign in_ready = 1'b1; // simple flow control; can be extended

always @(posedge clk or negedge rstn) begin
  if (!rstn) begin
    mul_reg  <= 0;
    mul_valid<= 0;
    out_valid<= 0;
    out      <= 0;
  end else begin
    // stage 0: multiply (combinational multiply captured)
    if (in_valid) begin
      mul_reg  <= a * b; // synthesizable signed multiplier
      mul_valid<= 1'b1;
    end else begin
      mul_valid<= 1'b0;
    end

    // stage 1: add fused (single-cycle add + register)
    if (mul_valid) begin
      // align c externally before calling this core; here we just add
      out <= {{(ADDW-PRODW){mul_reg[PRODW-1]}}, mul_reg} + c;
      out_valid <= 1'b1;
    end else begin
      out_valid <= 1'b0;
    end
  end
end

endmodule