module mem_protection #(
  parameter ADDR_W = 32,
  parameter CTX_W  = 8,
  parameter N_RULE = 4
)(
  input  wire                  clk,
  input  wire                  rst,
  input  wire [CTX_W-1:0]      ctx_id,       // context tag
  input  wire [ADDR_W-1:0]     req_addr,     // request address
  input  wire                  req_valid,
  output reg                   grant,
  output reg                   fault
);
  // simple rule table: base/limit per context index (synthesizable RAM)
  reg [ADDR_W-1:0] base   [0:N_RULE-1];
  reg [ADDR_W-1:0] limit  [0:N_RULE-1];
  integer i;
  // rule init (synthesizable constants, replace with config path)
  initial begin
    base[0]=32'h0000_0000; limit[0]=32'h0FFF_FFFF;
    base[1]=32'h1000_0000; limit[1]=32'h1FFF_FFFF;
    base[2]=32'h2000_0000; limit[2]=32'h2FFF_FFFF;
    base[3]=32'h3000_0000; limit[3]=32'h3FFF_FFFF;
  end
  always @(posedge clk) begin
    if (rst) begin grant <= 1'b0; fault <= 1'b0; end
    else if (req_valid) begin
      // simple match: ctx_id selects rule index (low bits)
      i = ctx_id % N_RULE;
      if (req_addr >= base[i] && req_addr <= limit[i]) begin
        grant <= 1'b1; fault <= 1'b0;
      end else begin
        grant <= 1'b0; fault <= 1'b1;
      end
    end
  end
endmodule