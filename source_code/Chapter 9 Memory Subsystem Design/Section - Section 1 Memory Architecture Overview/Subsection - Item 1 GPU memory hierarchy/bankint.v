module bank_interleaver #(
  parameter ADDR_WIDTH = 32,
  parameter BANKS = 8,
  parameter SWIZZLE = 1
)(
  input  wire [ADDR_WIDTH-1:0] addr,
  output wire [$clog2(BANKS)-1:0] bank_idx,
  output wire [ADDR_WIDTH-$clog2(BANKS)-1:0] row
);
  localparam BANK_BITS = $clog2(BANKS);
  wire [BANK_BITS-1:0] low = addr[BANK_BITS-1:0]; // low bits
  wire [ADDR_WIDTH-1:0] high = addr >> BANK_BITS;  // remaining bits
  wire [BANK_BITS-1:0] swizzled = SWIZZLE ? (low ^ high[BANK_BITS-1:0]) : low;
  assign bank_idx = swizzled;
  assign row = high;
endmodule
// Synthesizable: purely combinational mapping for memory controller/SM.