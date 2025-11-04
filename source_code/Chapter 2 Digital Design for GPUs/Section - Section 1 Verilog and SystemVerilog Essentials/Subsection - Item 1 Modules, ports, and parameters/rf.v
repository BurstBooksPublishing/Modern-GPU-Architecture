module sm_regfile #(
  parameter integer NUM_THREADS = 256,     // total threads in RF
  parameter integer REGS_PER_THREAD = 16,  // logical registers per thread
  parameter integer REG_WIDTH = 32         // bits per register
)(
  input  wire                     clk, rst,
  input  wire [$clog2(NUM_THREADS*REGS_PER_THREAD)-1:0] wr_addr, // write address
  input  wire [REG_WIDTH-1:0]     wr_data,
  input  wire                     wr_en,
  input  wire [$clog2(NUM_THREADS*REGS_PER_THREAD)-1:0] rd_addr, // read address
  output reg  [REG_WIDTH-1:0]     rd_data
);
  localparam integer DEPTH = NUM_THREADS * REGS_PER_THREAD;
  // storage inferred as block RAM / distributed RAM by synthesis
  reg [REG_WIDTH-1:0] mem [0:DEPTH-1];

  // synchronous write, synchronous read (common for BRAM)
  always @(posedge clk) begin
    if (rst) begin
      // optional: zero init; synthesis may map to reset logic or memory init
    end else begin
      if (wr_en) mem[wr_addr] <= wr_data; // write-first behavior
      rd_data <= mem[rd_addr];            // synchronous read
    end
  end
endmodule