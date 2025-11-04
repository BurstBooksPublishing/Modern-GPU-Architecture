module dual_clock_fifo #(
  parameter DATA_WIDTH = 32,
  parameter ADDR_WIDTH = 8
)(
  input  wire                     wr_clk,
  input  wire                     rd_clk,
  input  wire                     rst,      // synchronous reset for clarity
  input  wire                     wr_en,
  input  wire [DATA_WIDTH-1:0]    din,
  input  wire                     rd_en,
  output reg  [DATA_WIDTH-1:0]    dout,
  output wire                     full,
  output wire                     empty
);
  localparam DEPTH = (1<> 1));
  wire [ADDR_WIDTH:0] rd_gray = (rd_bin ^ (rd_bin >> 1));

  // synchronizers
  reg [ADDR_WIDTH:0] wr_gray_sync1 = 0, wr_gray_sync2 = 0;
  reg [ADDR_WIDTH:0] rd_gray_sync1 = 0, rd_gray_sync2 = 0;

  // convert gray to binary (function)
  function [ADDR_WIDTH:0] gray2bin;
    input [ADDR_WIDTH:0] g;
    integer i;
    begin
      gray2bin[ADDR_WIDTH] = g[ADDR_WIDTH];
      for (i = ADDR_WIDTH-1; i >= 0; i = i-1)
        gray2bin[i] = gray2bin[i+1] ^ g[i];
    end
  endfunction

  // write domain: memory write and write pointer
  always @(posedge wr_clk) begin
    if (rst) begin
      wr_bin <= 0;
      // sync rd pointer mirrors reset
      rd_gray_sync1 <= 0;
      rd_gray_sync2 <= 0;
    end else begin
      // synchronize remote read pointer into write domain
      rd_gray_sync1 <= rd_gray;
      rd_gray_sync2 <= rd_gray_sync1;

      if (wr_en && !full) begin
        mem[wr_bin[ADDR_WIDTH-1:0]] <= din; // write data
        wr_bin <= wr_bin + 1'b1;
      end
    end
  end

  // read domain: memory read and read pointer
  always @(posedge rd_clk) begin
    if (rst) begin
      rd_bin <= 0;
      wr_gray_sync1 <= 0;
      wr_gray_sync2 <= 0;
      dout <= {DATA_WIDTH{1'b0}};
    end else begin
      // synchronize remote write pointer into read domain
      wr_gray_sync1 <= wr_gray;
      wr_gray_sync2 <= wr_gray_sync1;

      if (rd_en && !empty) begin
        dout <= mem[rd_bin[ADDR_WIDTH-1:0]]; // read data
        rd_bin <= rd_bin + 1'b1;
      end
    end
  end

  // compute synchronized binary pointers for status
  wire [ADDR_WIDTH:0] rd_bin_sync = gray2bin(rd_gray_sync2);
  wire [ADDR_WIDTH:0] wr_bin_sync = gray2bin(wr_gray_sync2);

  // full and empty signals evaluated in respective domains
  assign full  = ((wr_bin + 1'b1) == rd_bin_sync); // evaluated in write clock domain
  assign empty = (rd_bin == wr_bin_sync);          // evaluated in read clock domain

endmodule