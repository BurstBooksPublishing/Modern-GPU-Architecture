module dual_clock_fifo #(
  parameter WIDTH = 128,
  parameter ADDR_WIDTH = 8              // FIFO depth = 2^ADDR_WIDTH
) (
  input  wire                 wr_clk,
  input  wire                 rd_clk,
  input  wire                 rst,       // synchronous reset to both domains
  input  wire                 wr_en,
  input  wire                 rd_en,
  input  wire [WIDTH-1:0]     din,
  output reg  [WIDTH-1:0]     dout,
  output wire                 full,
  output wire                 empty
);
  localparam DEPTH = (1<> 1);
      end
    end
    // sync read pointer into write clock domain
    rd_ptr_gray_sync1 <= rd_ptr_gray;
    rd_ptr_gray_sync2 <= rd_ptr_gray_sync1;
  end

  // read domain
  always @(posedge rd_clk) begin
    if (rst) begin
      rd_ptr_bin  <= 0;
      rd_ptr_gray <= 0;
      dout <= 0;
    end else begin
      if (rd_en && !empty) begin
        dout <= mem[rd_ptr_bin[ADDR_WIDTH-1:0]]; // read data
        rd_ptr_bin <= rd_ptr_bin + 1;
        rd_ptr_gray<= (rd_ptr_bin + 1) ^ ((rd_ptr_bin + 1) >> 1);
      end
    end
    // sync write pointer into read clock domain
    wr_ptr_gray_sync1 <= wr_ptr_gray;
    wr_ptr_gray_sync2 <= wr_ptr_gray_sync1;
  end

  // convert synchronized gray to binary for status checks
  function [ADDR_WIDTH:0] gray2bin(input [ADDR_WIDTH:0] g);
    integer i;
    begin
      gray2bin = 0;
      for (i=0;i<=ADDR_WIDTH;i=i+1)
        gray2bin = gray2bin ^ (g >> i);
    end
  endfunction

  wire [ADDR_WIDTH:0] rd_ptr_bin_sync = gray2bin(rd_ptr_gray_sync2);
  wire [ADDR_WIDTH:0] wr_ptr_bin_sync = gray2bin(wr_ptr_gray_sync2);

  assign full  = ( (wr_ptr_gray == {~rd_ptr_gray_sync2[ADDR_WIDTH:ADDR_WIDTH-1], rd_ptr_gray_sync2[ADDR_WIDTH-2:0]}) );
  assign empty = (wr_ptr_gray_sync2 == rd_ptr_gray);

endmodule