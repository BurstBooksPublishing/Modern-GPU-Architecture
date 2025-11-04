module dc_fifo #(
  parameter DATA_WIDTH = 128,
  parameter ADDR_WIDTH = 6  // depth = 2^ADDR_WIDTH
)(
  input  wire                    wr_clk,
  input  wire                    rd_clk,
  input  wire                    arst_n,    // async active-low reset
  input  wire                    wr_en,
  input  wire [DATA_WIDTH-1:0]   wr_data,
  input  wire                    rd_en,
  output reg  [DATA_WIDTH-1:0]   rd_data,
  output wire                    full,
  output wire                    empty
);
  localparam DEPTH = (1<Gray functions
  function [ADDR_WIDTH:0] bin2gray(input [ADDR_WIDTH:0] b);
    bin2gray = b ^ (b >> 1);
  endfunction
  function [ADDR_WIDTH:0] gray2bin(input [ADDR_WIDTH:0] g);
    integer i;
    reg [ADDR_WIDTH:0] b;
    begin
      b[ADDR_WIDTH] = g[ADDR_WIDTH];
      for (i = ADDR_WIDTH-1; i >= 0; i = i-1)
        b[i] = b[i+1] ^ g[i];
      gray2bin = b;
    end
  endfunction

  // write domain
  wire [ADDR_WIDTH-1:0] wr_addr = wr_ptr_bin[ADDR_WIDTH-1:0];
  always @(posedge wr_clk or negedge arst_n) begin
    if (!arst_n) begin
      wr_ptr_bin  <= 0;
      wr_ptr_gray <= 0;
    end else begin
      if (wr_en && !full) begin
        mem[wr_addr] <= wr_data; // write to dual-port mem
        wr_ptr_bin  <= wr_ptr_bin + 1;
        wr_ptr_gray <= bin2gray(wr_ptr_bin + 1);
      end
    end
  end

  // read domain
  wire [ADDR_WIDTH-1:0] rd_addr = rd_ptr_bin[ADDR_WIDTH-1:0];
  always @(posedge rd_clk or negedge arst_n) begin
    if (!arst_n) begin
      rd_ptr_bin  <= 0;
      rd_ptr_gray <= 0;
      rd_data     <= {DATA_WIDTH{1'b0}};
    end else begin
      if (rd_en && !empty) begin
        rd_data <= mem[rd_addr]; // read from dual-port mem
        rd_ptr_bin  <= rd_ptr_bin + 1;
        rd_ptr_gray <= bin2gray(rd_ptr_bin + 1);
      end
    end
  end

  // synchronizers: rd domain samples wr_gray, wr domain samples rd_gray
  always @(posedge rd_clk or negedge arst_n) begin
    if (!arst_n) begin
      wr_ptr_gray_sync1 <= 0; wr_ptr_gray_sync2 <= 0;
    end else begin
      wr_ptr_gray_sync1 <= wr_ptr_gray;
      wr_ptr_gray_sync2 <= wr_ptr_gray_sync1;
    end
  end
  always @(posedge wr_clk or negedge arst_n) begin
    if (!arst_n) begin
      rd_ptr_gray_sync1 <= 0; rd_ptr_gray_sync2 <= 0;
    end else begin
      rd_ptr_gray_sync1 <= rd_ptr_gray;
      rd_ptr_gray_sync2 <= rd_ptr_gray_sync1;
    end
  end

  // full/empty logic using converted binaries
  wire [ADDR_WIDTH:0] wr_sync_bin = gray2bin(wr_ptr_gray_sync2);
  wire [ADDR_WIDTH:0] rd_sync_bin = gray2bin(rd_ptr_gray_sync2);

  assign empty = (wr_sync_bin == rd_ptr_bin);
  // full when next write would equal synchronized read pointer with MSB complement
  wire [ADDR_WIDTH:0] wr_next = wr_ptr_bin + 1;
  assign full  = (wr_next[ADDR_WIDTH:0] == {~rd_sync_bin[ADDR_WIDTH], rd_sync_bin[ADDR_WIDTH-1:0]});

endmodule