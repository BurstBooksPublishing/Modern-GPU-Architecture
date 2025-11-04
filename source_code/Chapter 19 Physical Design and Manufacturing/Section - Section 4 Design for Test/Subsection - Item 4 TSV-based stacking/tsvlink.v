module tsv_link #(parameter DATA_W = 64, parameter DEPTH = 8) (
    input  wire                   clk,
    input  wire                   rst_n,
    // TX side (local die)
    input  wire [DATA_W-1:0]      tx_data,
    input  wire                   tx_valid,
    output wire                   tx_ready,
    // RX side (remote die)
    output reg  [DATA_W-1:0]      rx_data,
    output reg                    rx_valid,
    input  wire                   rx_ready,
    output reg                    rx_error
);
  // Simple synchronous FIFO storage
  reg [DATA_W-1:0] fifo [0:DEPTH-1];
  reg [$clog2(DEPTH):0] wr_ptr, rd_ptr, count;
  reg parity_in, parity_out;

  // parity generation (even parity)
  wire pgen = ^tx_data;

  // Write logic
  assign tx_ready = (count < DEPTH);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= 0; count <= 0; parity_in <= 0;
    end else if (tx_valid && tx_ready) begin
      fifo[wr_ptr] <= tx_data;
      parity_in <= pgen;
      wr_ptr <= wr_ptr + 1;
      count <= count + 1;
    end
  end

  // Read logic with parity check
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_ptr <= 0; rx_valid <= 0; rx_data <= 0; rx_error <= 0;
    end else if ((count > 0) && (!rx_valid || (rx_valid && rx_ready))) begin
      rx_data <= fifo[rd_ptr];
      parity_out <= ^fifo[rd_ptr];
      rx_error <= (parity_out != parity_in); // simple check across path
      rx_valid <= 1;
      rd_ptr <= rd_ptr + 1;
      count <= count - 1;
    end else if (rx_valid && rx_ready) begin
      rx_valid <= 0;
    end
  end
endmodule