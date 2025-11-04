module ddi_link_endpoint #(
  parameter FLIT_WIDTH = 256,
  parameter FIFO_DEPTH = 8,
  parameter CREDIT_WIDTH = 4
)(
  input  wire                     clk,
  input  wire                     rst_n,
  // Transmit side
  input  wire [FLIT_WIDTH-1:0]    tx_flit,
  input  wire                     tx_valid,
  output reg                      tx_ready,
  output reg [CREDIT_WIDTH-1:0]   tx_credits_out, // advertise free entries
  // Receive side
  output reg [FLIT_WIDTH-1:0]     rx_flit,
  output reg                      rx_valid,
  input  wire                     rx_ready,
  input  wire [CREDIT_WIDTH-1:0]  rx_credits_in  // remote credits returned
);
  // Simple FIFO for elastic buffering
  reg [FLIT_WIDTH-1:0] fifo [0:FIFO_DEPTH-1];
  reg [$clog2(FIFO_DEPTH):0] wr_ptr, rd_ptr, count;
  reg [CREDIT_WIDTH-1:0] local_credits;

  // parity bit per flit for error detect
  wire tx_parity = ^tx_flit;
  reg parity_error;

  // Transmit logic: push when tx_valid and credit available
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= 0; rd_ptr <= 0; count <= 0;
      tx_ready <= 1'b0; local_credits <= FIFO_DEPTH;
      tx_credits_out <= FIFO_DEPTH;
    end else begin
      // update credits when remote returns credits
      if (rx_credits_in != 0) begin
        local_credits <= local_credits + rx_credits_in;
      end
      tx_credits_out <= FIFO_DEPTH - count; // advertise available entries
      tx_ready <= (local_credits > 0) && (count < FIFO_DEPTH);
      if (tx_valid && tx_ready) begin
        fifo[wr_ptr] <= tx_flit;
        wr_ptr <= (wr_ptr + 1) % FIFO_DEPTH;
        count <= count + 1;
        local_credits <= local_credits - 1;
      end
      // transmit to receiver when available and receiver ready
      if ((count > 0) && rx_ready) begin
        rx_flit <= fifo[rd_ptr];
        rx_valid <= 1'b1;
        rd_ptr <= (rd_ptr + 1) % FIFO_DEPTH;
        count <= count - 1;
      end else begin
        if (rx_ready) rx_valid <= 1'b0;
      end
    end
  end

  // parity check on receive
  always @(posedge clk) begin
    if (rx_valid && rx_ready) begin
      parity_error <= (^(rx_flit) != tx_parity); // simple detect (example)
    end
  end
endmodule