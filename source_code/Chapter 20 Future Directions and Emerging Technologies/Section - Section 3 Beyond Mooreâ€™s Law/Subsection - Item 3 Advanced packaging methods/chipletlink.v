module chiplet_link_ctrl #(
  parameter LANES = 4,
  parameter DATA_WIDTH = 32,
  parameter FIFO_DEPTH = 8,
  parameter CREDIT_WIDTH = 8
)(
  input  wire                     clk,
  input  wire                     rstn,
  // TX side
  input  wire [DATA_WIDTH-1:0]    tx_data,
  input  wire                     tx_valid,
  output wire                     tx_ready,
  output reg  [CREDIT_WIDTH-1:0]  tx_credit_out, // advertise credits to remote
  // RX side
  output reg  [DATA_WIDTH-1:0]    rx_data,
  output reg                      rx_valid,
  input  wire                     rx_ready,
  input  wire [CREDIT_WIDTH-1:0]  rx_credit_in   // remote credits
);

  // Simple FIFO (circular buffer) for TX
  reg [DATA_WIDTH-1:0] tx_fifo [0:FIFO_DEPTH-1];
  reg [$clog2(FIFO_DEPTH):0] tx_wr_ptr, tx_rd_ptr;
  reg [$clog2(FIFO_DEPTH):0] tx_count;
  reg [CREDIT_WIDTH-1:0]     rem_credits;

  // TX write
  always @(posedge clk) begin
    if(!rstn) begin
      tx_wr_ptr <= 0; tx_count <= 0;
    end else if (tx_valid && tx_ready) begin
      tx_fifo[tx_wr_ptr[$clog2(FIFO_DEPTH)-1:0]] <= tx_data; // write data
      tx_wr_ptr <= tx_wr_ptr + 1;
      tx_count <= tx_count + 1;
    end
  end

  // TX readiness: can accept new data if FIFO not full
  assign tx_ready = (tx_count < FIFO_DEPTH-1);

  // Remotes credit update (simple model)
  always @(posedge clk) begin
    if(!rstn) begin
      rem_credits <= 0;
      tx_credit_out <= FIFO_DEPTH; // advertise available buffer
    end else begin
      rem_credits <= rx_credit_in; // assume credit feedback arrives timely
      tx_credit_out <= FIFO_DEPTH - tx_count; // advertise local free slots
    end
  end

  // Transmit over LANES when credits available and FIFO not empty
  always @(posedge clk) begin
    if(!rstn) begin
      tx_rd_ptr <= 0; rx_valid <= 0; rx_data <= 0;
    end else begin
      if (tx_count > 0 && rem_credits > 0) begin
        // send one beat per clock per LANES combined; simplified to DATA_WIDTH
        rx_data <= tx_fifo[tx_rd_ptr[$clog2(FIFO_DEPTH)-1:0]];
        rx_valid <= 1;
        tx_rd_ptr <= tx_rd_ptr + 1;
        tx_count <= tx_count - 1;
      end else if (rx_ready) begin
        rx_valid <= 0; // consumer accepted
      end
    end
  end

endmodule