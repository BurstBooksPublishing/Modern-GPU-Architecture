module sm_mem_link #(
  parameter DATA_W = 128, // payload width
  parameter DEPTH = 8     // fifo depth (number of packets)
)(
  input  wire clk,
  input  wire rst,
  // SM side (producer)
  input  wire                   sm_valid,
  input  wire [DATA_W-1:0]      sm_data,
  output reg                    sm_ready,
  // Partition side (consumer)
  output reg                    part_valid,
  output reg  [DATA_W-1:0]      part_data,
  input  wire                   part_ready,
  // Credit return from partition (for responses) - simple decrement
  input  wire [$clog2(DEPTH):0] credits_in,
  output reg  [$clog2(DEPTH):0] credits_out
);

  // simple synchronous FIFO
  reg [DATA_W-1:0] fifo_mem [0:DEPTH-1];
  reg [$clog2(DEPTH):0] wr_ptr, rd_ptr, count;

  // write side
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr <= 0; count <= 0;
      sm_ready <= 1'b1;
    end else begin
      if (sm_valid && sm_ready) begin
        fifo_mem[wr_ptr] <= sm_data; // write packet
        wr_ptr <= wr_ptr + 1;
        count <= count + 1;
      end
      // throttle SM when FIFO is full
      sm_ready <= (count < DEPTH-1);
    end
  end

  // read side to partition
  always @(posedge clk) begin
    if (rst) begin
      rd_ptr <= 0; part_valid <= 1'b0; part_data <= {DATA_W{1'b0}};
      credits_out <= 0;
    end else begin
      // supply credits seen from partition upstream (simple pass-through)
      credits_out <= credits_in;
      if (!part_valid && (count > 0)) begin
        part_data <= fifo_mem[rd_ptr];
        part_valid <= 1'b1;
      end
      if (part_valid && part_ready) begin
        rd_ptr <= rd_ptr + 1;
        part_valid <= 1'b0;
        count <= count - 1;
      end
    end
  end

endmodule