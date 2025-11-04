module stream_output_engine #(
  parameter ADDR_WIDTH = 32,
  parameter DEPTH = 8
)(
  input  wire                 clk,
  input  wire                 rst_n,
  // Input from shader/emission unit
  input  wire                 in_valid,
  input  wire [127:0]         in_data,
  input  wire                 in_last,      // end-of-primitive marker
  output reg                  in_ready,
  // Memory write/master (simple handshake)
  output reg                  mem_wr_valid,
  output reg [ADDR_WIDTH-1:0] mem_wr_addr,
  output reg [127:0]          mem_wr_data,
  input  wire                 mem_wr_ready,
  // Append counter (atomic reserve simple model)
  output reg [ADDR_WIDTH-1:0] append_counter
);

  // FIFO for one-cluster buffering
  reg [127:0] fifo [0:DEPTH-1];
  reg [$clog2(DEPTH):0] wr_ptr, rd_ptr, cnt;

  // DMA FSM
  localparam IDLE = 0, WRITE = 1;
  reg state;

  // Simple append: reserve space on first beat when in_valid & in_ready
  always @(posedge clk) begin
    if (!rst_n) begin
      wr_ptr <= 0; rd_ptr <= 0; cnt <= 0;
      in_ready <= 1'b1; mem_wr_valid <= 1'b0;
      mem_wr_addr <= 0; mem_wr_data <= 0;
      append_counter <= 0; state <= IDLE;
    end else begin
      // Accept input if FIFO not full
      if (in_valid && in_ready) begin
        fifo[wr_ptr] <= in_data;
        wr_ptr <= wr_ptr + 1;
        cnt <= cnt + 1;
        // reserve space on append when first beat (simple model)
        if (cnt == 0) begin
          mem_wr_addr <= append_counter; // base addr
          append_counter <= append_counter + 16; // reserve one beat (example)
        end
      end
      in_ready <= (cnt < DEPTH-1);

      // DMA FSM: issue writes while FIFO not empty
      case (state)
      IDLE: if (cnt > 0) begin
        mem_wr_valid <= 1'b1;
        mem_wr_data <= fifo[rd_ptr];
        state <= WRITE;
      end
      WRITE: if (mem_wr_ready) begin
        rd_ptr <= rd_ptr + 1;
        cnt <= cnt - 1;
        mem_wr_addr <= mem_wr_addr + 16; // next beat address
        if (cnt > 1) begin
          mem_wr_data <= fifo[rd_ptr+1];
          mem_wr_valid <= 1'b1;
          state <= WRITE;
        end else begin
          mem_wr_valid <= 1'b0;
          state <= IDLE;
        end
      end
      endcase
    end
  end

endmodule