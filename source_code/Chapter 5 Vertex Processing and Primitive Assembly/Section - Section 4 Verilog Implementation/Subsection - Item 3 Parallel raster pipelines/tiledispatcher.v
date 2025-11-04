module tile_dispatcher #(
  parameter NUM_PIPELINES = 4,
  parameter TILE_ID_WIDTH = 12,
  parameter FIFO_DEPTH = 16,
  parameter PTR_WIDTH = $clog2(FIFO_DEPTH)
)(
  input  wire                         clk,
  input  wire                         rst_n,
  // input from binning stage
  input  wire                         enqueue_valid,
  input  wire  [TILE_ID_WIDTH-1:0]    enqueue_tile_id,
  output wire                         enqueue_ready,
  // pipeline interfaces
  input  wire  [NUM_PIPELINES-1:0]    pipeline_ready, // pipeline ready to accept
  input  wire  [NUM_PIPELINES-1:0]    pipeline_done,  // pipeline finished tile
  output reg   [NUM_PIPELINES-1:0]    out_valid,
  output reg   [TILE_ID_WIDTH-1:0]    out_tile_id [0:NUM_PIPELINES-1]
);

  // simple circular FIFO
  reg [TILE_ID_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];
  reg [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
  reg [PTR_WIDTH:0] fifo_count;

  assign enqueue_ready = (fifo_count < FIFO_DEPTH);

  // enqueue logic
  always @(posedge clk) begin
    if (!rst_n) begin
      wr_ptr <= 0; rd_ptr <= 0; fifo_count <= 0;
    end else begin
      if (enqueue_valid & (fifo_count < FIFO_DEPTH)) begin
        fifo_mem[wr_ptr] <= enqueue_tile_id;                 // store tile id
        wr_ptr <= (wr_ptr == FIFO_DEPTH-1) ? 0 : wr_ptr + 1;
        fifo_count <= fifo_count + 1;
      end
      // assign tiles to pipelines when available
      if (fifo_count > 0) begin
        integer i;
        for (i = 0; i < NUM_PIPELINES; i = i + 1) begin
          if (pipeline_ready[i] && !out_valid[i]) begin
            out_tile_id[i] <= fifo_mem[rd_ptr];
            rd_ptr <= (rd_ptr == FIFO_DEPTH-1) ? 0 : rd_ptr + 1;
            fifo_count <= fifo_count - 1;
            out_valid[i] <= 1'b1; // pipeline now has work
          end
        end
      end
      // clear valid when pipeline signals done
      integer j;
      for (j = 0; j < NUM_PIPELINES; j = j + 1) begin
        if (pipeline_done[j]) out_valid[j] <= 1'b0;
      end
    end
  end
endmodule