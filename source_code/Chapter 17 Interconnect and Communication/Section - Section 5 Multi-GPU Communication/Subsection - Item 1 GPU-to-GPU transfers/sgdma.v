module sg_dma #(
  parameter ADDR_WIDTH = 64,
  parameter DATA_WIDTH = 512,
  parameter LEN_WIDTH  = 16
)(
  input  wire                     clk,
  input  wire                     rst_n,
  // Descriptor input (one descriptor at a time)
  input  wire                     desc_valid,
  input  wire [ADDR_WIDTH-1:0]    desc_addr,
  input  wire [LEN_WIDTH-1:0]     desc_len,   // number of beats
  output reg                      desc_ready,
  // Simplified AXI4 read master (single outstanding burst)
  output reg                      arvalid,
  output reg [ADDR_WIDTH-1:0]     araddr,
  output reg [7:0]                arlen,
  input  wire                     arready,
  input  wire                     rvalid,
  input  wire [DATA_WIDTH-1:0]    rdata,
  input  wire                     rlast,
  output reg                      rready,
  // AXI-Stream output to link
  output reg                      tvalid,
  output reg [DATA_WIDTH-1:0]     tdata,
  input  wire                     tready
);

localparam IDLE       = 3'd0,
           ISSUE_AR   = 3'd1,
           WAIT_RESP  = 3'd2,
           STREAMING  = 3'd3,
           DONE       = 3'd4;

reg [2:0]               state, next_state;
reg [LEN_WIDTH-1:0]     beats_rem;
reg [7:0]               burst_len;

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state <= IDLE;
    desc_ready <= 1'b0;
    arvalid <= 1'b0;
    rready <= 1'b0;
    tvalid <= 1'b0;
    beats_rem <= 0;
  end else begin
    state <= next_state;
    // handshake updates
    if (state==IDLE && desc_valid) begin
      desc_ready <= 1'b1;
      beats_rem <= desc_len;
      araddr <= desc_addr;
    end else begin
      desc_ready <= 1'b0;
    end
    if (state==ISSUE_AR && arready) arvalid <= 1'b0;
    if (state==ISSUE_AR && !arvalid) begin
      arvalid <= 1'b1;
      // burst length = min(255,beats_rem)-1 per AXI spec
      burst_len <= (beats_rem > 8'd255) ? 8'd255 : (beats_rem[7:0]);
      arlen <= (burst_len==0) ? 8'd0 : (burst_len-1);
    end
    // streaming: forward read data to stream
    if (state==STREAMING) begin
      rready <= 1'b1;
      if (rvalid && rready && tready) begin
        tvalid <= 1'b1;
        tdata <= rdata;
      end
      if (rvalid && rready && tready && rlast) begin
        // decrement beats_rem by beats in this burst
        beats_rem <= beats_rem - (burst_len==0 ? 1 : burst_len);
        tvalid <= 1'b0;
        rready <= 1'b0;
      end
    end else begin
      rready <= 1'b0;
      tvalid <= 1'b0;
    end
  end
end

always @(*) begin
  next_state = state;
  case (state)
    IDLE: if (desc_valid) next_state = ISSUE_AR;
    ISSUE_AR: if (arvalid && arready) next_state = WAIT_RESP;
    WAIT_RESP: if (rvalid) next_state = STREAMING;
    STREAMING: if (rvalid && rlast) begin
                 if (beats_rem> (burst_len==0 ? 1 : burst_len)) next_state = ISSUE_AR;
                 else next_state = DONE;
               end
    DONE: next_state = IDLE;
  endcase
end

endmodule