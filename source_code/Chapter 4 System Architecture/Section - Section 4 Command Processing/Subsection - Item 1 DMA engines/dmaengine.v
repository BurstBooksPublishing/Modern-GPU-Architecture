module dma_engine #(
  parameter ADDR_WIDTH = 48,
  parameter LEN_WIDTH  = 16
)(
  input  wire                 clk,
  input  wire                 rstn,
  // descriptor ring read port
  output reg  [ADDR_WIDTH-1:0] desc_rd_addr,
  input  wire [ADDR_WIDTH-1:0] desc_rd_addr_data, // next buffer base
  input  wire [LEN_WIDTH-1:0]  desc_rd_len_data,  // length in bytes
  input  wire                 desc_rd_valid,
  output reg                  desc_rd_ready,
  // read master (source memory)
  output reg  [ADDR_WIDTH-1:0] r_addr,
  output reg  [LEN_WIDTH-1:0]  r_len,
  output reg                  r_valid,
  input  wire                 r_ready,
  input  wire [63:0]          r_data, // data stream (not used here)
  input  wire                 r_last,
  // write master (destination memory)
  output reg  [ADDR_WIDTH-1:0] w_addr,
  output reg  [LEN_WIDTH-1:0]  w_len,
  output reg                  w_valid,
  input  wire                 w_ready,
  input  wire                 start, // engine enable
  output reg                  busy
);
  localparam IDLE = 2'd0, FETCH = 2'd1, TRANSFER = 2'd2;
  reg [1:0] state, next_state;
  // descriptor latch
  reg [ADDR_WIDTH-1:0] cur_addr;
  reg [LEN_WIDTH-1:0]  cur_len;
  always @(posedge clk) begin
    if (!rstn) begin
      state <= IDLE;
      busy  <= 1'b0;
    end else begin
      state <= next_state;
      if (state==FETCH && desc_rd_valid && desc_rd_ready) begin
        cur_addr <= desc_rd_addr_data;
        cur_len  <= desc_rd_len_data;
      end
      if (state==TRANSFER) busy <= 1'b1;
      if (state==IDLE)     busy <= 1'b0;
    end
  end
  // next-state logic and outputs
  always @(*) begin
    next_state = state;
    desc_rd_ready = 1'b0;
    r_valid = 1'b0; w_valid = 1'b0;
    desc_rd_addr = 0; r_addr = 0; w_addr = 0;
    r_len = 0; w_len = 0;
    case(state)
      IDLE: if (start) next_state = FETCH;
      FETCH: begin
        desc_rd_addr = 0; desc_rd_ready = 1'b1;
        if (desc_rd_valid && desc_rd_ready) next_state = TRANSFER;
      end
      TRANSFER: begin
        // issue read and write handshakes for entire descriptor
        r_addr = cur_addr; r_len = cur_len; r_valid = 1'b1;
        w_addr = cur_addr; w_len = cur_len; w_valid = 1'b1;
        if (r_ready && w_ready) next_state = FETCH;
      end
    endcase
  end
endmodule