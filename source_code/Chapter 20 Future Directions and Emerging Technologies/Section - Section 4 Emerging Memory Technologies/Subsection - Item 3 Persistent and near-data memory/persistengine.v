module persist_write_engine #(
  parameter ADDR_WIDTH = 48,
  parameter DATA_WIDTH = 64
)(
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   start,         // start persistence transaction
  input  wire [ADDR_WIDTH-1:0]  addr,
  input  wire [DATA_WIDTH-1:0]  data_in,
  output reg                    busy,
  output reg                    done,
  // memory-side interface
  output reg                    mem_req,       // request write
  output reg                    mem_persist_req,// request persistence/flush
  output reg [ADDR_WIDTH-1:0]   mem_addr,
  output reg [DATA_WIDTH-1:0]   mem_wdata,
  input  wire                   mem_ack,
  input  wire                   mem_persist_ack
);
  // FSM states
  localparam IDLE = 2'd0, WRITE = 2'd1, FLUSH = 2'd2, WAIT = 2'd3;
  reg [1:0] state, next_state;

  always @(posedge clk) begin
    if (rst) state <= IDLE; else state <= next_state;
  end

  always @(*) begin
    // default outputs
    mem_req = 1'b0; mem_persist_req = 1'b0;
    mem_addr = addr; mem_wdata = data_in;
    busy = (state != IDLE); done = (state == IDLE && start == 1'b0);
    next_state = state;
    case (state)
      IDLE: if (start) next_state = WRITE;
      WRITE: begin mem_req = 1'b1; if (mem_ack) next_state = FLUSH; end
      FLUSH: begin mem_persist_req = 1'b1; if (mem_persist_ack) next_state = WAIT; end
      WAIT: next_state = IDLE;
    endcase
  end
endmodule