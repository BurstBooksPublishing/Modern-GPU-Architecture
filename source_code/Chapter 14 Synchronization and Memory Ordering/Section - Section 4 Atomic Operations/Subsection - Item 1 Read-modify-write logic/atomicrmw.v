module atomic_rmw #(
  parameter ADDR_WIDTH = 64,
  parameter DATA_WIDTH = 32
)(
  input  wire                      clk,
  input  wire                      reset,
  // request interface (from SM warp scheduler)
  input  wire                      req_valid,
  input  wire [ADDR_WIDTH-1:0]     req_addr,
  input  wire [DATA_WIDTH-1:0]     req_operand,
  input  wire [2:0]                req_op,    // 0=NOP,1=ADD,2=AND,3=OR,4=MIN,5=MAX
  output reg                       req_ready,
  // memory/cache port (simple exclusive handshake)
  output reg                       mem_read,
  output reg                       mem_write,
  output reg [ADDR_WIDTH-1:0]      mem_addr,
  input  wire [DATA_WIDTH-1:0]     mem_rdata,
  output reg [DATA_WIDTH-1:0]      mem_wdata,
  input  wire                      mem_ready,
  // response to caller
  output reg                       resp_valid,
  output reg [DATA_WIDTH-1:0]      resp_old
);
  typedef enum reg [2:0] {IDLE, READ, COMPUTE, WRITEBACK, RESP} state_t;
  state_t state, next_state;
  reg [DATA_WIDTH-1:0] oldv, newv;
  // FSM
  always @(posedge clk) begin
    if (reset) state <= IDLE;
    else state <= next_state;
  end
  always @(*) begin
    // default outputs
    req_ready = 1'b0; mem_read = 0; mem_write = 0; mem_addr = req_addr;
    mem_wdata = newv; resp_valid = 0; resp_old = oldv; next_state = state;
    case (state)
      IDLE: begin
        req_ready = 1; if (req_valid) next_state = READ;
      end
      READ: begin
        mem_read = 1;
        if (mem_ready) next_state = COMPUTE;
      end
      COMPUTE: begin
        oldv = mem_rdata;
        // compute newv combinationally
        case (req_op)
          3'd1: newv = oldv + req_operand; // ADD
          3'd2: newv = oldv & req_operand; // AND
          3'd3: newv = oldv | req_operand; // OR
          3'd4: newv = (oldv < req_operand) ? oldv : req_operand; // MIN unsigned
          3'd5: newv = (oldv > req_operand) ? oldv : req_operand; // MAX unsigned
          default: newv = req_operand;
        endcase
        next_state = WRITEBACK;
      end
      WRITEBACK: begin
        mem_write = 1;
        if (mem_ready) next_state = RESP;
      end
      RESP: begin
        resp_valid = 1; resp_old = oldv; next_state = IDLE;
      end
    endcase
  end
endmodule