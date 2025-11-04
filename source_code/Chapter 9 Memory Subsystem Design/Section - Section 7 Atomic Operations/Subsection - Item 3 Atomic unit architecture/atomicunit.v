module atomic_unit
 #(parameter ADDR_W=40, DATA_W=64, OP_W=3)
 (
  input  wire                 clk,
  input  wire                 rstn,
  // request from SM
  input  wire                 req_valid,
  input  wire [ADDR_W-1:0]    req_addr,
  input  wire [OP_W-1:0]      req_op,    // 0=ADD,1=CAS
  input  wire [DATA_W-1:0]    req_src,
  output reg                  req_ready,
  // cache interface
  output reg                  cache_req_valid,
  output reg  [ADDR_W-1:0]    cache_req_addr,
  input  wire                 cache_req_ready,
  input  wire                 cache_resp_valid,
  input  wire [DATA_W-1:0]    cache_resp_data,
  // completion
  output reg                  resp_valid,
  output reg  [DATA_W-1:0]    resp_data
 );

 // simple FSM states
 localparam IDLE=2'd0, SEND=2'd1, WAIT=2'd2, WRITE=2'd3;
 reg [1:0] state;
 reg [ADDR_W-1:0] latched_addr;
 reg [OP_W-1:0]   latched_op;
 reg [DATA_W-1:0] latched_src;
 reg [DATA_W-1:0] result;

 // lock bit for single-line (example; production uses CAM/assoc)
 reg locked;

 always @(posedge clk) begin
  if (!rstn) begin
    state <= IDLE; req_ready <= 1'b1; cache_req_valid <= 1'b0;
    cache_req_addr <= 0; resp_valid <= 1'b0; locked <= 1'b0;
  end else begin
    resp_valid <= 1'b0;
    case (state)
      IDLE: begin
        if (req_valid && req_ready && !locked) begin
          // accept request and lock line
          latched_addr <= req_addr; latched_op <= req_op; latched_src <= req_src;
          locked <= 1'b1; req_ready <= 1'b0;
          cache_req_valid <= 1'b1; cache_req_addr <= req_addr;
          state <= SEND;
        end
      end
      SEND: begin
        if (cache_req_ready) begin
          cache_req_valid <= 1'b0; state <= WAIT;
        end
      end
      WAIT: begin
        if (cache_resp_valid) begin
          // perform RMW
          if (latched_op==1'b0) result <= cache_resp_data + latched_src; // ADD
          else result <= (cache_resp_data==latched_src) ? cache_resp_data : cache_resp_data; // CAS placeholder
          // writeback (reuse cache_req as write)
          cache_req_valid <= 1'b1; cache_req_addr <= latched_addr; state <= WRITE;
        end
      end
      WRITE: begin
        if (cache_req_ready) begin
          cache_req_valid <= 1'b0; resp_valid <= 1'b1; resp_data <= result;
          locked <= 1'b0; req_ready <= 1'b1; state <= IDLE;
        end
      end
    endcase
  end
 end

endmodule