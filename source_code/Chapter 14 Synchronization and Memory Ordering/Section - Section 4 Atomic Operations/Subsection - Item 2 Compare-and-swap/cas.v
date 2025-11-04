module cas_unit (
  input  wire         clk,
  input  wire         rst_n,
  input  wire         req_valid,       // request valid
  input  wire [31:0]  req_addr,        // byte address
  input  wire [31:0]  req_expected,    // expected value
  input  wire [31:0]  req_desired,     // desired value
  output reg          req_ready,       // accept next request
  output reg          rsp_valid,
  output reg  [31:0]  rsp_old,         // returned old value
  output reg          rsp_success      // 1 if swap performed
);
  // simple memory model: synchronous single-ported RAM (externalized in real design)
  reg [31:0] mem [0:1023]; // example depth; replace with real memory interface

  typedef enum reg [1:0] {IDLE=0, READ=1, COMPARE=2, WRITE=3} state_t;
  state_t state, next;

  reg [31:0] saved_addr;
  reg [31:0] saved_expected, saved_desired;
  reg [31:0] read_value;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE; req_ready <= 1'b1; rsp_valid <= 1'b0;
    end else begin
      state <= next;
      if (state==IDLE && req_valid && req_ready) begin
        saved_addr <= req_addr[11:2]; // word index
        saved_expected <= req_expected;
        saved_desired <= req_desired;
      end
      if (state==READ) read_value <= mem[saved_addr];
      if (state==WRITE) mem[saved_addr] <= saved_desired;
      if (state==COMPARE) begin
        rsp_old <= read_value;
        if (read_value==saved_expected) rsp_success <= 1'b1; else rsp_success <= 1'b0;
        rsp_valid <= 1'b1;
      end
      if (state==IDLE) rsp_valid <= 1'b0;
      // flow control
      req_ready <= (state==IDLE);
    end
  end

  always @(*) begin
    next = state;
    case (state)
      IDLE:   if (req_valid && req_ready) next = READ;
      READ:   next = COMPARE;
      COMPARE: if (rsp_success) next = WRITE; else next = IDLE;
      WRITE:  next = IDLE;
    endcase
  end
endmodule