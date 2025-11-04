module bvh_traversal_fsm #(
  parameter ADDR_W = 32,
  parameter STACK_D = 16
)(
  input  wire             clk,
  input  wire             rst_n,
  input  wire             start,           // start traversal
  input  wire [ADDR_W-1:0] root_addr,
  // BVH node memory interface
  output reg  [ADDR_W-1:0] node_rd_addr,
  output reg               node_rd_en,
  input  wire [127:0]      node_rd_data,    // node payload (child addrs, leaf flag)
  input  wire              node_rd_valid,
  // ray-box unit interface
  output reg               box_req,
  output reg  [ADDR_W-1:0] box_node_addr,
  input  wire              box_ready,
  input  wire              box_hit,
  input  wire [31:0]       box_t0,
  input  wire [31:0]       box_t1,
  // tri unit interface
  output reg               tri_req,
  output reg  [31:0]       tri_tri_id,
  input  wire              tri_ready,
  input  wire              tri_hit,
  input  wire [31:0]       tri_t,
  // hit report
  output reg               hit_valid,
  output reg  [31:0]       hit_t,
  output reg  [31:0]       hit_tri_id
);

localparam IDLE=0, FETCH_NODE=1, TEST_BOX=2, PUSH_CHILDREN=3,
           DESCEND=4, LEAF_HANDLE=5, TRI_DISPATCH=6, BACKTRACK=7, REPORT=8;

reg [3:0] state, next_state;

// stack memory
reg [ADDR_W-1:0] stack_mem [0:STACK_D-1];
reg [4:0] stack_ptr;

// traversal registers
reg [ADDR_W-1:0] cur_node;
reg [31:0] t_best;

// node fields (simple unpack)
wire leaf = node_rd_data[0];
wire [ADDR_W-1:0] left_child = node_rd_data[ADDR_W +: ADDR_W];
wire [ADDR_W-1:0] right_child = node_rd_data[2*ADDR_W +: ADDR_W];
wire [31:0] tri_id = node_rd_data[96 +: 32];

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state <= IDLE; stack_ptr <= 0; hit_valid <= 0; t_best <= 32'h7f7f_ffff;
  end else begin
    state <= next_state;
    if (state==IDLE && start) begin
      stack_mem[0] <= root_addr; stack_ptr <= 1;
    end
    if (tri_hit) begin // update best hit
      if (tri_t < t_best) begin t_best <= tri_t; hit_valid <= 1; hit_t <= tri_t; hit_tri_id <= tri_tri_id; end
    end
  end
end

always @(*) begin
  // default signals
  node_rd_en = 0; node_rd_addr = 0; box_req = 0; box_node_addr = 0;
  tri_req = 0; tri_tri_id = 0; next_state = state;
  case (state)
    IDLE: if (start) next_state = FETCH_NODE;
    FETCH_NODE: begin
      if (stack_ptr==0) next_state = REPORT;
      else begin
        node_rd_en = 1; node_rd_addr = stack_mem[stack_ptr-1]; next_state = TEST_BOX;
      end
    end
    TEST_BOX: begin
      box_req = 1; box_node_addr = node_rd_addr;
      if (box_hit && node_rd_valid) begin
        if (leaf) next_state = LEAF_HANDLE; else next_state = PUSH_CHILDREN;
      end else if (!box_hit) next_state = BACKTRACK;
    end
    PUSH_CHILDREN: begin
      // order children by t0; push far child first (handled when box_ready && node_rd_valid)
      if (node_rd_valid) begin
        // push left/right based on box_t0 comparisons (external box unit provides both children intervals).
        // For brevity, push both then descend.
        stack_mem[stack_ptr] = right_child; stack_mem[stack_ptr+1] = left_child;
        next_state = DESCEND;
      end
    end
    DESCEND: begin stack_ptr = stack_ptr + 2; next_state = FETCH_NODE; end
    LEAF_HANDLE: begin
      // dispatch triangle test
      tri_req = 1; tri_tri_id = tri_id;
      if (tri_ready) next_state = BACKTRACK;
    end
    BACKTRACK: begin
      if (stack_ptr==0) next_state = REPORT; else begin stack_ptr = stack_ptr - 1; next_state = FETCH_NODE; end
    end
    REPORT: next_state = IDLE;
  endcase
end

endmodule