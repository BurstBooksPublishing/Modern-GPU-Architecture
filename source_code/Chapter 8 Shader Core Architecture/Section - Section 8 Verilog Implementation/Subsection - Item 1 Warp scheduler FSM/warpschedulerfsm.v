module warp_scheduler_fsm #(
  parameter W = 32,                       // number of warps
  parameter WIDX = (W<=1)?1: $clog2(W)    // index width (synthesis-friendly)
)(
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire [W-1:0]          ready_mask,     // scoreboard: 1 = ready to issue
  input  wire [W-1:0]          active_mask,    // warp active in SM
  input  wire                  issue_grant_ack,// pipeline accepted this cycle
  output reg                   issue_valid,    // assert when issue_warp_id valid
  output reg  [WIDX-1:0]       issue_warp_id   // granted warp id
);

  // internal state encoding
  localparam IDLE  = 2'd0, ARB = 2'd1, ISSUE = 2'd2;
  reg [1:0] state, next_state;

  reg [W-1:0] rr_ptr_onehot;           // one-hot pointer for rotation
  reg [W-1:0] candidate_mask, rot_mask, sel_mask;
  integer i;

  // rotate-left by pointer (one-hot) to implement round-robin search
  always @(*) begin
    candidate_mask = ready_mask & active_mask;
    rot_mask = candidate_mask;
    // rotate by finding index of rr_ptr_onehot (synthesizable loop)
    for (i=0; i<W; i=i+1) begin
      if (rr_ptr_onehot[i]) begin
        rot_mask = (candidate_mask << i) | (candidate_mask >> (W - i));
      end
    end
    // priority: select lowest set bit after rotation
    sel_mask = {W{1'b0}};
    for (i=0; i<W; i=i+1) begin
      if (rot_mask[i]) begin
        sel_mask[i] = 1'b1;
        break;
      end
    end
  end

  // FSM state transitions
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= IDLE;
      rr_ptr_onehot <= 1'b1;
      issue_valid <= 1'b0;
      issue_warp_id <= {WIDX{1'b0}};
    end else begin
      state <= next_state;
      case (state)
        IDLE: begin
          if (|candidate_mask) begin
            next_state <= ARB;
          end else begin
            next_state <= IDLE;
          end
        end
        ARB: begin
          next_state <= ISSUE;
          issue_valid <= 1'b1;
          // map selected bit back to original index
          for (i=0; i<W; i=i+1) begin
            if (sel_mask[i]) begin
              issue_warp_id <= (i - $clog2(rr_ptr_onehot)) % W;
            end
          end
        end
        ISSUE: begin
          if (issue_grant_ack) begin
            next_state <= UPDATE;
            issue_valid <= 1'b0;
          end else begin
            next_state <= ISSUE;
          end
        end
        UPDATE: begin
          // advance round-robin pointer
          rr_ptr_onehot <= {rr_ptr_onehot[W-2:0], rr_ptr_onehot[W-1]};
          next_state <= IDLE;
        end
        default: next_state <= IDLE;
      endcase
    end
  end
endmodule