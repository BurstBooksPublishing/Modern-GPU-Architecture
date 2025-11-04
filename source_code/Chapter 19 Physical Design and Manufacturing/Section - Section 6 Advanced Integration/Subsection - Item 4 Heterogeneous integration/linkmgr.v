module link_manager #(
  parameter LANES = 8,        // physical lanes
  parameter ID_WIDTH = 4
)(
  input  wire clk, reset_n,
  input  wire [LANES-1:0] lane_rx_valid, // per-lane status
  output reg  [LANES-1:0] lane_tx_enable, // enable lanes
  input  wire start_training,
  output reg  trained, // training done
  output reg  [ID_WIDTH-1:0] status_id
);
  // simple FSM: IDLE -> TRAIN -> ACTIVE
  localparam IDLE=2'd0, TRAIN=2'd1, ACTIVE=2'd2;
  reg [1:0] state, next_state;
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) state <= IDLE; else state <= next_state;
  end
  always @(*) begin
    next_state = state;
    lane_tx_enable = {LANES{1'b0}};
    trained = 1'b0;
    case(state)
      IDLE: if (start_training) next_state = TRAIN;
      TRAIN: begin
        // enable lanes if receiver sees valid signal
        lane_tx_enable = lane_rx_valid;
        if (&lane_rx_valid) next_state = ACTIVE; // all lanes good
      end
      ACTIVE: begin
        lane_tx_enable = {LANES{1'b1}};
        trained = 1'b1;
      end
    endcase
  end
  // simple status identifier
  always @(posedge clk) if (!reset_n) status_id <= 0; else status_id <= status_id + trained;
endmodule