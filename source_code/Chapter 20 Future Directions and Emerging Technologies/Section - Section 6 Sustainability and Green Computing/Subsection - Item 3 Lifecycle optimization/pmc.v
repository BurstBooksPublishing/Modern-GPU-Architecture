module power_mode_controller #(
  parameter IDLE_THRESH = 1000, // cycles before idle
  parameter WAKE_LATENCY = 50   // cycles to ramp up
) (
  input  wire        clk, rst_n,
  input  wire [7:0]  util,      // utilization percent (0-100)
  input  wire [15:0] temp,      // temperature sensor
  output reg  [2:0]  dvfs_level,// 0=lowest..7=highest
  output reg         pg_enable  // 1 = power on, 0 = gated
);
  // simple FSM states
  localparam S_OFF = 2'd0, S_SLEEP = 2'd1, S_ACTIVE = 2'd2;
  reg [1:0] state, next_state;
  reg [15:0] idle_ctr, wake_ctr;

  // state transition combinational
  always @(*) begin
    next_state = state;
    case (state)
      S_ACTIVE: if (util < 8) next_state = S_SLEEP;
      S_SLEEP:  if (util >= 16) next_state = S_ACTIVE;
                else if (idle_ctr > IDLE_THRESH) next_state = S_OFF;
      S_OFF:    if (util >= 8) next_state = S_SLEEP;
      default: next_state = S_OFF;
    endcase
  end

  // sequential: counters, outputs
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_OFF; idle_ctr <= 0; wake_ctr <= 0;
      dvfs_level <= 3'd0; pg_enable <= 1'b0;
    end else begin
      state <= next_state;
      // idle/wake counters
      if (state == S_ACTIVE && util < 8) idle_ctr <= idle_ctr + 1;
      else if (state != S_ACTIVE) idle_ctr <= idle_ctr + 1;
      else idle_ctr <= 0;

      if (state == S_OFF && util >= 8) wake_ctr <= wake_ctr + 1;
      else wake_ctr <= 0;

      // outputs: simple DVFS mapping and pg control
      case (next_state)
        S_ACTIVE: begin pg_enable <= 1'b1; dvfs_level <= (util > 80)?7:
                                                  (util > 50)?5:
                                                  (util > 20)?3:1; end
        S_SLEEP:  begin pg_enable <= 1'b1; dvfs_level <= 3'd0; end
        S_OFF:    begin pg_enable <= 1'b0; dvfs_level <= 3'd0; end
      endcase
    end
  end
endmodule