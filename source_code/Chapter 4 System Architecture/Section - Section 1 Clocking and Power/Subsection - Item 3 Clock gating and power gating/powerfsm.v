module power_gate_ctrl #(
  parameter WAKE_DELAY = 1000  // cycles to wait for rail settle
) (
  input  wire clk,
  input  wire rst_n,
  input  wire sleep_req,      // request from scheduler
  input  wire wake_req,       // request to wake
  output reg  retention_en,   // enable retention FFs
  output reg  isolation_en,   // enable isolation cells
  output reg  power_off,      // assert to turn off power switch
  output reg  ready           // ack when domain fully off/on
);

typedef enum logic [1:0] {IDLE=2'b00, PREP=2'b01, OFF=2'b10, WAKE=2'b11} state_t;
state_t state;
reg [15:0] counter;

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state <= IDLE; counter <= 0;
    retention_en <= 1'b0; isolation_en <= 1'b0; power_off <= 1'b0; ready <= 1'b1;
  end else begin
    case (state)
      IDLE: begin
        ready <= 1'b1;
        if (sleep_req) begin
          // prepare: enable retention and isolate I/Os
          retention_en <= 1'b1;
          isolation_en <= 1'b1;
          ready <= 1'b0;
          state <= PREP;
          counter <= 0;
        end
      end
      PREP: begin
        // allow a few cycles for state capture and safe clock stop
        if (counter < 4) counter <= counter + 1;
        else begin
          power_off <= 1'b1; // assert power switch
          state <= OFF;
          counter <= 0;
        end
      end
      OFF: begin
        // wait for rails to discharge/confirm off (policy-dependent)
        if (power_off && counter < WAKE_DELAY) counter <= counter + 1;
        else if (power_off) begin
          ready <= 1'b1; // domain acknowledged off
          if (wake_req) begin
            power_off <= 1'b0; // start wake
            ready <= 1'b0;
            state <= WAKE;
            counter <= 0;
          end
        end
      end
      WAKE: begin
        // wait for regulator settle, then release isolation and retention
        if (counter < WAKE_DELAY) counter <= counter + 1;
        else begin
          isolation_en <= 1'b0;
          retention_en <= 1'b0;
          ready <= 1'b1;
          state <= IDLE;
        end
      end
    endcase
  end
end

endmodule