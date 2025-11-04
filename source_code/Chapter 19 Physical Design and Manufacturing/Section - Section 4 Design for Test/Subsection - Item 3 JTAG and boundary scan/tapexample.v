module jtag_tap #(
  parameter IR_LEN = 4,
  parameter BOUNDARY_LEN = 128
)(
  input  wire TCK,        // test clock
  input  wire TMS,        // test mode select
  input  wire TDI,        // serial input
  input  wire TRST_n,     // optional async reset (active low)
  output reg  TDO         // serial output
);

  // TAP state encoding
  typedef enum reg [3:0] {
    RESET=4'd0, IDLE=4'd1, SELECT_DR=4'd2, CAPTURE_DR=4'd3,
    SHIFT_DR=4'd4, EXIT1_DR=4'd5, PAUSE_DR=4'd6, EXIT2_DR=4'd7,
    UPDATE_DR=4'd8, SELECT_IR=4'd9, CAPTURE_IR=4'd10, SHIFT_IR=4'd11,
    EXIT1_IR=4'd12, PAUSE_IR=4'd13, EXIT2_IR=4'd14, UPDATE_IR=4'd15
  } tap_state_t;

  reg [3:0] state, next_state;

  // Instruction and boundary registers
  reg [IR_LEN-1:0] ir;
  reg [BOUNDARY_LEN-1:0] boundary_chain;

  // State transition (sampled on rising TCK)
  always @(posedge TCK or negedge TRST_n) begin
    if (!TRST_n) state <= RESET;
    else state <= next_state;
  end

  // Combinational next-state logic
  always @(*) begin
    case (state)
      RESET:         next_state = TMS ? RESET : IDLE;
      IDLE:          next_state = TMS ? SELECT_DR : IDLE;
      SELECT_DR:     next_state = TMS ? SELECT_IR : CAPTURE_DR;
      CAPTURE_DR:    next_state = TMS ? EXIT1_DR : SHIFT_DR;
      SHIFT_DR:      next_state = TMS ? EXIT1_DR : SHIFT_DR;
      EXIT1_DR:      next_state = TMS ? UPDATE_DR : PAUSE_DR;
      PAUSE_DR:      next_state = TMS ? EXIT2_DR : PAUSE_DR;
      EXIT2_DR:      next_state = TMS ? UPDATE_DR : SHIFT_DR;
      UPDATE_DR:     next_state = TMS ? SELECT_DR : IDLE;
      SELECT_IR:     next_state = TMS ? RESET : CAPTURE_IR;
      CAPTURE_IR:    next_state = TMS ? EXIT1_IR : SHIFT_IR;
      SHIFT_IR:      next_state = TMS ? EXIT1_IR : SHIFT_IR;
      EXIT1_IR:      next_state = TMS ? UPDATE_IR : PAUSE_IR;
      PAUSE_IR:      next_state = TMS ? EXIT2_IR : PAUSE_IR;
      EXIT2_IR:      next_state = TMS ? UPDATE_IR : SHIFT_IR;
      UPDATE_IR:     next_state = TMS ? SELECT_DR : IDLE;
      default:       next_state = RESET;
    endcase
  end

  // Shift IR and DR on SHIFT states (TDI sampled at TCK posedge)
  always @(posedge TCK or negedge TRST_n) begin
    if (!TRST_n) begin
      ir <= {IR_LEN{1'b0}};
      boundary_chain <= {BOUNDARY_LEN{1'b0}};
      TDO <= 1'b0;
    end else begin
      if (state == SHIFT_IR) begin
        TDO <= ir[0];                 // TDO driven from LSB of IR
        ir <= {TDI, ir[IR_LEN-1:1]}; // shift right, new bit at MSB
      end else if (state == SHIFT_DR) begin
        TDO <= boundary_chain[0];                        // drive chain LSB
        boundary_chain <= {TDI, boundary_chain[BOUNDARY_LEN-1:1]}; // shift
      end else if (state == UPDATE_IR) begin
        // IR updated; in a full design, update control signals here
        TDO <= ir[0];
      end else begin
        TDO <= 1'bz; // float TDO when not shifting (tooling may drive pull)
      end
    end
  end

endmodule