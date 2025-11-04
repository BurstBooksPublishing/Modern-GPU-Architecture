module thermal_throttle #(
  parameter N_REGIONS = 8,
  parameter TEMP_WIDTH = 8,        // sensor bits (degC)
  parameter MASK_WIDTH = 32        // SM mask width
)(
  input  wire                    clk,
  input  wire                    rstn,
  input  wire [N_REGIONS*TEMP_WIDTH-1:0] temps_flat, // concatenated temps
  input  wire                    pmic_ack,           // power cap applied
  output reg  [2:0]              dvfs_level,         // 0..7 coarse levels
  output reg  [MASK_WIDTH-1:0]   sm_disable_mask,    // 1 disables SM
  output reg                     critical_fault
);
  integer i;
  reg [TEMP_WIDTH-1:0] temps [0:N_REGIONS-1];
  reg [TEMP_WIDTH-1:0] max_temp;

  // unpack temps
  always @(*) begin
    for (i=0;i max_temp) max_temp = temps[i];
  end

  // simple FSM: normal -> throttle -> critical
  localparam NORMAL = 2'b00, THROTTLE = 2'b01, CRITICAL = 2'b10;
  reg [1:0] state, next_state;

  always @(posedge clk or negedge rstn) begin
    if (!rstn) state <= NORMAL;
    else state <= next_state;
  end

  always @(*) begin
    next_state = state;
    dvfs_level = 3'd0;
    sm_disable_mask = {MASK_WIDTH{1'b0}};
    critical_fault = 1'b0;
    case (state)
      NORMAL: begin
        if (max_temp >= 90) next_state = THROTTLE; // degC threshold
      end
      THROTTLE: begin
        dvfs_level = 3'd4; // reduced frequency
        // disable half of SMs as simple policy
        sm_disable_mask = { {MASK_WIDTH/2{1'b1}}, {MASK_WIDTH/2{1'b0}} };
        if (max_temp >= 105) next_state = CRITICAL;
        else if (max_temp <= 80 && pmic_ack) next_state = NORMAL;
      end
      CRITICAL: begin
        dvfs_level = 3'd7; // maximum safe downshift (interpretation dependent)
        sm_disable_mask = {MASK_WIDTH{1'b1}}; // disable all SMs
        critical_fault = 1'b1;
        if (max_temp <= 70) next_state = NORMAL;
      end
    endcase
  end
endmodule