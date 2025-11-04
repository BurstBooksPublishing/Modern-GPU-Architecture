module sm_power_manager #(
  parameter WID = 6,              // width of active_warps
  parameter IDLE_THRESH = 256,    // cycles to consider idle
  parameter LOW_THRESH  = 2,      // warps -> low power
  parameter MED_THRESH  = 8       // warps -> medium power
)(
  input  wire               clk,
  input  wire               rst_n,
  input  wire [WID-1:0]     active_warps, // number of ready warps
  output reg                clk_gate,     // 1 = clock enabled
  output reg  [1:0]         dvfs_level    // 00=off,01=low,10=med,11=high
);
  reg [15:0] idle_ctr;
  wire idle = (active_warps == 0);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      idle_ctr <= 0;
      clk_gate <= 1;
      dvfs_level <= 2'b11; // default high
    end else begin
      idle_ctr <= idle ? idle_ctr + 1 : 0;
      // clock gating: if idle for long, gate clock
      if (idle_ctr >= IDLE_THRESH) clk_gate <= 0;
      else clk_gate <= 1;
      // DVFS selection based on workload
      if (active_warps <= LOW_THRESH) dvfs_level <= 2'b01;
      else if (active_warps <= MED_THRESH) dvfs_level <= 2'b10;
      else dvfs_level <= 2'b11;
    end
  end
endmodule