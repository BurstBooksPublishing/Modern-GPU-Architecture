module sm_selfopt_ctrl #(
  parameter EWMA_SHIFT = 4,      // decay factor 1/16
  parameter TARGET = 16'd1000,   // target perf units per epoch
  parameter HYST_UP = 2'd3,      // hysteresis up threshold
  parameter HYST_DOWN = 2'd1     // hysteresis down threshold
)(
  input  wire         clk,
  input  wire         rst_n,
  input  wire [15:0]  perf_cnt,   // sampled counter (e.g., useful_inst)
  input  wire [15:0]  power_budget,
  output reg  [1:0]   dvfs_req,   // 0..3 frequency levels
  output reg  [7:0]   warp_cap    // max active warps
);
  reg [15:0] ewma;                // fixed-point EWMA
  reg [1:0]  up_vote, down_vote;  // simple hysteresis votes

  // EWMA update: ewma <= (15/16)*ewma + (1/16)*perf_cnt
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) ewma <= 16'd0;
    else ewma <= ewma - (ewma >> EWMA_SHIFT) + (perf_cnt >> EWMA_SHIFT);
  end

  // Simple policy: compare ewma to TARGET, adjust dvfs with hysteresis.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dvfs_req <= 2'd0;
      up_vote <= 2'd0;
      down_vote <= 2'd0;
      warp_cap <= 8'd32;
    end else begin
      // voting to avoid transient changes
      if (ewma > TARGET + HYST_UP) up_vote <= up_vote + 1'b1; else up_vote <= 2'd0;
      if (ewma < TARGET - HYST_DOWN) down_vote <= down_vote + 1'b1; else down_vote <= 2'd0;

      // adjust DVFS
      if (up_vote == 2'd3 && dvfs_req != 2'd3) dvfs_req <= dvfs_req + 1'b1;
      else if (down_vote == 2'd3 && dvfs_req != 2'd0) dvfs_req <= dvfs_req - 1'b1;

      // warp cap tied to dvfs and power budget (simple affine mapping)
      warp_cap <= (power_budget < 16'd200) ? 8'd16 : (dvfs_req==2'd3 ? 8'd48 : 8'd32);
    end
  end
endmodule