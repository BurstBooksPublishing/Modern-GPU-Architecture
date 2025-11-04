module dvfs_governor (
  input  wire         clk,         // reference clock
  input  wire         rst_n,       // active-low reset
  input  wire         high_activity, // workload indicator
  input  wire         temp_warn,   // over-temp signal
  output reg  [2:0]   freq_level,  // 0..7 coarse levels
  output reg  [2:0]   volt_level   // matched voltage level
);
  // parameters for hysteresis timers
  parameter UP_THRESH = 1000; // cycles
  parameter DOWN_THRESH = 4000;
  reg [15:0] up_cnt, down_cnt;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      freq_level <= 3'd2; volt_level <= 3'd2;
      up_cnt <= 0; down_cnt <= 0;
    end else begin
      // increase attempt: require sustained high activity and no temp warning
      if (high_activity && !temp_warn) begin
        up_cnt <= up_cnt + 1; down_cnt <= 0;
        if (up_cnt >= UP_THRESH && freq_level < 3'd7) begin
          freq_level <= freq_level + 1; volt_level <= volt_level + 1;
          up_cnt <= 0;
        end
      end
      // decrease: sustained low activity or temperature warning
      else begin
        down_cnt <= down_cnt + 1; up_cnt <= 0;
        if ((down_cnt >= DOWN_THRESH || temp_warn) && freq_level > 3'd0) begin
          freq_level <= freq_level - 1; volt_level <= volt_level - 1;
          down_cnt <= 0;
        end
      end
    end
  end
endmodule