module sm_clk_en_ctrl #(
  parameter IDLE_THRESH = 1024  // cycles before disabling
)(
  input  wire clk,
  input  wire rstn,
  input  wire activity,         // pulse when SM issues work
  output reg  clk_en            // clock-enable to SM registers
);
  reg [15:0] idle_cnt;
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      idle_cnt <= 0;
      clk_en   <= 1'b0;
    end else begin
      if (activity) begin
        idle_cnt <= 0;
        clk_en   <= 1'b1;        // re-enable on activity
      end else if (idle_cnt >= IDLE_THRESH) begin
        clk_en   <= 1'b0;        // disable after threshold
      end else begin
        idle_cnt <= idle_cnt + 1;
      end
    end
  end
endmodule