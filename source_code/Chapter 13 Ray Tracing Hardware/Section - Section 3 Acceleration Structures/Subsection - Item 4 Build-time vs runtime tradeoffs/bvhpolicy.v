module bvh_policy (
  input  clk,               // system clock
  input  rst_n,             // active-low reset
  input  scene_change,      // flag when geometry change detected
  input  high_ray_load,     // heuristic: many rays in flight
  output reg start_rebuild, // trigger full GPU/host rebuild
  output reg start_refit,   // trigger refit/update kernel
  output reg busy           // busy while scheduling
);
  // simple counter to measure frequency of scene changes
  reg [7:0] change_count;
  parameter THRESH=8'd4;    // threshold for rebuild
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      change_count<=8'd0; busy<=1'b0;
      start_rebuild<=1'b0; start_refit<=1'b0;
    end else begin
      // clear pulses
      start_rebuild<=1'b0; start_refit<=1'b0;
      if (scene_change) change_count<=change_count+1'b1;
      // decision logic: rebuild if many changes and ray load high
      if (change_count>=THRESH && high_ray_load && !busy) begin
        start_rebuild<=1'b1; busy<=1'b1; change_count<=8'd0;
      end else if (scene_change && !busy) begin
        start_refit<=1'b1; busy<=1'b1;
      end else if (busy) begin
        // consumer clears busy when the kernel completes (not shown)
        busy<=busy; 
      end
    end
  end
endmodule