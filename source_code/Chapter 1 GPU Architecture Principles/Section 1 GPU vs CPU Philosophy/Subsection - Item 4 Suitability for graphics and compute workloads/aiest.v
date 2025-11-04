module ai_estimator (
  input  wire        clk,        // clock
  input  wire        rst_n,      // active-low reset
  input  wire [31:0] flop_inc,   // reported FLOPs this cycle
  input  wire [31:0] byte_inc,   // reported bytes transferred this cycle
  input  wire        valid,      // valid sample
  output reg  [31:0] intensity_q // Q16.16 arithmetic intensity
);
  reg [47:0] flop_acc;
  reg [47:0] byte_acc;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      flop_acc <= 48'd0;
      byte_acc <= 48'd0;
      intensity_q <= 32'd0;
    end else if (valid) begin
      flop_acc <= flop_acc + flop_inc;
      byte_acc <= byte_acc + byte_inc;
      // compute intensity = (flops << 16) / bytes, saturate on zero
      if (byte_acc + byte_inc == 48'd0)
        intensity_q <= 32'hFFFF_FFFF; // max
      else
        intensity_q <= ((flop_acc + flop_inc) << 16) / (byte_acc + byte_inc);
    end
  end
endmodule