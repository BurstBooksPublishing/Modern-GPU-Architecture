module lif_neuron #(
  parameter integer THRESH = 16'd256, // Q8.8 threshold = 1.0
  parameter integer ALPHA  = 16'd240, // Q8.8 alpha ~0.9375
  parameter integer BETA   = 16'd16   // Q8.8 beta ~0.0625
) (
  input  wire        clk,
  input  wire        rst,
  input  wire [15:0] I_q8_8,   // input current Q8.8
  output reg         spike,
  output reg  [15:0] V_q8_8    // membrane Q8.8
);
  reg [31:0] prod1, prod2;
  always @(posedge clk) begin
    if (rst) begin
      V_q8_8 <= 16'd0;
      spike  <= 1'b0;
    end else begin
      // fixed-point multiply Q8.8 * Q8.8 -> Q16.16; then shift back by 8
      prod1 = V_q8_8 * ALPHA; // Q16.16
      prod2 = I_q8_8  * BETA; // Q16.16
      V_q8_8 <= (prod1 + prod2) >> 8; // back to Q8.8
      if (V_q8_8 >= THRESH) begin
        spike  <= 1'b1;
        V_q8_8 <= 16'd0; // reset
      end else begin
        spike <= 1'b0;
      end
    end
  end
endmodule