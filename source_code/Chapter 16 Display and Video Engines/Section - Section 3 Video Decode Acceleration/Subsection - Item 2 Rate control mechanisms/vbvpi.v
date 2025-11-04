module vbv_qp_ctrl #(
  parameter QP_MIN = 0,
  parameter QP_MAX = 51,
  parameter BW = 32,
  parameter Kp = 16,     // proportional gain (fixed scale)
  parameter Ki = 2,      // integral gain (fixed scale)
  parameter SCALE = 16
) (
  input  wire               clk,
  input  wire               rst,
  input  wire               interval_tick,           // once per update interval
  input  wire [BW-1:0]      bits_generated,          // measured bits this interval
  input  wire [BW-1:0]      target_bits_per_interval,// target bits for interval
  output reg  [7:0]         qp_out                  // current QP
);
  // signed registers for error accumulation
  reg signed [BW+15:0] acc_err;
  reg signed [BW+7:0]   err; 
  reg [BW-1:0]          buf; // optional VBV fullness (simple) - not modeled fully

  always @(posedge clk) begin
    if (rst) begin
      qp_out <= 8'd26; acc_err <= 0; err <= 0; buf <= 0;
    end else if (interval_tick) begin
      // compute error = target - generated
      err <= $signed({1'b0,target_bits_per_interval}) - $signed({1'b0,bits_generated});
      acc_err <= acc_err + err;
      // compute delta_qp = (Kp*err + Ki*acc_err) / SCALE
      // use signed widths to avoid overflow; clamp result into safe range
      reg signed [BW+31:0] term_p = Kp * err;
      reg signed [BW+31:0] term_i = Ki * acc_err;
      reg signed [15:0] delta_qp = (term_p + term_i) / SCALE;
      // update and clamp QP
      reg signed [15:0] qp_signed = $signed({8'd0, qp_out}) - delta_qp; // subtract because positive err -> target>generated -> lower bits so decrease QP? adjust sign as policy
      if (qp_signed < QP_MIN) qp_out <= QP_MIN;
      else if (qp_signed > QP_MAX) qp_out <= QP_MAX;
      else qp_out <= qp_signed[7:0];
      // update simple VBV fullness estimator
      if ($unsigned(bits_generated) >= $unsigned(target_bits_per_interval)) buf <= buf + (bits_generated - target_bits_per_interval);
      else buf <= buf - (target_bits_per_interval - bits_generated);
    end
  end
endmodule