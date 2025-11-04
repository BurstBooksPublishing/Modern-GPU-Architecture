module trunc_adder #(
  parameter N = 32,                // total width
  parameter K = 8                  // preserved LSB width
)(
  input  wire [N-1:0] a,
  input  wire [N-1:0] b,
  output wire [N-1:0] sum
);
  // lower K-bit exact addition (produces carry_out but ignored)
  wire [K:0] low_sum = a[K-1:0] + b[K-1:0];          // K+1 bits
  // upper bits computed without carry_in to avoid long chain
  wire [N-K-1:0] high_sum = a[N-1:K] + b[N-1:K];    // no carry_in
  assign sum = { high_sum, low_sum[K-1:0] };        // concatenate
endmodule