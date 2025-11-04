module rr_arbiter #(parameter N = 8) (
  input  wire             clk,
  input  wire             rst,
  input  wire [N-1:0]     req,    // one bit per requester
  output reg  [N-1:0]     grant   // one-hot grant
);
  reg [$clog2(N)-1:0] ptr;        // rotating priority pointer
  integer i;
  wire [N-1:0] masked;
  // rotate requests so ptr is MSB-aligned
  assign masked = (req << ptr) | (req >> (N - ptr));
  always @(*) begin
    grant = {N{1'b0}};
    for (i = 0; i < N; i = i + 1)
      if (masked[i]) begin
        grant = ( ({{N{1'b1}} >> (N-1-i)) & ~({{N{1'b1}} >> (N-i))) ) >> ptr;
        disable for; // break
      end
  end
  always @(posedge clk) begin
    if (rst) ptr <= 0;
    else if (|req) begin
      // advance pointer to next position after current grant
      ptr <= (ptr + (grant ? $clog2(N) : 1)) % N;
    end
  end
endmodule