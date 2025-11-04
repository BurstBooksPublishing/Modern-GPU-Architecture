module scan_dff (
  input  wire clk,      // functional clock
  input  wire rst_n,    // active-low synchronous reset
  input  wire se,       // scan enable (1 = shift)
  input  wire si,       // scan in
  input  wire d,        // functional data
  output reg  q,        // flop output
  output wire so        // scan out (next chain element)
);
  // synchronous logic: on scan enable load si, otherwise load d
  always @(posedge clk) begin
    if (!rst_n)
      q <= 1'b0;
    else if (se)
      q <= si;        // shift path
    else
      q <= d;         // functional path
  end
  assign so = q;       // scan chain output
endmodule