module pipeline_reg #(
  parameter WIDTH = 64
)(
  input  wire clk,
  input  wire rst_n,        // synchronous active-low reset
  input  wire en,           // clock-enable (synthesis-friendly)
  input  wire [WIDTH-1:0] d,
  output reg  [WIDTH-1:0] q
);
  always @(posedge clk) begin
    if (!rst_n) q <= {WIDTH{1'b0}}; // reset to zero
    else if (en) q <= d;            // enable-controlled capture
  end
endmodule