module interp_inc #(
  parameter WIDTH = 32,       // total bits
  parameter FRAC  = 16        // fractional bits (Q(WIDTH-FRAC-1).FRAC)
) (
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  ena,        // advance one pixel when high
  input  wire                  load,       // load initial value
  input  wire signed [WIDTH-1:0] init,     // initial Q value
  input  wire signed [WIDTH-1:0] grad,     // per-pixel gradient Q value
  output reg  signed [WIDTH-1:0] out      // current interpolated Q value
);
  reg signed [WIDTH-1:0] acc;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= {WIDTH{1'b0}};
      out <= {WIDTH{1'b0}};
    end else begin
      if (load) acc <= init;                // load starting predivided A'
      else if (ena) acc <= acc + grad;     // incremental update (wrap checked by synthesis)
      out <= acc;                           // truncated output (consumer must handle fractional bits)
    end
  end
endmodule