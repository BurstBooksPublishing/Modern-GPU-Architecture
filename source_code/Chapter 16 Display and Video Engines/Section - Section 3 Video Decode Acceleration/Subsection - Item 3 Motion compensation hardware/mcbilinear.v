module mc_bilinear #(
  parameter PIXEL_W = 8,
  parameter FRAC_W  = 8
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     valid_in,           // input valid
  input  wire [PIXEL_W-1:0]       p00, p01, p10, p11, // neighborhood
  input  wire [FRAC_W-1:0]        frac_x, frac_y,     // fixed-point [0,1)
  output reg                      valid_out,
  output reg  [PIXEL_W-1:0]       pixel_out
);
  // Stage 1: horizontal interpolation (unsigned math)
  localparam MUL_W = PIXEL_W+FRAC_W;
  reg [MUL_W-1:0] hor0, hor1;
  wire [MUL_W-1:0] mult0 = p00 * ( {1'b0, ( {FRAC_W{1'b1}} - frac_x) } ); // p00*(1-frac_x)
  wire [MUL_W-1:0] mult1 = p01 * frac_x; // p01*frac_x
  wire [MUL_W-1:0] mult2 = p10 * ( {1'b0, ( {FRAC_W{1'b1}} - frac_x) } );
  wire [MUL_W-1:0] mult3 = p11 * frac_x;
  // Stage registers
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      hor0 <= 0; hor1 <= 0; valid_out <= 0; pixel_out <= 0;
    end else begin
      if (valid_in) begin
        hor0 <= mult0 + mult1; // horizontal top row
        hor1 <= mult2 + mult3; // horizontal bottom row
      end
      // Stage 2: vertical combine and normalization (scale by 2^FRAC_W)
      // vertical: result = (hor0*(1-frac_y) + hor1*frac_y) >> FRAC_W
      if (valid_in) begin
        // compute and round
        reg [MUL_W+FRAC_W:0] vert;
        vert = hor0 * ( {1'b0, ( {FRAC_W{1'b1}} - frac_y) } ) +
               hor1 * frac_y;
        pixel_out <= vert >> (FRAC_W*2); // adjust shift for earlier scaling
        valid_out <= 1;
      end else begin
        valid_out <= 0;
      end
    end
  end
endmodule