module degenerate_detector #(
  parameter signed [31:0] THRESH = 32'h0000_0010  // small Q16.16 epsilon
) (
  input  wire signed [31:0] x0,y0,x1,y1,x2,y2,   // Q16.16 screen coords
  output reg  signed [63:0] area2,               // 64-bit signed area*2
  output reg               is_degenerate,       // exact zero
  output reg               is_near_degenerate,  // |area2| < THRESH
  output reg               backface            // true if negative winding
);
  // intermediate deltas
  wire signed [31:0] dx1 = x1 - x0;
  wire signed [31:0] dy1 = y1 - y0;
  wire signed [31:0] dx2 = x2 - x0;
  wire signed [31:0] dy2 = y2 - y0;

  // 64-bit cross product
  wire signed [63:0] cp1 = $signed(dx1) * $signed(dy2);
  wire signed [63:0] cp2 = $signed(dy1) * $signed(dx2);

  always @* begin
    area2 = cp1 - cp2;                         // eq. (1)
    is_degenerate = (area2 == 64'sd0);
    is_near_degenerate = (area2 > -$signed({{32{THRESH[31]}},THRESH}) &&
                          area2 <  $signed({{32{THRESH[31]}},THRESH}));
    backface = (area2 < 64'sd0);               // consistent with backface culling
  end
endmodule