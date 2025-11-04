module ray_box_slab(
  input  wire        clk,
  input  wire        rst,
  input  wire        valid_in,
  input  wire signed [31:0] ox, oy, oz,        // Q16.16 origin
  input  wire signed [31:0] invdx, invdy, invdz, // Q16.16 reciprocal dir
  input  wire signed [31:0] bxmin, bymin, bzmin,
  input  wire signed [31:0] bxmax, bymax, bzmax,
  output reg         hit,
  output reg signed [31:0] t_near, t_far,
  output reg         ready
);
  // Multiply Q16.16 * Q16.16 -> Q32.32, then shift >>16 to get Q16.16
  wire signed [63:0] t1x = ($signed(bxmin) - $signed(ox)) * $signed(invdx);
  wire signed [63:0] t2x = ($signed(bxmax) - $signed(ox)) * $signed(invdx);
  wire signed [63:0] t1y = ($signed(bymin) - $signed(oy)) * $signed(invdy);
  wire signed [63:0] t2y = ($signed(bymax) - $signed(oy)) * $signed(invdy);
  wire signed [63:0] t1z = ($signed(bzmin) - $signed(oz)) * $signed(invdz);
  wire signed [63:0] t2z = ($signed(bzmax) - $signed(oz)) * $signed(invdz);

  // reduce to Q16.16
  wire signed [31:0] tx1 = t1x[47:16], tx2 = t2x[47:16];
  wire signed [31:0] ty1 = t1y[47:16], ty2 = t2y[47:16];
  wire signed [31:0] tz1 = t1z[47:16], tz2 = t2z[47:16];

  wire signed [31:0] tminx = (tx1 < tx2) ? tx1 : tx2;
  wire signed [31:0] tmaxx = (tx1 > tx2) ? tx1 : tx2;
  wire signed [31:0] tminy = (ty1 < ty2) ? ty1 : ty2;
  wire signed [31:0] tmaxy = (ty1 > ty2) ? ty1 : ty2;
  wire signed [31:0] tminz = (tz1 < tz2) ? tz1 : tz2;
  wire signed [31:0] tmaxz = (tz1 > tz2) ? tz1 : tz2;

  wire signed [31:0] enter = (tminx > tminy) ? ((tminx > tminz) ? tminx : tminz) : ((tminy > tminz) ? tminy : tminz);
  wire signed [31:0] exitv = (tmaxx < tmaxy) ? ((tmaxx < tmaxz) ? tmaxx : tmaxz) : ((tmaxy < tmaxz) ? tmaxy : tmaxz);

  always @(posedge clk or posedge rst) begin
    if (rst) begin
      hit <= 1'b0; ready <= 1'b0; t_near <= 32'b0; t_far <= 32'b0;
    end else begin
      ready <= valid_in;
      if (valid_in) begin
        hit <= (enter <= exitv) && (exitv >= 32'b0); // exit >= 0
        t_near <= enter;
        t_far  <= exitv;
      end
    end
  end
endmodule