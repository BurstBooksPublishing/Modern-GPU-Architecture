module tex_addr_gen #(
  parameter W_BITS = 12,            // bits to index width/height (max 4096)
  parameter COORD_W = 32            // Q16.16 fixed-point
)(
  input  wire [COORD_W-1:0] u_in,   // Q16.16 normalized U
  input  wire [COORD_W-1:0] v_in,   // Q16.16 normalized V
  input  wire [W_BITS-1:0] width,   // texture width in texels
  input  wire [W_BITS-1:0] height,  // texture height in texels
  input  wire [1:0] mode_u,         // 00=CLAMP_EDGE,01=REPEAT,10=MIRROR,11=CLAMP_BORDER
  input  wire [1:0] mode_v,
  output reg  [W_BITS-1:0] tex_x,
  output reg  [W_BITS-1:0] tex_y,
  output reg  border_flag           // high when sample should use border color
);
  // local values
  wire signed [COORD_W-1:0] u = u_in;
  wire signed [COORD_W-1:0] v = v_in;
  // scale normalized coordinates: scaled = floor(u * size)
  wire [COORD_W+W_BITS-1:0] scaled_u = u * {{(W_BITS){1'b0}}, width}; // full precision
  wire [COORD_W+W_BITS-1:0] scaled_v = v * {{(W_BITS){1'b0}}, height};

  // integer part extraction (floor for non-negative; handle negatives below)
  wire signed [COORD_W+W_BITS-1:0] s_u = scaled_u >>> 16;
  wire signed [COORD_W+W_BITS-1:0] s_v = scaled_v >>> 16;

  // helper function via tasks omitted; implement inline for clarity
  always @(*) begin
    border_flag = 1'b0;
    // U axis
    if (mode_u == 2'b01) begin // REPEAT
      // modulo using Verilog '%' (synthesizable)
      tex_x = (s_u >= 0) ? (s_u % width) : (((-s_u) % width) ? (width - ((-s_u) % width)) % width : 0);
    end else if (mode_u == 2'b10) begin // MIRROR
      // map to period 2W, then reflect
      integer p; p = (s_u >= 0) ? (s_u % (width*2)) : ((-s_u) % (width*2));
      if (p >= width) tex_x = (width*2 - p - 1);
      else tex_x = p;
    end else if (mode_u == 2'b11) begin // CLAMP_BORDER
      if (s_u < 0 || s_u >= width) begin border_flag = 1'b1; tex_x = 0; end
      else tex_x = s_u;
    end else begin // CLAMP_EDGE
      if (s_u < 0) tex_x = 0;
      else if (s_u >= width) tex_x = width - 1;
      else tex_x = s_u;
    end

    // V axis (same logic)
    if (mode_v == 2'b01) begin
      tex_y = (s_v >= 0) ? (s_v % height) : (((-s_v) % height) ? (height - ((-s_v) % height)) % height : 0);
    end else if (mode_v == 2'b10) begin
      integer p2; p2 = (s_v >= 0) ? (s_v % (height*2)) : ((-s_v) % (height*2));
      if (p2 >= height) tex_y = (height*2 - p2 - 1);
      else tex_y = p2;
    end else if (mode_v == 2'b11) begin
      if (s_v < 0 || s_v >= height) begin border_flag = 1'b1; tex_y = 0; end
      else tex_y = s_v;
    end else begin
      if (s_v < 0) tex_y = 0;
      else if (s_v >= height) tex_y = height - 1;
      else tex_y = s_v;
    end
  end
endmodule