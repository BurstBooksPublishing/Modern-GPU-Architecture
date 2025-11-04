module bc1_decompressor(
  input  wire        clk,
  input  wire        rst_n,
  input  wire        start,            // start decode of new 64-bit block
  input  wire [63:0] block_in,         // compressed BC1 block
  output reg  [23:0] pixel_out,        // RGB8 output
  output reg         pixel_valid
);
  // internal registers
  reg [15:0] c0, c1;
  reg [31:0] indices;
  reg [7:0] pal_r [0:3], pal_g [0:3], pal_b [0:3];
  reg [3:0] idx; // 0..15
  reg busy;

  // unpack on start
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; pixel_valid <= 1'b0; idx <= 4'd0;
    end else begin
      if (start && !busy) begin
        busy <= 1'b1;
        c0 <= block_in[15:0];
        c1 <= block_in[31:16];
        indices <= block_in[63:32];
        idx <= 4'd0;
        pixel_valid <= 1'b0;
        // compute palette now (combinational below)
      end else if (busy) begin
        // produce one pixel per cycle
        integer sh; reg [1:0] sel;
        sh = idx * 2;
        sel = (indices >> sh) & 2'b11;
        pixel_out <= {pal_r[sel], pal_g[sel], pal_b[sel]};
        pixel_valid <= 1'b1;
        if (idx == 4'd15) begin busy <= 1'b0; end
        idx <= idx + 1;
      end else begin
        pixel_valid <= 1'b0;
      end
    end
  end

  // palette computation (BC1 rules)
  always @(*) begin
    // expand RGB565 to 8-bit
    reg [4:0] r5_0; reg [5:0] g6_0; reg [4:0] b5_0;
    reg [4:0] r5_1; reg [5:0] g6_1; reg [4:0] b5_1;
    r5_0 = c0[15:11]; g6_0 = c0[10:5]; b5_0 = c0[4:0];
    r5_1 = c1[15:11]; g6_1 = c1[10:5]; b5_1 = c1[4:0];
    pal_r[0] = {r5_0, r5_0[4:2]}; // R5->R8
    pal_g[0] = {g6_0, g6_0[5:4]}; // G6->G8
    pal_b[0] = {b5_0, b5_0[4:2]};
    pal_r[1] = {r5_1, r5_1[4:2]};
    pal_g[1] = {g6_1, g6_1[5:4]};
    pal_b[1] = {b5_1, b5_1[4:2]};
    if (c0 > c1) begin
      // 4-color block
      pal_r[2] = (2*pal_r[0] + pal_r[1]) / 3;
      pal_g[2] = (2*pal_g[0] + pal_g[1]) / 3;
      pal_b[2] = (2*pal_b[0] + pal_b[1]) / 3;
      pal_r[3] = (pal_r[0] + 2*pal_r[1]) / 3;
      pal_g[3] = (pal_g[0] + 2*pal_g[1]) / 3;
      pal_b[3] = (pal_b[0] + 2*pal_b[1]) / 3;
    end else begin
      // 3-color + transparent (map transparent to black here)
      pal_r[2] = (pal_r[0] + pal_r[1]) >> 1;
      pal_g[2] = (pal_g[0] + pal_g[1]) >> 1;
      pal_b[2] = (pal_b[0] + pal_b[1]) >> 1;
      pal_r[3] = 8'd0; pal_g[3] = 8'd0; pal_b[3] = 8'd0;
    end
  end
endmodule