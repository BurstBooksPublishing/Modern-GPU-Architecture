module bc1_block_decoder #(
  parameter CLK_FREQ_HZ = 250_000_000
)(
  input  wire         clk,
  input  wire         rstn,
  // input handshake: 64-bit BC1 block (color0[15:0], color1[15:0], indices[31:0])
  input  wire         in_valid,
  output wire         in_ready,
  input  wire [63:0]  in_block,
  // output handshake: concatenated 16 * 24-bit RGB pixels (pix0 LSB.. pix15 MSB)
  output reg          out_valid,
  input  wire         out_ready,
  output reg  [16*24-1:0] out_pixels
);

  // Stage valid flags
  reg s0_valid, s1_valid, s2_valid;
  assign in_ready = !s0_valid; // simple single-entry intake

  // Stage registers
  reg [63:0] s0_block;
  reg [23:0] palette0, palette1, palette2, palette3;
  reg [31:0] s1_indices;

  // Stage 0: latch input
  always @(posedge clk) begin
    if (!rstn) begin
      s0_valid <= 1'b0;
      s0_block <= 64'b0;
    end else begin
      if (in_valid && in_ready) begin
        s0_block <= in_block;
        s0_valid <= 1'b1;
      end else if (s1_valid && !s0_valid) begin
        // no-op
        s0_valid <= s0_valid;
      end
      if (s1_valid && !s0_valid) ; // keep semantics simple
      if (s1_valid) s0_valid <= 1'b0; // advance when stage1 captures
    end
  end

  // Stage 1: endpoint decode and palette compute
  always @(posedge clk) begin
    if (!rstn) begin
      s1_valid <= 1'b0;
      palette0 <= 24'b0; palette1 <= 24'b0; palette2 <= 24'b0; palette3 <= 24'b0;
      s1_indices <= 32'b0;
    end else begin
      if (s0_valid && !s1_valid) begin
        // Extract endpoints
        wire [15:0] c0 = s0_block[63:48];
        wire [15:0] c1 = s0_block[47:32];
        s1_indices <= s0_block[31:0];

        // Expand 5:6:5 -> 8:8:8 by replication (fast hardware-friendly)
        wire [4:0] r5_0 = c0[15:11]; wire [5:0] g6_0 = c0[10:5]; wire [4:0] b5_0 = c0[4:0];
        wire [4:0] r5_1 = c1[15:11]; wire [5:0] g6_1 = c1[10:5]; wire [4:0] b5_1 = c1[4:0];
        wire [7:0] r8_0 = {r5_0, r5_0[4:2]};
        wire [7:0] g8_0 = {g6_0, g6_0[5:4]};
        wire [7:0] b8_0 = {b5_0, b5_0[4:2]};
        wire [7:0] r8_1 = {r5_1, r5_1[4:2]};
        wire [7:0] g8_1 = {g6_1, g6_1[5:4]};
        wire [7:0] b8_1 = {b5_1, b5_1[4:2]};

        palette0 <= {r8_0, g8_0, b8_0};
        palette1 <= {r8_1, g8_1, b8_1};

        if (c0 > c1) begin
          // four-color block: interp 2/3 and 1/3, 1/3 and 2/3
          palette2 <= { ( (2*r8_0 + r8_1) / 3 ),
                        ( (2*g8_0 + g8_1) / 3 ),
                        ( (2*b8_0 + b8_1) / 3 ) };
          palette3 <= { ( (r8_0 + 2*r8_1) / 3 ),
                        ( (g8_0 + 2*g8_1) / 3 ),
                        ( (b8_0 + 2*b8_1) / 3 ) };
        end else begin
          // three-color block: interp 1/2 and 0 (transparent treated externally)
          palette2 <= { ( (r8_0 + r8_1) >> 1 ),
                        ( (g8_0 + g8_1) >> 1 ),
                        ( (b8_0 + b8_1) >> 1 ) };
          palette3 <= 24'h000000; // convention: transparent/black
        end
        s1_valid <= 1'b1;
        s0_valid <= 1'b0;
      end else if (s2_valid && !s1_valid) begin
        // stall/advance handled by simple pipeline: allow stage2 capture
        s1_valid <= s1_valid;
      end
      if (s2_valid) s1_valid <= 1'b0; // advance to stage2 when captured
    end
  end

  // Stage 2: index unpack and pixel assembly
  integer i;
  reg [1:0] idx;
  reg [23:0] pal_sel [0:3];
  always @(posedge clk) begin
    if (!rstn) begin
      s2_valid <= 1'b0;
      out_valid <= 1'b0;
      out_pixels <= {16*24{1'b0}};
    end else begin
      if (s1_valid && !s2_valid) begin
        pal_sel[0] <= palette0; pal_sel[1] <= palette1;
        pal_sel[2] <= palette2; pal_sel[3] <= palette3;
        // assemble pixels
        for (i=0;i<16;i=i+1) begin
          idx = s1_indices[2*i +: 2]; // 2-bit selector
          out_pixels[24*i +: 24] <= pal_sel[idx];
        end
        out_valid <= 1'b1;
        s2_valid <= 1'b1;
        s1_valid <= 1'b0;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
        s2_valid <= 1'b0;
      end
    end
  end

endmodule