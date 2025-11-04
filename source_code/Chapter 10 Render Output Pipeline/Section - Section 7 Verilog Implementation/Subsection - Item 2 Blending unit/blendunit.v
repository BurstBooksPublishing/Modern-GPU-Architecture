module blend_unit (
  input  wire         clk,
  input  wire         rst_n,
  // input pixel (fragment)
  input  wire         in_valid,
  output wire         in_ready,
  input  wire [31:0]  in_color,   // [31:24]=A [23:16]=R [15:8]=G [7:0]=B
  // framebuffer read pixel
  input  wire [31:0]  fb_color,
  // config: 2-bit per factor select, 1=alpha, 2=one, 0=zero; premultiplied flag
  input  wire [1:0]   src_factor_sel,
  input  wire [1:0]   dst_factor_sel,
  input  wire         premult,
  // output
  output reg          out_valid,
  input  wire         out_ready,
  output reg  [31:0]  out_color
);
  // simple ready/valid passthrough (one-stage)
  assign in_ready = out_ready | ~out_valid;

  // extract channels
  wire [7:0] Sa = in_color[31:24];
  wire [7:0] Sr = in_color[23:16];
  wire [7:0] Sg = in_color[15:8];
  wire [7:0] Sb = in_color[7:0];
  wire [7:0] Dr = fb_color[23:16];
  wire [7:0] Dg = fb_color[15:8];
  wire [7:0] Db = fb_color[7:0];

  function [7:0] factor;
    input [1:0] sel;
    input [7:0] alpha;
    begin
      case(sel)
        2'b00: factor = 8'd0;
        2'b01: factor = alpha;         // SRC_ALPHA or DST_ALPHA externally mapped
        2'b10: factor = 8'd255;        // ONE
        default: factor = 8'd0;
      endcase
    end
  endfunction

  // pipeline registers
  reg [7:0] rs, gs, bs, aa, rd, gd, bd;
  reg [1:0] sfs, dfs;
  reg       prm;

  // compute stage
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid <= 1'b0;
    end else begin
      if (in_valid && in_ready) begin
        // latch inputs
        aa <= Sa; rs <= Sr; gs <= Sg; bs <= Sb;
        rd <= Dr; gd <= Dg; bd <= Db;
        sfs <= src_factor_sel; dfs <= dst_factor_sel;
        prm <= premult;

        // determine factors (use source alpha for both if selected)
        // here factor() uses alpha; in real config mapping, sel encodes SRC_ALPHA etc.
        // multiply-add with 16-bit intermediates and shift by 8 (divide by 256)
        reg [15:0] Fs, Fd;
        Fs = factor(sfs, Sa);
        Fd = factor(dfs, Sa);
        // compute per-channel
        reg [15:0] sum_r, sum_g, sum_b;
        if (prm) begin
          // source already premultiplied: Cs * 1 + Cd * (1 - Sa)
          sum_r = (rs) + ((rd * (16'd255 - Fs)) >> 8);
          sum_g = (gs) + ((gd * (16'd255 - Fs)) >> 8);
          sum_b = (bs) + ((db * (16'd255 - Fs)) >> 8);
        end else begin
          sum_r = ((rs * Fs) + (rd * Fd)) >> 8;
          sum_g = ((gs * Fs) + (gd * Fd)) >> 8;
          sum_b = ((bs * Fs) + (db * Fd)) >> 8;
        end
        // clamp to 255
        out_color[23:16] <= (sum_r > 16'd255) ? 8'd255 : sum_r[7:0];
        out_color[15:8]  <= (sum_g > 16'd255) ? 8'd255 : sum_g[7:0];
        out_color[7:0]   <= (sum_b > 16'd255) ? 8'd255 : sum_b[7:0];
        out_color[31:24] <= aa; // preserve alpha (common)
        out_valid <= 1'b1;
      end else if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end
    end
  end
endmodule