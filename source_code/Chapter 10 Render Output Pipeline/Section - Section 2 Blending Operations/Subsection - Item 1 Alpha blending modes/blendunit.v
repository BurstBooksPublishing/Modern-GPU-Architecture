module blend_unit #(
  parameter W = 8  // bits per channel
)(
  input  wire [4*W-1:0] src,    // {R,G,B,A}
  input  wire [4*W-1:0] dst,    // {R,G,B,A}
  input  wire [3:0] mode,       // blend mode selector
  output reg  [3*W-1:0] outrgb  // {R,G,B} result
);
  // split channels
  wire [W-1:0] src_r = src[4*W-1:3*W];
  wire [W-1:0] src_g = src[3*W-1:2*W];
  wire [W-1:0] src_b = src[2*W-1:1*W];
  wire [W-1:0] src_a = src[1*W-1:0*W];
  wire [W-1:0] dst_r = dst[4*W-1:3*W];
  wire [W-1:0] dst_g = dst[3*W-1:2*W];
  wire [W-1:0] dst_b = dst[2*W-1:1*W];

  // intermediate widths
  localparam IW = 2*W;
  wire [IW-1:0] sr = src_r, sg = src_g, sb = src_b, sa = src_a;
  wire [IW-1:0] dr = dst_r, dg = dst_g, db = dst_b;

  reg [IW-1:0] out_r, out_g, out_b;
  always @(*) begin
    case (mode)
      4'd0: begin // Replace
        out_r = sr; out_g = sg; out_b = sb;
      end
      4'd1: begin // Alpha (SrcOver), straight alpha
        out_r = (sr*sa + dr*((1<<W)-sa)) >> W;
        out_g = (sg*sa + dg*((1<<W)-sa)) >> W;
        out_b = (sb*sa + db*((1<<W)-sa)) >> W;
      end
      4'd2: begin // Premultiplied SrcOver
        out_r = (sr + dr*((1<<W)-sa)) >> W;
        out_g = (sg + dg*((1<<W)-sa)) >> W;
        out_b = (sb + db*((1<<W)-sa)) >> W;
      end
      4'd3: begin // Additive
        out_r = sr + dr; out_g = sg + dg; out_b = sb + db;
      end
      4'd4: begin // Subtractive
        out_r = (dr > sr) ? dr - sr : 0;
        out_g = (dg > sg) ? dg - sg : 0;
        out_b = (db > sb) ? db - sb : 0;
      end
      4'd5: begin // Min
        out_r = (sr < dr) ? sr : dr;
        out_g = (sg < dg) ? sg : dg;
        out_b = (sb < db) ? sb : db;
      end
      4'd6: begin // Max
        out_r = (sr > dr) ? sr : dr;
        out_g = (sg > dg) ? sg : dg;
        out_b = (sb > db) ? sb : db;
      end
      4'd7: begin // Logical AND
        out_r = sr & dr; out_g = sg & dg; out_b = sb & db;
      end
      4'd8: begin // Logical OR
        out_r = sr | dr; out_g = sg | dg; out_b = sb | db;
      end
      default: begin // XOR / fallback
        out_r = sr ^ dr; out_g = sg ^ dg; out_b = sb ^ db;
      end
    endcase
    // clamp to W bits
    out_r = out_r > ((1<<W)-1) ? ((1<<W)-1) : out_r;
    out_g = out_g > ((1<<W)-1) ? ((1<<W)-1) : out_g;
    out_b = out_b > ((1<<W)-1) ? ((1<<W)-1) : out_b;
  end

  assign outrgb = {out_r[W-1:0], out_g[W-1:0], out_b[W-1:0]};
endmodule