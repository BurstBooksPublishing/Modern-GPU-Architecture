module blend_unit(
  input  [7:0] src_r, src_g, src_b, src_a, // source RGBA
  input  [7:0] dst_r, dst_g, dst_b, dst_a, // destination RGBA
  input  [2:0] mode,                        // 0=SRC_OVER,1=ADD,2=MULT,3=MIN,4=MAX
  output [7:0] out_r, out_g, out_b, out_a  // blended RGBA
);
  // intermediate wide arithmetic to avoid overflow
  wire [15:0] sr = src_r;
  wire [15:0] sg = src_g;
  wire [15:0] sb = src_b;
  wire [15:0] sa = src_a;
  wire [15:0] dr = dst_r;
  wire [15:0] dg = dst_g;
  wire [15:0] db = dst_b;
  wire [15:0] da = dst_a;

  reg [15:0] rr, rg, rb, ra;
  always @(*) begin
    case(mode)
      3'd0: begin // SRC_OVER: (src * a + dst * (255-a))/255
        rr = (sr*sa + dr*(255-sa) + 127) / 255;
        rg = (sg*sa + dg*(255-sa) + 127) / 255;
        rb = (sb*sa + db*(255-sa) + 127) / 255;
        ra = (sa*255 + da*(255-sa) + 127) / 255; // composite alpha
      end
      3'd1: begin // ADD
        rr = sr + dr; rg = sg + dg; rb = sb + db; ra = sa + da;
      end
      3'd2: begin // MULT (per-channel multiply normalized)
        rr = (sr*dr + 127) / 255; rg = (sg*dg + 127) / 255;
        rb = (sb*db + 127) / 255; ra = (sa*da + 127) / 255;
      end
      3'd3: begin // MIN
        rr = (sr < dr) ? sr : dr; rg = (sg < dg) ? sg : dg;
        rb = (sb < db) ? sb : db; ra = (sa < da) ? sa : da;
      end
      3'd4: begin // MAX
        rr = (sr > dr) ? sr : dr; rg = (sg > dg) ? sg : dg;
        rb = (sb > db) ? sb : db; ra = (sa > da) ? sa : da;
      end
      default: begin rr=dr; rg=dg; rb=db; ra=da; end
    endcase
  end

  assign out_r = rr[7:0];
  assign out_g = rg[7:0];
  assign out_b = rb[7:0];
  assign out_a = ra[7:0];
endmodule