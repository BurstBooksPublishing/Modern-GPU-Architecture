module guard_band_reject
#(parameter W = 32,               // total width (signed)
  parameter FRACT = 16,           // fractional bits (Q)
  parameter GB_Q = (1< product in Q with extra FRACT bits
  wire signed [W+FRACT:0] GQ = ONE_Q + GB_Q; // safe small width
  assign b0x = w0_ext * GQ; assign b0y = b0x; // reuse for x/y checks
  assign b1x = w1_ext * GQ; assign b1y = b1x;
  assign b2x = w2_ext * GQ; assign b2y = b2x;

  // extend x/y to product width for fair compares
  wire signed [PW-1:0] x0_ext = {{W{x0[W-1]}}, x0};
  wire signed [PW-1:0] y0_ext = {{W{y0[W-1]}}, y0};
  wire signed [PW-1:0] x1_ext = {{W{x1[W-1]}}, x1};
  wire signed [PW-1:0] y1_ext = {{W{y1[W-1]}}, y1};
  wire signed [PW-1:0] x2_ext = {{W{x2[W-1]}}, x2};
  wire signed [PW-1:0] y2_ext = {{W{y2[W-1]}}, y2};

  // per-vertex tests: left/right/top/bottom
  wire v0_left  = (x0_ext < -b0x);
  wire v0_right = (x0_ext >  b0x);
  wire v0_bottom= (y0_ext < -b0y);
  wire v0_top   = (y0_ext >  b0y);

  wire v1_left  = (x1_ext < -b1x);
  wire v1_right = (x1_ext >  b1x);
  wire v1_bottom= (y1_ext < -b1y);
  wire v1_top   = (y1_ext >  b1y);

  wire v2_left  = (x2_ext < -b2x);
  wire v2_right = (x2_ext >  b2x);
  wire v2_bottom= (y2_ext < -b2y);
  wire v2_top   = (y2_ext >  b2y);

  // trivial reject if all verts are on same outside side
  assign reject = (v0_left  & v1_left  & v2_left)  |
                  (v0_right & v1_right & v2_right) |
                  (v0_bottom& v1_bottom& v2_bottom)|
                  (v0_top   & v1_top   & v2_top);
endmodule