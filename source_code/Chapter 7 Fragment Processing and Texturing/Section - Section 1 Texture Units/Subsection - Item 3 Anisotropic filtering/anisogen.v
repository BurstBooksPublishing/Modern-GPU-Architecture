module anisotropic_generator
 #(parameter MAX_ANISO=8, Q=8) // Q = fractional bits (Q8.8)
 (
  input  wire signed [15:0] ddx_u, ddx_v, ddy_u, ddy_v, // Q8.8 signed
  input  wire [2:0] max_aniso,                           // user cap (1..8)
  output reg  [3:0] sample_count,                        // 0..8
  output reg  [(MAX_ANISO*16)-1:0] offsets_u_packed,     // MAX_ANISO x Q-format
  output reg  [(MAX_ANISO*16)-1:0] offsets_v_packed      // packed
 );
 // abs
 function [15:0] abs16; input signed [15:0] x; begin abs16 = x[15]? -x : x; end endfunction
 wire [15:0] a0 = abs16(ddx_u), a1 = abs16(ddx_v), a2 = abs16(ddy_u), a3 = abs16(ddy_v);
 // max/min
 reg [15:0] vmax, vmin;
 integer i;
 always @(*) begin
   vmax = a0; if (a1>vmax) vmax = a1; if (a2>vmax) vmax = a2; if (a3>vmax) vmax = a3;
   vmin = a0; if (a1 max_aniso ? max_aniso : ratio_int[3:0]);
   // pick major axis: compare contributions projected to u/v
   // approximate major axis vector as sign of component with vmax
   reg signed [15:0] maj_u, maj_v;
   if (vmax==a0) begin maj_u = ddx_u; maj_v = ddx_v; end
   else if (vmax==a1) begin maj_u = ddx_u; maj_v = ddx_v; end
   else if (vmax==a2) begin maj_u = ddy_u; maj_v = ddy_v; end
   else begin maj_u = ddy_u; maj_v = ddy_v; end
   // normalize axis to unit step approximated by shifting: axis = sign * (|maj|>>Q)
   reg signed [15:0] axis_u, axis_v;
   axis_u = (maj_u[15]? -maj_u: maj_u); axis_v = (maj_v[15]? -maj_v: maj_v);
   // produce equally spaced offsets in range [-0.5,0.5] * axis (scaled)
   for (i=0;i> Q (Q-format multiply)
     offsets_u_packed[(i+1)*16-1 -: 16] = (t_q * axis_u) >>> Q;
     offsets_v_packed[(i+1)*16-1 -: 16] = (t_q * axis_v) >>> Q;
   end
 end
 endmodule