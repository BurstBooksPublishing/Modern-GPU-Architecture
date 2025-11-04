module clipper #(
  parameter COORD_W = 32,
  parameter FRAC = 16
)(
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   in_valid,
  input  wire signed [COORD_W-1:0] in_v0_x, in_v0_y, in_v0_z, in_v0_w,
  input  wire signed [COORD_W-1:0] in_v1_x, in_v1_y, in_v1_z, in_v1_w,
  input  wire signed [COORD_W-1:0] in_v2_x, in_v2_y, in_v2_z, in_v2_w,
  output reg                    out_valid,
  output reg  [3:0]             out_count,
  output reg signed [COORD_W-1:0] out_x [0:6],
  output reg signed [COORD_W-1:0] out_y [0:6],
  output reg signed [COORD_W-1:0] out_z [0:6],
  output reg signed [COORD_W-1:0] out_w [0:6]
);
  // Plane coefficients for left,right,bottom,top,near,far as (a,b,c,d)
  // left: x + w >= 0 -> (1,0,0,1)
  localparam signed [15:0] A0 = 16'sd1, B0 = 16'sd0, C0 = 16'sd0, D0 = 16'sd1;
  // right: w - x >= 0 -> (-1,0,0,1)
  localparam signed [15:0] A1 = -16'sd1, B1 = 16'sd0, C1 = 16'sd0, D1 = 16'sd1;
  // bottom: y + w >= 0
  localparam signed [15:0] A2 = 16'sd0, B2 = 16'sd1, C2 = 16'sd0, D2 = 16'sd1;
  // top: w - y >= 0
  localparam signed [15:0] A3 = 16'sd0, B3 = -16'sd1, C3 = 16'sd0, D3 = 16'sd1;
  // near: z + w >= 0
  localparam signed [15:0] A4 = 16'sd0, B4 = 16'sd0, C4 = 16'sd1, D4 = 16'sd1;
  // far: w - z >= 0
  localparam signed [15:0] A5 = 16'sd0, B5 = 16'sd0, C5 = -16'sd1, D5 = 16'sd1;

  // Internal polygon storage max 7 verts
  reg signed [COORD_W-1:0] poly_x [0:6];
  reg signed [COORD_W-1:0] poly_y [0:6];
  reg signed [COORD_W-1:0] poly_z [0:6];
  reg signed [COORD_W-1:0] poly_w [0:6];
  reg [3:0] poly_n;

  integer i, j;
  // Compute plane value helper (fixed-point dot). Uses 64-bit temp.
  function signed [47:0] plane_val;
    input signed [15:0] a,b,c,d;
    input signed [COORD_W-1:0] vx,vy,vz,vw;
    begin
      plane_val = a * vx + b * vy + c * vz + d * vw; // Q16.16*int -> wider
    end
  endfunction

  // Linear interpolation: res = v0 + t*(v1-v0), t is Q16.16 fraction
  function signed [COORD_W-1:0] lerp;
    input signed [COORD_W-1:0] v0;
    input signed [COORD_W-1:0] v1;
    input signed [COORD_W-1:0] t; // Q16.16
    reg signed [63:0] diff;
    reg signed [63:0] prod;
    begin
      diff = v1 - v0;
      prod = diff * t; // Q16.16*Q16.16 -> Q32.32
      lerp = v0 + (prod >>> FRAC); // back to Q16.16
    end
  endfunction

  // Sequential clipping pipeline triggered on in_valid (combinational loops avoided)
  always @(posedge clk) begin
    if (rst) begin
      out_valid <= 0; out_count <= 0; poly_n <= 0;
    end else if (in_valid) begin
      // init polygon with triangle
      poly_x[0] <= in_v0_x; poly_y[0] <= in_v0_y; poly_z[0] <= in_v0_z; poly_w[0] <= in_v0_w;
      poly_x[1] <= in_v1_x; poly_y[1] <= in_v1_y; poly_z[1] <= in_v1_z; poly_w[1] <= in_v1_w;
      poly_x[2] <= in_v2_x; poly_y[2] <= in_v2_y; poly_z[2] <= in_v2_z; poly_w[2] <= in_v2_w;
      poly_n <= 3;
      // iterate planes sequentially
      for (i=0;i<6;i=i+1) begin
        // select plane coeffs
        reg signed [15:0] pa,pb,pc,pd;
        case(i)
          0: begin pa=A0; pb=B0; pc=C0; pd=D0; end
          1: begin pa=A1; pb=B1; pc=C1; pd=D1; end
          2: begin pa=A2; pb=B2; pc=C2; pd=D2; end
          3: begin pa=A3; pb=B3; pc=C3; pd=D3; end
          4: begin pa=A4; pb=B4; pc=C4; pd=D4; end
          default: begin pa=A5; pb=B5; pc=C5; pd=D5; end
        endcase
        // temp output poly
        reg signed [COORD_W-1:0] tmp_x [0:6];
        reg signed [COORD_W-1:0] tmp_y [0:6];
        reg signed [COORD_W-1:0] tmp_z [0:6];
        reg signed [COORD_W-1:0] tmp_w [0:6];
        reg [3:0] tmp_n;
        tmp_n = 0;
        for (j=0;j= 0) begin
            tmp_x[tmp_n] = poly_x[j]; tmp_y[tmp_n] = poly_y[j];
            tmp_z[tmp_n] = poly_z[j]; tmp_w[tmp_n] = poly_w[j]; tmp_n = tmp_n + 1;
          end
          if ((p0 >= 0 && p1 < 0) || (p0 < 0 && p1 >= 0)) begin
            // compute t = p0/(p0-p1). Convert 48-bit->32-bit Q16.16 approx.
            signed [63:0] num = p0 <<< (FRAC-16); // align to Q16.16 (p scaled by input Q16.16)
            signed [63:0] den = (p0 - p1
\subsection{Item 3: Triangle setup unit}
The clipping module supplies post-clipped, screen-space vertices and the matrix multiplier provides correctly projected coordinates; the triangle setup consumes these to form edge equations and per-attribute gradients with subpixel fixed-point precision. Below we state the numerical problem, derive the core formulas, describe a pipelined Verilog implemention, and summarize implications for SM/ROP integration.

Problem and analysis: the unit must produce, for each triangle, three edge equations and interpolation gradients for attributes (e.g., $1/w$, texture UVs). Edge equations follow the standard 2D line form; gradients use the triangle area as the denominator for barycentric interpolation. For vertices $v_0=(x_0,y_0)$, $v_1=(x_1,y_1)$, $v_2=(x_2,y_2)$, an edge function is
\begin{equation}[H]\label{eq:edge}
e_i(x,y)=A_i x + B_i y + C_i,\quad A_i = y_{j}-y_{k},\; B_i = x_{k}-x_{j},\; C_i = x_{j}y_{k}-x_{k}y_{j},
\end{equation}
where $(i,j,k)$ cycles over $(0,1,2)$. For an attribute $a$ with vertex values $a_0,a_1,a_2$ the screen-space gradients are
\begin{equation}[H]\label{eq:grad}
\frac{\partial a}{\partial x} = \frac{(a_1-a_0)(y_2-y_0)-(a_2-a_0)(y_1-y_0)}{D},\quad
\frac{\partial a}{\partial y} = \frac{(a_2-a_0)(x_1-x_0)-(a_1-a_0)(x_2-x_0)}{D},
\end{equation}
with $D=(x_1-x_0)(y_2-y_0)-(x_2-x_0)(y_1-y_0)$ (twice signed area).

Implementation notes: we choose a Q16.8 fixed-point representation for screen coordinates and attributes to retain subpixel precision while bounding bitwidths. The Verilog below is a synthesizable, single-stage combinational setup with registered inputs and outputs and a valid/ready handshake. It computes signed edge coefficients, the signed area $D$, and attribute gradients using an integer divider (tool-provided divider IP is assumed for synthesis). The module is parameterizable for coordinate and attribute widths.

\begin{lstlisting}[language=Verilog,caption={Triangle setup unit (synthesizable)},label={lst:tri_setup}]
module triangle_setup #(
  parameter IW=24, // integer+frac for coords (Q16.8)
  parameter AW=32  // attribute width
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 in_valid,
  output wire                 in_ready,
  input  wire signed [IW-1:0] x0, input wire signed [IW-1:0] y0,
  input  wire signed [IW-1:0] x1, input wire signed [IW-1:0] y1,
  input  wire signed [IW-1:0] x2, input wire signed [IW-1:0] y2,
  input  wire signed [AW-1:0] a0, input wire signed [AW-1:0] a1, input wire signed [AW-1:0] a2,
  output reg                  out_valid,
  input  wire                 out_ready,
  output reg signed [IW+8:0]  A0, output reg signed [IW+8:0] B0, output reg signed [IW+8:0] C0,
  output reg signed [IW+8:0]  A1, output reg signed [IW+8:0] B1, output reg signed [IW+8:0] C1,
  output reg signed [IW+8:0]  A2, output reg signed [IW+8:0] B2, output reg signed [IW+8:0] C2,
  output reg signed [AW+IW:0] grad_ax, output reg signed [AW+IW:0] grad_ay
);
  // ready/valid simple flow
  assign in_ready = ~out_valid | out_ready;
  // registers stage inputs
  reg signed [IW-1:0] rx0,ry0,rx1,ry1,rx2,ry2;
  reg signed [AW-1:0] ra0,ra1,ra2;
  always @(posedge clk) begin
    if (rst) begin rx0<=0; ry0<=0; rx1<=0; ry1<=0; rx2<=0; ry2<=0; ra0<=0; ra1<=0; ra2<=0; out_valid<=0; end
    else begin
      if (in_valid & in_ready) begin
        rx0<=x0; ry0<=y0; rx1<=x1; ry1<=y1; rx2<=x2; ry2<=y2; ra0<=a0; ra1<=a1; ra2<=a2;
      end
      // compute edges combinationally then register outputs
      // Edge0: between v1 and v2
      A0 <= ry1 - ry2;
      B0 <= rx2 - rx1;
      C0 <= rx1*ry2 - rx2*ry1;
      A1 <= ry2 - ry0;
      B1 <= rx0 - rx2;
      C1 <= rx2*ry0 - rx0*ry2;
      A2 <= ry0 - ry1;
      B2 <= rx1 - rx0;
      C2 <= rx0*ry1 - rx1*ry0;
      // denom D
      signed [2*IW-1:0] D;
      D = (rx1 - rx0)*(ry2 - ry0) - (rx2 - rx0)*(ry1 - ry0);
      // attribute deltas
      signed [AW:0] da1 = ra1 - ra0;
      signed [AW:0] da2 = ra2 - ra0;
      // numerators for grads (extended)
      signed [AW+IW:0] nx = da1*(ry2-ry0) - da2*(ry1-ry0);
      signed [AW+IW:0] ny = da2*(rx1-rx0) - da1*(rx2-rx0);
      // divide (synthesis uses IP); handle zero-area by clamping to zero
      if (D == 0) begin grad_ax <= 0; grad_ay <= 0; end
      else begin grad_ax <= nx / D; grad_ay <= ny / D; end
      out_valid <= in_valid & in_ready;
      if (out_ready & out_valid) out_valid <= 0;
    end
  end
endmodule