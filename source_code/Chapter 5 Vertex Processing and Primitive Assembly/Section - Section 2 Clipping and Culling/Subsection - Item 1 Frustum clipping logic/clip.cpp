struct Vertex { float x,y,z,w; std::vector attrs; }; // attrs: arbitrary layout
// Clip triangle vertices against six planes (a,b,c,d). Returns clipped polygon vertices.
std::vector clipTriangleToFrustum(
    const std::array& tri,
    const std::array,6>& planes)
{
  std::vector poly(tri.begin(), tri.end());
  for (auto &pl : planes) {
    std::vector out;
    if (poly.empty()) break;
    for (size_t i=0;i= 0.0f; bool inB = sB >= 0.0f;
      if (inA && inB) { out.push_back(B); }
      else if (inA && !inB) { // A inside, B outside -> emit intersection
        float t = sA / (sA - sB);
        Vertex I; I.x = A.x + t*(B.x-A.x); I.y = A.y + t*(B.y-A.y);
        I.z = A.z + t*(B.z-A.z); I.w = A.w + t*(B.w-A.w);
        I.attrs.resize(A.attrs.size());
        for (size_t k=0;k intersection then B
        float t = sA / (sA - sB);
        Vertex I; I.x = A.x + t*(B.x-A.x); I.y = A.y + t*(B.y-A.y);
        I.z = A.z + t*(B.z-A.z); I.w = A.w + t*(B.w-A.w);
        I.attrs.resize(A.attrs.size());
        for (size_t k=0;k
\subsection{Item 2:  Backface culling}
The frustum clipping logic produces normalized device coordinates and typically hands down screen-space positions to the culling stage; backface culling then uses those post-projection positions to reject triangles whose facing does not contribute to the final image, reducing work for later raster and fragment units.

Backface culling problem → analysis → implementation → implications:

\begin{itemize}
\item Problem: determine whether a triangle faces away from the camera to avoid rasterization and fragment shading.
\item Analysis: compute the 2D signed area (z-component of the cross product) of the triangle in screen (or NDC) space. Using screen-space coordinates avoids issues with perspective sign changes; viewport scale does not change the sign of the area.
\item Operational relevance: culling reduces SM/TMU/ROP load for opaque geometry, improving throughput for GPGPU and raster workloads. It should occur as early as possible, ideally in the primitive assembly unit before triangle setup.
\end{itemize}

Mathematical test (post-perspective-divide, using 2D screen coordinates $(x_i,y_i)$):
\begin{equation}[H]\label{eq:signed_area}
A_z = (x_1-x_0)(y_2-y_0) - (y_1-y_0)(x_2-x_0).
\end{equation}
If $A_z>0$ the triangle has one winding (say CCW); if $A_z<0$ it has the opposite winding (CW). The GPU compares the sign to the configured cull mode:
\begin{itemize}
  \item Front-face defined as CCW (common default). Cull mode options: none, cull front, cull back.
  \item Degenerate triangles where $A_z=0$ should be discarded to avoid divide-by-zero issues downstream.
\end{itemize}

Implementation notes for hardware:
\begin{itemize}
\item Use fixed-point representation (e.g.\ 16.16) for $x,y$ after viewport transform; this provides deterministic, synthesizable arithmetic and avoids per-vertex division.
\item Compute differences in signed arithmetic, multiply in double-width, then subtract to produce a signed 2×precision result.
\item Perform culling combinationally in the primitive assembly pipeline stage to avoid added latency; register the result if pipeline timing requires.
\end{itemize}

Verilog synthesizable module implementing the signed-area test and a configurable cull mode:
\begin{lstlisting}[language=Verilog,caption={Backface culling combinational unit (fixed-point 16.16)},label={lst:bfcull}]
module backface_cull
  #(parameter IW=32, // input width (signed, e.g., 16.16)
    parameter DW=64) // product width
  (
    input  signed [IW-1:0] x0, y0, x1, y1, x2, y2, // screen-space coords
    input  wire           cull_enable,            // 0: off, 1: on
    input  wire [1:0]     cull_mode,              // 00:none,01:cull_back,10:cull_front
    output wire           cull_out                // 1 = discard triangle
  );
  // differences
  wire signed [IW-1:0] dx10 = x1 - x0;
  wire signed [IW-1:0] dy10 = y1 - y0;
  wire signed [IW-1:0] dx20 = x2 - x0;
  wire signed [IW-1:0] dy20 = y2 - y0;
  // cross product z = dx10*dy20 - dy10*dx20
  wire signed [DW-1:0] prod1 = $signed(dx10) * $signed(dy20); // signed mult
  wire signed [DW-1:0] prod2 = $signed(dy10) * $signed(dx20);
  wire signed [DW-1:0] cross_z = prod1 - prod2;
  // decide sign: >0 => CCW, <0 => CW, ==0 => degenerate
  wire ccw = cross_z > 0;
  wire cw  = cross_z < 0;
  wire degen = cross_z == 0;
  // mode encoding: 00 none, 01 cull_back (cull when not front), 10 cull_front
  assign cull_out = cull_enable && (
                     (cull_mode==2'b01 && cw) || // cull back-facing (CW)
                     (cull_mode==2'b10 && ccw) || // cull front-facing (CCW)
                     degen); // always drop degenerate
endmodule