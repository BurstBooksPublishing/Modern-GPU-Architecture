/* Fixed-point Q8 (8 fractional bits) */
typedef int32_t fx; const int FP=8;
fx to_fx(float v){ return (fx)roundf(v*(1<
\subsection{Item 2:  Pixel coverage determination}
The previous subsection derived per-triangle edge equations and barycentric weights at arbitrary sample positions; pixel coverage determination uses those evaluated edge functions to decide which pixel samples the raster pipeline sends to the fragment stage. This subsection turns those edge values into a compact coverage mask used by MSAA, ROPs, and early-Z logic.

Problem: given the edge function value for each triangle edge at a set of subpixel sample positions, produce a per-pixel sample coverage mask that obeys hardware fill rules (notably the top-left rule) so that adjacent triangles share pixels without double-coverage. Analysis begins with the edge equation for a directed edge:
\begin{equation}[H]\label{eq:edge}
E(x,y)=Ax+By+C,
\end{equation}
where the sign of $E$ measures which side of the infinite edge contains the triangle interior. For $S$ fixed sample positions $(x_s,y_s)$ per pixel, the coverage mask is the bitfield
\begin{equation}
M=\sum_{s=0}^{S-1} 2^{s}\,\mathbb{1}\Big(\bigwedge_{i=0}^{2}\;T_i\big(E_i(x_s,y_s)\big)\Big),
\end{equation}
where $T_i$ is the comparison predicate for edge $i$ (either $>0$ or $\ge 0$ depending on top-left tie-breaking) and $\mathbb{1}$ is the indicator function. Implementing $T_i$ correctly requires knowing the edge orientation (which defines whether it is a top or left edge) so shared-edge pixels are attributed deterministically.

Implementation: evaluate each edge function at each sample (usually done in fixed-point to avoid FP hardware in the inner raster loop). The per-sample cover is the AND of the three edge predicates; combining four 2$\times$2 MSAA samples yields a 4-bit mask. The following synthesizable Verilog module converts three signed fixed-point edge evaluations for four sample points into a 4-bit coverage mask and accepts per-edge top-left flags to implement the tie-break rule.

\begin{lstlisting}[language=Verilog,caption=Coverage mask generator for 4-sample MSAA,label={lst:coverage_mask}]
module coverage_mask_4s
#(parameter WIDTH = 24) // signed Q16.8 fixed-point
(
  input  signed [WIDTH-1:0] e0_s0, e0_s1, e0_s2, e0_s3, // edge0 @ samples
  input  signed [WIDTH-1:0] e1_s0, e1_s1, e1_s2, e1_s3, // edge1 @ samples
  input  signed [WIDTH-1:0] e2_s0, e2_s1, e2_s2, e2_s3, // edge2 @ samples
  input  [2:0] top_left_flags, // bit per edge: 1==include zero (top/left)
  output reg [3:0] mask
);
  // compare helper: returns 1 if value passes edge test
  function automatic bit pass(input signed [WIDTH-1:0] v, input bit include_zero);
    begin
      pass = (v > 0) || ((v == 0) && include_zero);
    end
  endfunction

  always @(*) begin
    bit s0 = pass(e0_s0, top_left_flags[0]) & pass(e1_s0, top_left_flags[1]) & pass(e2_s0, top_left_flags[2]);
    bit s1 = pass(e0_s1, top_left_flags[0]) & pass(e1_s1, top_left_flags[1]) & pass(e2_s1, top_left_flags[2]);
    bit s2 = pass(e0_s2, top_left_flags[0]) & pass(e1_s2, top_left_flags[1]) & pass(e2_s2, top_left_flags[2]);
    bit s3 = pass(e0_s3, top_left_flags[0]) & pass(e1_s3, top_left_flags[1]) & pass(e2_s3, top_left_flags[2]);
    mask = {s3, s2, s1, s0}; // bit 0 = sample 0
  end
endmodule