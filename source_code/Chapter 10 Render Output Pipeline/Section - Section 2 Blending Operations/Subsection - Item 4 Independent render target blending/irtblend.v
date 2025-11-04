module irt_blend #(
  parameter N_RT = 4,
  parameter CH_BITS = 8,                // bits per channel
  parameter PIXEL_CH = 4,               // RGBA
  parameter PIXEL_W = CH_BITS*PIXEL_CH  // bits per pixel
)(
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    valid_in,
  output wire                    ready_in,
  input  wire [N_RT*PIXEL_W-1:0] src_pixels, // packed per-target src
  input  wire [N_RT*PIXEL_W-1:0] dst_pixels, // packed per-target dst
  input  wire [N_RT-1:0]         blend_enable,
  input  wire [2:0]              src_factor_sel [N_RT-1:0], // 0:ZERO,1:ONE,2:SRC_A,3:ONE_MINUS_SRC_A
  input  wire [2:0]              dst_factor_sel [N_RT-1:0],
  input  wire [1:0]              eq_sel        [N_RT-1:0], // 0:add,1:sub
  output reg  [N_RT*PIXEL_W-1:0] out_pixels,
  output reg                     valid_out,
  input  wire                    ready_out
);
  assign ready_in = ready_out; // pass-through backpressure

  genvar t, c;
  generate
    for (t=0; t
\section{Section 3: Color Compression}
\subsection{Item 1:  Delta color compression (DCC)}
Following the previous discussion of per-pixel blending and fast-clear optimizations, we now focus on a ROP-side, block-based color bandwidth reduction technique widely used in modern GPUs: delta color compression (DCC). DCC trades small compute and metadata storage at the render-output stage for reduced DRAM and L2 traffic, benefiting render-target-heavy workloads (G-buffer fills, HDR framebuffers, deferred shading).

Problem: high sustained write/read bandwidth from SMs and TMUs to the framebuffer and ROP cache limits frame rate and increases power. Analysis: DCC operates on small tiles or blocks (commonly 4×4 or 8×8 pixels). For a block, choose a base color and represent each pixel as a signed delta from that base. The block is compressible if all component deltas fit in a reduced bit-width; otherwise the block is stored uncompressed and a metadata bit marks that. This enables typical scenes with smooth color areas and small gradients to compress effectively.

Operational constraint and math: let color vectors be 4-channel (RGBA) 8/16/32-bit per channel depending on RT format. For a chosen base color $b$ and pixel colors $c_i$ (vector per pixel), the per-component delta is $d_{i} = c_i - b$. A block is compressible with per-component delta bit-width $b_w$ if and only if:
\begin{equation}[H]\label{eq:delta_bound}
\max_{i,\;comp\in\{R,G,B,A\}} |d_{i,comp}| \le 2^{b_w-1}-1.
\end{equation}
Selecting $b_w$ trades compression ratio and complexity: smaller $b_w$ yields higher compression but lowers the chance that all deltas fit.

Implementation sketch: the hardware pipeline stage that consumes pixel outputs from the ROP computes candidate bases (e.g., first pixel, median, or per-component min) and evaluates Eq.~\eqref{eq:delta_bound}. When compressible, metadata stores (base, packed deltas, selector bits); otherwise a bypass path writes raw pixels. Metadata lookup and eviction integrate with the render-target cache so decompress is a fast ROP-side operation on subsequent reads.

A synthesizable Verilog module below demonstrates a simplified DCC decision unit for a 4×4 block of 32-bit RGBA pixels (8 bits per channel). The encoder chooses pixel[0] as base and checks a configurable delta width parameter.

\begin{lstlisting}[language=Verilog,caption={Simple DCC block encoder (4x4, RGBA8) — combinational decision},label={lst:dcc_encoder}]
module dcc_encoder #(
  parameter PIXEL_BITS = 32,        // RGBA8
  parameter BLOCK_PIXELS = 16,
  parameter DELTA_BITS = 4          // candidate b_w
)(
  input  wire [BLOCK_PIXELS*PIXEL_BITS-1:0] pixels, // packed MSB..LSB
  output wire compressible,
  output wire [PIXEL_BITS-1:0] base_color,          // chosen base
  output wire [(BLOCK_PIXELS*(4*DELTA_BITS))-1:0] packed_deltas // per comp
);
  // unpack pixels
  wire [7:0] px[0:BLOCK_PIXELS-1][0:3];
  genvar i,j;
  generate
    for (i=0;i (2**(DELTA_BITS-1)-1) || signed_delta < -((2**(DELTA_BITS-1)))) begin
          comp_flag = 1'b0;
        end
        // store lower DELTA_BITS bits (two's complement truncation).
        deltas_r[(ii*4 + cj)*DELTA_BITS +: DELTA_BITS] = signed_delta[DELTA_BITS-1:0];
      end
    end
  end

  assign compressible = comp_flag;
  assign packed_deltas = deltas_r;
endmodule