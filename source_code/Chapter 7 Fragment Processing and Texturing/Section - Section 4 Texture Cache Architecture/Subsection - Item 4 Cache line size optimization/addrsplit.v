module tex_addr_split #(
  parameter ADDR_W = 48,         // physical/virtual address width
  parameter LINE_BYTES = 64      // line size in bytes (power of two)
)(
  input  wire [ADDR_W-1:0] addr, // incoming texture address
  output wire [IDX_W-1:0] index, // cache index bits
  output wire [TAG_W-1:0]  tag,  // tag bits
  output wire [OFF_W-1:0]  offset,// offset within line
  output wire [15:0]       burst  // burst count to fetch whole line
);
  // synthesizable clog2
  function integer clog2(input integer x); integer i; begin i=0; while((1<
\subsection{Item 5:  Texture cache coherency}
The cache line sizing and replacement choices discussed previously shape both the frequency and cost of coherency events; smaller lines and aggressive eviction raise invalidation traffic, while replacement policy choices determine which lines become vulnerable to cross-writer races. Here we focus on the coherence problem that arises when texture data read by the texture mapping unit (TMU) can be modified by other agents (compute SMs, ROPs, CPU DMA), and on a minimal, implementable hardware mechanism to maintain correctness with acceptable latency and bandwidth cost.

Problem and analysis.
\begin{itemize}
\item In typical modern GPUs the TMU uses a read-optimized L1 texture cache to serve bilinear/trilinear and anisotropic filtering work for many threads (SIMT warps). Texture caches are often treated as effectively read-only for graphics texture accesses, but coherent behaviour is required whenever writes occur to the same address range (streaming updates, CUDA graphics interop, or shader writes to texture-backed buffers).
\item Strong coherence (snooping every write) costs memory bandwidth and adds latency to both TMU hits and write completion. Relaxed strategies rely on explicit synchronization (fences/barriers) but must still provide fast invalidation on asynchronous writes to avoid stale-sample hazards.
\item A common compromise is write-notify + invalidate: the writer posts a compact notification (address) to the cache-coherency controller; the controller locates any matching cache lines and clears their valid bits, forcing subsequent TMU accesses to fetch fresh data from L2/DRAM.
\end{itemize}

Mapping and cost model.
\begin{itemize}
\item For a direct-mapped texture cache with $N$ lines and line size $L$, the set index is
\begin{equation}[H]\label{eq:set_index}
\text{index} = \left\lfloor\frac{\text{addr}}{L}\right\rfloor \bmod N,
\end{equation}
and the probability a write causes an invalidation equals the fraction of writes that address currently-valid cached lines. Invalidation bandwidth scales with write rate times the false-positive fraction when using set-based notifications.
\end{itemize}

Implementation (synthesizable Verilog).
\begin{itemize}
\item The following module implements a parameterizable direct-mapped invalidator: on a write notification it computes the index, compares the stored tag, and clears the valid bit synchronously. This minimal logic is low-area and low-latency and suits cases where writers are less frequent than TMU reads.
\end{itemize}

\begin{lstlisting}[language=Verilog,caption={Direct-mapped texture cache invalidator (synthesizable).},label={lst:tc_invalidator}]
module tex_cache_invalidator #(
  parameter ADDR_WIDTH = 40,
  parameter LINE_BYTES  = 64,
  parameter NUM_LINES   = 1024,
  parameter TAG_WIDTH   = ADDR_WIDTH - $clog2(LINE_BYTES) - $clog2(NUM_LINES)
) (
  input  wire                   clk,
  input  wire                   rst_n,
  // write-notify interface (from writers / L2)
  input  wire                   wr_notify_v,    // valid notification
  input  wire [ADDR_WIDTH-1:0]  wr_addr,        // physical address of write
  // tag array interface (for cache controller/debug)
  output reg                    invalidated     // pulse: an invalidation occurred
);

  localparam IDX_WIDTH = $clog2(NUM_LINES);
  localparam OFF_BITS  = $clog2(LINE_BYTES);

  // tag and valid storage (synchronous)
  reg [TAG_WIDTH-1:0] tag_mem [0:NUM_LINES-1];
  reg                 valid_mem [0:NUM_LINES-1];

  wire [IDX_WIDTH-1:0] index;
  wire [TAG_WIDTH-1:0] wr_tag;

  assign index = (wr_addr >> OFF_BITS) & (NUM_LINES-1);
  assign wr_tag = wr_addr >> (OFF_BITS + IDX_WIDTH);

  integer i;
  // reset init
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (i=0; i
\section{Section 5: Texture Unit Pipeline}
\subsection{Item 1:  Address calculation stage}
Following coordinate wrapping and interpolator behavior previously described, the address calculation stage turns per-fragment texture coordinates and LOD information into physical memory addresses tailored for the texture cache and backend fetch units. This stage must satisfy alignment, compression-block layout, mipmap offsets, array/cubemap slicing, and footprint requirements for filtering while preserving SIMT-friendly coherence across fragments.

Problem: map continuous UV (or 3D) coordinates plus derivatives into an address for the correct mip level, slice, and block. Analysis: for block-compressed formats and cache-line friendly tiling, address computation reduces to integer texel coordinates, block indices, and per-mipbase offsets. Let texel coordinates $(u,v)$ be quantized to integer texels; with block dimensions $B_w\times B_h$ (for BC formats typically $4\times4$) the block index is
\begin{equation}[H]\label{eq:block_idx}
\mathrm{block\_idx} \;=\; \left\lfloor\frac{v}{B_h}\right\rfloor \cdot N_x \;+\; \left\lfloor\frac{u}{B_w}\right\rfloor,
\end{equation}
where $N_x=\left\lceil W/B_w\right\rceil$ is blocks per row for texture width $W$.

LOD selection must be derivative-aware for perspective-correct filtering. The hardware computes
\begin{equation}[H]\label{eq:lod}
\rho \;=\; \max\!\big(\sqrt{(\partial_u/\partial x)^2+(\partial_v/\partial x)^2},\;\sqrt{(\partial_u/\partial y)^2+(\partial_v/\partial y)^2}\big),\quad \mathrm{LOD}=\log_2(\rho).
\end{equation}
Practical implementations approximate the $\sqrt{\cdot}$/$\log_2$ with fixed-point magnitude and leading-one detection to produce an integer mip level.

Implementation: pipeline the computation into a light-weight integer stage with valid-ready handshaking. Steps per fragment:
\begin{enumerate}
\item Quantize $u,v$ to integer texels and compute integer $\lfloor u/B_w\rfloor,\lfloor v/B_h\rfloor$.
\item Fetch per-mipbase offset from a small LUT indexed by computed LOD and slice.
\item Compute block index per equation (1) and multiply by block byte size to form final address.
\item Emit address plus metadata (mip, slice, footprint bitmask) to cache lookup stage.
\end{enumerate}

A synthesizable Verilog example below implements these steps with fixed-point texel coordinates and derivative-based approximate LOD. It outputs a block-aligned address and a clipped mip level.

\begin{lstlisting}[language=Verilog,caption={Texture address calculation (synthesizable) — block-aligned address and simple LOD},label={lst:tex_addr}]
module tex_addr_calc #(
  parameter ADDR_W = 64,
  parameter TEXCOORD_W = 32, // 16.16 fixed
  parameter DER_W = 16,      // derivative fixed
  parameter BLOCK_W = 4,
  parameter BLOCK_H = 4,
  parameter BLOCK_BYTES = 16
) (
  input  wire                  clk,
  input  wire                  rst,
  input  wire                  valid_in,
  input  wire [TEXCOORD_W-1:0] u_in, // 16.16 fixed
  input  wire [TEXCOORD_W-1:0] v_in, // 16.16 fixed
  input  wire [DER_W-1:0]      dudx, dvdx, dudy, dvdy, // unsigned magnitude approx
  input  wire [15:0]           tex_width,
  input  wire [15:0]           tex_height,
  input  wire [ADDR_W-1:0]     mip_base_offset [0:15], // per-mipbase offsets
  output reg                   valid_out,
  output reg  [ADDR_W-1:0]     addr_out,
  output reg  [3:0]            mip_out,
  output reg                   ready
);
  // simple handshake
  always @(posedge clk) ready <= ~rst;
  reg [TEXCOORD_W-1:0] u_r, v_r;
  reg [DER_W-1:0] max_der;
  integer i;
  reg [15:0] tex_x, tex_y;
  reg [15:0] block_x, block_y;
  reg [31:0] num_blocks_x;
  reg [31:0] block_idx;
  reg [3:0]  mip_r;

  always @(posedge clk) begin
    if (rst) begin valid_out <= 0; addr_out <= 0; mip_out <= 0; end
    else begin
      if (valid_in && ready) begin
        u_r <= u_in; v_r <= v_in;
        // quantize texel coords (take upper 16 bits)
        tex_x <= u_in[TEXCOORD_W-1:16];
        tex_y <= v_in[TEXCOORD_W-1:16];
        // approximate max derivative magnitude
        max_der <= dudx;
        if (dvdx > max_der) max_der <= dvdx;
        if (dudy > max_der) max_der <= dudy;
        if (dvdy > max_der) max_der <= dvdy;
        // compute blocks per row
        num_blocks_x <= (tex_width + (BLOCK_W-1)) >> $clog2(BLOCK_W); // uses constant shift when BLOCK_W pow2
        // compute block indices
        block_x <= tex_x >> $clog2(BLOCK_W);
        block_y <= tex_y >> $clog2(BLOCK_H);
        block_idx <= block_y * num_blocks_x + block_x;
        // simple leading-one to approximate mip (clamp 0..15)
        mip_r = 0;
        for (i=DER_W-1;i>=0;i=i-1) if (max_der[i]) mip_r = i;
        if (mip_r > 15) mip_r = 15;
        // compose address: base + block_idx * BLOCK_BYTES
        addr_out <= mip_base_offset[mip_r] + (block_idx * BLOCK_BYTES);
        mip_out  <= mip_r;
        valid_out <= 1;
      end else valid_out <= 0;
    end
  end
endmodule