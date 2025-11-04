module fast_clear_meta #(
  parameter TILE_COUNT = 1024,
  parameter PALETTE_IDX_BITS = 4
)(
  input  wire clk,
  input  wire rst,
  // clear command
  input  wire clear_v,
  input  wire [$clog2(TILE_COUNT)-1:0] clear_tile, // tile index
  input  wire [PALETTE_IDX_BITS-1:0] clear_idx,
  // invalidate (on write) command
  input  wire inv_v,
  input  wire [$clog2(TILE_COUNT)-1:0] inv_tile,
  // query port
  input  wire q_v,
  input  wire [$clog2(TILE_COUNT)-1:0] q_tile,
  output reg  q_fast_clear, // metadata result
  output reg  [PALETTE_IDX_BITS-1:0] q_idx
);
  // metadata arrays
  reg fast_clear_flag [0:TILE_COUNT-1];
  reg [PALETTE_IDX_BITS-1:0] fast_clear_idx [0:TILE_COUNT-1];

  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i
\subsection{Item 3:  Metadata management}
The previous discussion of delta color compression and the fast clear optimization motivated keeping per-tile compression state and clear-key values close to the ROP to avoid unnecessary VRAM traffic. Metadata management is the mechanism that preserves those invariants across ROP cache hits, evictions, fast clears, and resolve operations.

Problem: the ROP must know, for each tile or block, whether the color payload is compressed, the compression mode, the current fast-clear key, and whether the tile is dirty. These metadata updates must be atomic with pixel writes to preserve correctness during concurrent fragment streams from multiple SMs and when the TMU or DMA performs partial writes or resolves. Excessive metadata read/writes inflate memory bandwidth and add latency; insufficient metadata locality reduces compression efficiency.

Analysis: represent the set of metadata fields per tile as a fixed-width word: bits for compressed flag, compression mode, clear\_key (e.g., 32-bit hash or palette index), and dirty/valid bits. The total on-chip metadata SRAM size is
\begin{equation}[H]\label{eq:meta_size}
\mathrm{Bytes}_{\mathrm{meta}} = \frac{N_{\mathrm{tiles}}\cdot B_{\mathrm{tile}}}{8},
\end{equation}
where $N_{\mathrm{tiles}}$ is number of tiles per render target and $B_{\mathrm{tile}}$ is bits per tile metadata. Example: $N_{\mathrm{tiles}}=16{,}384$ and $B_{\mathrm{tile}}=64$ gives 131,072 bytes (128 KiB) of metadata per render target.

Implementation: a small, single-ported synchronous metadata SRAM indexed by tile coordinate provides low-latency lookup. On a cache miss or tile eviction, the controller writes metadata back to VRAM using a batched write to amortize latency. The metadata controller supports atomic compare-and-swap for clear-key updates to support concurrent fast clear and partial writes.

\begin{lstlisting}[language=Verilog,caption={Tile metadata SRAM with atomic CAS and read/write interface},label={lst:meta_verilog}]
module tile_meta #(
  parameter INDEX_WIDTH = 14,      // supports 16K tiles
  parameter DATA_WIDTH  = 64
)(
  input  wire                     clk,
  input  wire                     rst_n,
  // Read port
  input  wire [INDEX_WIDTH-1:0]   idx_r,
  input  wire                     rd_en,
  output reg  [DATA_WIDTH-1:0]    rd_data,
  // Write port
  input  wire [INDEX_WIDTH-1:0]   idx_w,
  input  wire                     wr_en,
  input  wire [DATA_WIDTH-1:0]    wr_data,
  // Atomic CAS: if memory equals cmp_data, replace with new_data, return success
  input  wire                     cas_en,
  input  wire [DATA_WIDTH-1:0]    cmp_data,
  input  wire [DATA_WIDTH-1:0]    new_data,
  output reg                      cas_success
);
  // Simple synchronous RAM
  reg [DATA_WIDTH-1:0] mem [0:(1<
\subsection{Item 4:  Compression efficiency}
The preceding discussion of per-tile metadata layouts and fast-clear encodings established how metadata and special clear codes reduce redundant memory traffic; here we quantify how those mechanisms combine to determine end-to-end compression efficiency and the resulting ROP bandwidth impact.

Problem: render targets are written by many SMs through the ROP (raster operations) pipeline; compression reduces external memory traffic but incurs metadata storage, encode/decode latency in the ROP cache, and fragmentation when incompressible pixels intermingle with compressible regions. To analyze tradeoffs, model a tiled framebuffer region composed of $N$ fixed-size compression blocks of size $B$ bytes each. Let $p$ be the fraction of blocks that are compressible to an average payload size $s_c$ bytes; let $m$ be per-block metadata overhead in bytes (including tags, checksums, and state bits). Then the uncompressed and compressed sizes are:
\begin{equation}[H]\label{eq:eff_cr}
S_{u}=N\cdot B,\qquad
S_{c}=N\cdot\bigl(p\,s_{c} + (1-p)\,B + m\bigr).
\end{equation}
The effective compression ratio is
\begin{equation}[H]\label{eq:cr}
\mathrm{CR}_{\mathrm{eff}}=\frac{S_{u}}{S_{c}}=\frac{B}{p\,s_{c} + (1-p)\,B + m}.
\end{equation}
Bandwidth saved relative to an uncompressed write-back is $1-1/\mathrm{CR}_{\mathrm{eff}}$.

Analysis: use this model to explore regimes relevant to real workloads.
\begin{itemize}
\item If $p\rightarrow1$ (large homogeneous regions, e.g., clear or flat shading), $s_c\ll B$ and metadata dominates; thus minimize $m$ with aggregated metadata entries (one tag per tile rather than per block).
\item For high-frequency texture-compute shading with per-pixel variance, $p$ is small and compression yields little benefit; hardware should bypass compression to avoid decode latency in the ROP cache.
\end{itemize}

Implementation: the ROP cache should:
\begin{enumerate}
\item Track per-tile compressible-run length to amortize $m$ over multiple blocks.
\item Provide a fast path for fast-clear codes (zero-cost writes into metadata only).
\item Fall back to pass-through writes when predicted $\mathrm{CR}_{\mathrm{eff}}$ is below a threshold to avoid added latency.
\end{enumerate}

A small utility to compute $\mathrm{CR}_{\mathrm{eff}}$ and bandwidth savings (useful in architecture simulation) follows.

\begin{lstlisting}[language=C++,caption={Compute effective compression ratio and bandwidth savings.},label={lst:cr_calc}]
#include 
#include 
// B: block size bytes, p: compressible fraction, s_c: compressed payload (bytes), m: metadata bytes
void calc(double B, double p, double s_c, double m, int N=1024){
  double S_u = N*B;
  double S_c = N*(p*s_c + (1.0-p)*B + m);
  double CR = S_u / S_c;
  double bw_saved = 1.0 - 1.0/CR;
  std::cout<
\section{Section 4: Render Target Cache}
\subsection{Item 1:  Color and depth cache structure}
Building on the earlier discussion of depth/stencil testing and blending paths, the render target cache (RTC) bridges pixel shader outputs and the ROPs by holding color and depth tiles close to the ROP units to reduce external memory traffic and enable in-cache blending, depth tests, and compression metadata updates. The remainder of this subsection analyzes a practical color/depth cache structure, gives a compact synthesizable controller implementing write-combine and write-back semantics, and highlights concrete design trade-offs for ROP designers.

Problem and analysis: ROP stages issue high-rate, small-granularity writes (pixels or samples) that must be coalesced into larger DRAM-friendly transactions. A color/depth RTC therefore must:
\begin{itemize}
\item co-locate multiple pixels per cache line (tile-line) to amortize DRAM latency,
\item support atomic per-pixel depth test and blending logic in-cache to avoid read-modify-write round-trips,
\item maintain tags, valid and dirty state, and a small eviction policy to flush lines to the frame buffer (possibly compressed).
\end{itemize}

Key sizing relation: given an address width $A$ (bits), line size $L$ (bytes), and number of cache lines $S$, the per-line tag width is
\begin{equation}[H]\label{eq:tagbits}
\text{TAG\_BITS} \;=\; A - \log_2(S) - \log_2(L).
\end{equation}
This governs tag-array storage cost and affects associativity choices because larger tags per line reduce effective data storage for a fixed SRAM budget.

Implementation sketch: use a direct-mapped or small set-associative RTC with
\begin{itemize}
\item line size tuned to hold a tile strip (e.g., 64–256 bytes),
\item per-line metadata: tag, valid, dirty, compression-state,
\item in-cache depth compare unit (early-Z) that can mark a pixel as discarded without writing color bytes,
\item write-combine buffer that accumulates multiple pixel writes to the same line before eviction, and
\item write-back eviction that optionally applies DCC (delta color compression) before issuing DRAM writes.
\end{itemize}

The following Verilog module is a compact, synthesizable direct-mapped RTC controller implementing tag, valid, dirty, read-hit, write-hit, and dirty-line eviction. It omits the blending and depth-compare datapath for brevity but provides the atomic in-cache storage and eviction mechanism needed by ROP hardware.

\begin{lstlisting}[language=Verilog,caption={Simple direct-mapped render target cache controller (synthesizable)},label={lst:rt_cache}]
module rt_cache_controller #(
  parameter ADDR_WIDTH = 32,
  parameter LINE_BYTES = 64,
  parameter NUM_LINES  = 256,
  parameter DATA_WIDTH = 32
)(
  input  wire                      clk,
  input  wire                      rst,
  input  wire                      rd_en,
  input  wire                      wr_en,
  input  wire [ADDR_WIDTH-1:0]     addr,
  input  wire [DATA_WIDTH-1:0]     wdata,
  output reg                       hit,
  output reg  [DATA_WIDTH-1:0]     rdata,
  // memory interface for line writeback
  output reg                       mem_wr,
  output reg  [ADDR_WIDTH-1:0]     mem_addr,
  output reg  [LINE_BYTES*8-1:0]   mem_wdata
);

  // compile-time clog2
  function integer clog2; input integer v; begin v=v-1; for(clog2=0; v>0; clog2=clog2+1) v=v>>1; end endfunction

  localparam OFFSET_BITS = clog2(LINE_BYTES);
  localparam WORDS_PER_LINE = LINE_BYTES / (DATA_WIDTH/8);
  localparam WORD_OFF_BITS = clog2(WORDS_PER_LINE);
  localparam INDEX_BITS = clog2(NUM_LINES);
  localparam TAG_BITS = ADDR_WIDTH - INDEX_BITS - OFFSET_BITS;

  // arrays
  reg [DATA_WIDTH-1:0] data_array [0:NUM_LINES*WORDS_PER_LINE-1];
  reg [TAG_BITS-1:0]   tag_array  [0:NUM_LINES-1];
  reg                  valid      [0:NUM_LINES-1];
  reg                  dirty      [0:NUM_LINES-1];

  // index/tag/offset extraction
  wire [INDEX_BITS-1:0] index = addr[OFFSET_BITS +: INDEX_BITS];
  wire [WORD_OFF_BITS-1:0] word_off = addr[OFFSET_BITS +: WORD_OFF_BITS];
  wire [TAG_BITS-1:0] tag = addr[ADDR_WIDTH-1 -: TAG_BITS];

  integer i;
  // simple synchronous read/write and eviction
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i
\subsection{Item 2:  Tile-based write combining}
Building on the cache organization for color and depth, tile-based write combining centralizes per-tile pixel updates to reduce DRAM transaction overhead and improve ROP throughput. The next paragraphs analyze the problem, propose a practical combiner microarchitecture, and show a synthesizable Verilog implementation that can be integrated into a ROP cache controller.

Problem and analysis:
\begin{itemize}
\item Fragment shading produces many scattered, small writes (per-pixel or per-sample) that, if sent individually to L2/VRAM, generate poor bus utilization and high command overhead. Tile-based combining coalesces these writes inside the render target cache (RTC) by buffering a tile's pixels until a wide, cache-line-aligned write can be issued.
\item Let tile pixel count be $P = W_{\mathrm{tile}}\times H_{\mathrm{tile}}$ and bytes per pixel $B$. The tile footprint is $S = P\cdot B$. With a memory burst width $W_{\mathrm{burst}}$, the minimum number of bursts to write a full tile is
\begin{equation}[H]\label{eq:bursts}
N_{\mathrm{burst}} \;=\; \left\lceil \frac{S}{W_{\mathrm{burst}}} \right\rceil .
\end{equation}
Combining reduces per-pixel command overhead; effective bandwidth improvement is roughly proportional to $S / (N_{\mathrm{burst}}\cdot W_{\mathrm{burst}})$ when metadata and alignment losses are small.
\end{itemize}

Practical microarchitecture:
\begin{enumerate}
\item Per-tile buffer: small SRAM storing $P$ pixel entries plus a valid mask and per-sample mask for MSAA. Use one buffer per active tile or a small set of tile slots with LRU eviction.
\item Merge semantics: on a fragment write, merge the pixel using the blending pipeline if necessary, then set the corresponding valid bit. Deferred blending can also be used if hardware supports atomic ROP ops.
\item Flush policy:
\begin{itemize}
\item Flush when tile buffer becomes full (all valid bits set), when tile residency age exceeds threshold, or on context/frame boundaries.
\item Flush must produce aligned bursts and optionally run compression (DCC) before issuing memory writes.
\end{itemize}
\item Atomicity: blending and stencil/z tests must be resolved before the final tile write to avoid read-modify-write to memory; early-Z and hierarchical-Z integration helps reject pixels before they enter the combiner.
\end{enumerate}

Implementation example (synthesizable Verilog): a simplified write-combiner slot that accepts per-pixel writes and emits aligned memory bursts on flush.

\begin{lstlisting}[language=Verilog,caption={Tile write-combiner slot (simplified, synthesizable)},label={lst:tile_wc}]
module tile_write_combiner #(
  parameter TILE_W = 8, TILE_H = 8, BPP = 4, P = TILE_W*TILE_H,
  parameter BURST_WORDS = 8 // words per memory burst
)(
  input  wire                 clk, rst,
  // pixel write from ROP pipeline (tile-local index)
  input  wire                 wr_valid,
  input  wire [$clog2(P)-1:0] wr_index,
  input  wire [BPP*8-1:0]     wr_data,
  input  wire                 wr_mask,   // write enable for pixel
  // flush interface
  input  wire                 flush_req,
  output reg                  mem_valid,
  output reg  [31:0]          mem_addr,
  output reg  [BPP*8-1:0]     mem_data,
  input  wire                 mem_ready
);
  // per-pixel storage and valid bits
  reg [BPP*8-1:0] pix_mem [0:P-1];
  reg [P-1:0]     valid_mask;
  integer i;
  // simple write logic
  always @(posedge clk) begin
    if (rst) begin
      valid_mask <= 0;
      mem_valid <= 0;
    end else begin
      if (wr_valid && wr_mask) begin
        pix_mem[wr_index] <= wr_data; // merge done earlier if needed
        valid_mask[wr_index] <= 1'b1;
      end
      // flush when requested and there is any valid pixel
      if (flush_req && |valid_mask) begin
        // start streaming burst from index 0..P-1, aligned externally
        // simple blocking stream for illustration
        for (i=0;i
\subsection{Item 3:  Compression integration}
The previous subsections showed how tile-based write combining concentrates pixel writes and how color/depth cache structures partition data and tags for low-latency ROP access. Building on those, compression integration must place metadata and compressor/decompressor paths inside the render-target cache (RTC) so tile-local writes benefit without breaking blending, fast-clear, or MSAA semantics.

Problem statement and analysis. Compression reduces external memory bandwidth but incurs extra latency, metadata storage, and update complexity. For block-based delta color compression (DCC) the RTC must:
\begin{itemize}
\item store per-block metadata (compressed, raw, fast-clear tag),
\item update metadata atomically with write-combining merges,
\item drive decompression on reads before blending or when partial writes force expansion,
\item support fast-clear by publishing a clear-color plus per-block cleared flag instead of writing full blocks.
\end{itemize}

A simple analytic cost model exposes trade-offs. Let $R$ be the compression ratio (original\_size/compressed\_size), $S$ the block size in bytes, and $M$ the metadata overhead per block in bytes. Net fractional bandwidth used per block is
\begin{equation}[H]\label{eq:bandwidth_net}
B_{\mathrm{net}} = \frac{1}{R} + \frac{M}{S}.
\end{equation}
Compression is beneficial when $B_{\mathrm{net}} < 1$. For example, $R=4$, $S=256$, $M=4$ yields $B_{\mathrm{net}}=0.25+0.0156\approx0.265$, a 73.5\% bandwidth saving.

Implementation sketch (hardware roles). Integrate a lightweight metadata RAM collocated with RTC tags so lookups are single-cycle with tag hits. Compression hardware must be on the write path that receives tile-combined pixels from the ROP merge stage; on cache eviction, the compressor produces compressed payload plus metadata written to VRAM. On read misses, the decompressor expands blocks into RTC line buffers before ROP uses them.

The following synthesizable Verilog implements a compact metadata unit that:
\begin{itemize}
\item provides synchronous read after request,
\item accepts metadata updates on writes and evictions,
\item supports atomic fast-clear set/clear operations.
\end{itemize}

\begin{lstlisting}[language=Verilog,caption={DCC metadata RAM and control (synthesizable)},label={lst:dcc_meta}]
module dcc_meta_unit #(
  parameter ADDR_WIDTH = 14,                    // block address width
  parameter META_WIDTH = 8                       // per-block metadata bits
)(
  input  wire                   clk,
  input  wire                   rst_n,
  // read port
  input  wire                   rd_req,           // request read
  input  wire [ADDR_WIDTH-1:0]  rd_addr,
  output reg                    rd_valid,
  output reg [META_WIDTH-1:0]   rd_meta,
  // write/update port (from ROP/compressor)
  input  wire                   wr_req,
  input  wire [ADDR_WIDTH-1:0]  wr_addr,
  input  wire [META_WIDTH-1:0]  wr_meta,
  output reg                    wr_ack,
  // fast-clear command (atomic set of fast-clear flag)
  input  wire                   fast_clear_req,
  input  wire [META_WIDTH-1:0]  fast_clear_meta // encoded clear state
);
  localparam NUM_BLOCKS = (1<
\subsection{Item 4:  Eviction policy design}
These ideas build on compression metadata placement discussed earlier and on tile-based write combining that aggregates fragment writes before memory flush. Eviction policy must therefore reconcile compression state, tile atomicity, and write-combiner batching to minimize ROP-to-memory stalls.

Problem: when RTC (render target cache) capacity is exceeded or tiles must be flushed for resolve, the controller must pick lines to evict that minimize DRAM traffic and preserve compression efficiency while meeting ordering constraints from blending, MSAA, and fast clear. Analysis identifies three orthogonal costs:
\begin{enumerate}
\item Write bandwidth cost: evicting a dirty compressed line costs less than an uncompressed line.
\item Latency-criticality: lines holding pixels still being rasterized should be retained.
\item Compression metadata coherence: evicting lines with high compression ratio hurts DCC efficiency.
\end{enumerate}

We express a tunable eviction score $S$ per cache line to combine metrics, used to select victim lines:
\begin{equation}[H]\label{eq:evict_score}
S = \alpha\cdot \mathrm{Age} + \beta\cdot \mathrm{Dirty} + \gamma\cdot (1-\mathrm{CompRatio})
\end{equation}
where Age is normalized recency, Dirty is 1 for modified lines, and $\mathrm{CompRatio}\in[0,1]$ estimates compression benefit; coefficients $\alpha,\beta,\gamma$ set policy bias.

Implementation: practical GPU ROPs use lightweight, synthesizable hardware:
\begin{itemize}
\item A per-set PLRU tree approximates LRU with minimal bits.
\item A small saturating counter encodes Age for long-lived priority inversion correction.
\item Per-way metadata includes Dirty bit, CompRatio estimate bucket, and TileID to favor tile affinity.
\item Eviction arbitration prioritizes: (a) non-active tiles, (b) low-compression lines, (c) dirty lines only when write-combiner ready, to allow atomic batched writebacks.
\end{itemize}

The following Verilog module is a synthesizable eviction controller implementing PLRU selection with compression-aware prioritization and explicit writeback gating to the write-combiner. Inputs include per-way metadata; outputs assert evict_grant and evict_addr. Comments are brief in-code notes.

\begin{lstlisting}[language=Verilog,caption={RTC eviction controller (synthesizable)},label={lst:rtc_evict}]
module rtc_eviction_controller #(
  parameter SETS = 256,
  parameter WAYS = 4,
  parameter TAG_W = 32,
  parameter TILEID_W = 16
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire                 evict_req,       // request to evict a set
  input  wire [$clog2(SETS)-1:0] evict_set,
  input  wire [WAYS-1:0]      way_dirty,       // per-way dirty bit
  input  wire [WAYS-1:0]      way_active,      // per-way active-in-tile (avoid)
  input  wire [WAYS*2-1:0]    way_compr,       // 2-bit comp ratio bucket per way
  output reg                  evict_valid,
  output reg  [$clog2(WAYS)-1:0] evict_way,
  output reg  [TAG_W-1:0]     evict_tag
);

  // PLRU tree bits per set (synthesizable SRAM implied by reg array)
  reg [WAYS-2:0] plru_bits [0:SETS-1]; // binary tree bits
  // small tag RAM port assumed elsewhere; here we only select way index

  integer i;
  // simple priority: prefer ways that are not active, minimal comp bucket, then non-dirty if combiner busy.
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      evict_valid <= 0;
      evict_way <= 0;
      for (i=0;i
\section{Section 5: Multi-Sample Anti-Aliasing}
\subsection{Item 1:  Sampling positions and coverage masks}
Building on the preceding ROP topics that determine per-pixel depth/stencil and blending behavior, we now focus on where within a pixel samples are taken and how their in/out results are encoded as coverage masks for downstream ROP stages and resolves.

MSAA problem statement and analysis. The renderer must decide a small set of sample coordinates inside each pixel and test triangle coverage at each coordinate to form a per-pixel bitmask. Sample positions are typically expressed relative to the pixel center in normalized pixel space (range roughly $[-0.5,0.5]$). Common placement strategies include regular grid, rotated/stratified grids, and jittered Poisson-like distributions; the chosen pattern directly impacts aliasing reduction and cache-friendly coherency across neighboring pixels. Coverage evaluation reduces to computing edge functions (signed line equations) at each sample position and applying a top-left or fill rule. For a single edge with coefficients $(a,b,c)$, the signed value at sample $s$ with offset $(o_x^{(s)},o_y^{(s)})$ is
\begin{equation}[H]\label{eq:edge_eval}
E^{(s)} = a\,(x_\text{pixel}+o_x^{(s)}) + b\,(y_\text{pixel}+o_y^{(s)}) + c,
\end{equation}
and a sample is covered if all three triangle edges satisfy the fill criterion (e.g., $E^{(s)}\ge 0$ with consistent conventions). The coverage mask for $S$ samples is the $S$-bit word
\begin{equation}[H]\label{eq:mask}
M=\sum_{s=0}^{S-1}\; \big( \mathbf{1}\{ \text{sample }s\ \text{covered}\} \big)\,2^{s}.
\end{equation}

Hardware implementation notes. ROP sample-test units implement Equation (1) in fixed-point and evaluate multiple samples either in parallel (bit-parallel comparators and multiply-add arrays) or in a small pipeline. To minimize datapath width, edge coefficients are precomputed in the triangle setup stage using the same fixed-point format, and sample offsets are stored in a tiny ROM indexed by \lstinline|SAMPLE_COUNT|. Because the comparison is simple sign checking, many designs pack per-sample results into a word then use population-count or table lookup for fast decisions (e.g., early-compress or fast clear integration).

A compact synthesizable Verilog module that computes the coverage mask for a 4-sample pattern is shown below. It assumes precomputed edge coefficients in Q.frac fixed-point and uses sample offsets defined in Q.frac integers.

\begin{lstlisting}[language=Verilog,caption={MSAA coverage mask generator (4x) — fixed-point Q8 offsets},label={lst:msaa_cov}]
module msaa_covgen #(
  parameter integer COEFF_W = 24, // signed width for a,b,c
  parameter integer FRAC = 8      // fractional bits (Q.FRAC)
) (
  input  signed [COEFF_W-1:0] a0, b0, c0, // edge0 coeffs
  input  signed [COEFF_W-1:0] a1, b1, c1, // edge1 coeffs
  input  signed [COEFF_W-1:0] a2, b2, c2, // edge2 coeffs
  output reg [3:0] cov_mask                  // 4-sample coverage mask
);
  // sample offsets in Q.FRAC (here a simple 2x2 grid: ±0.25)
  function signed [FRAC-1:0] sx(input integer i);
    case(i)
      0: sx = - (1<< (FRAC-2)); // -0.25 -> -64 when FRAC=8
      1: sx =   (1<< (FRAC-2));
      2: sx = - (1<< (FRAC-2));
      3: sx =   (1<< (FRAC-2));
      default: sx = 0;
    endcase
  endfunction
  function signed [FRAC-1:0] sy(input integer i);
    case(i)
      0: sy = - (1<< (FRAC-2));
      1: sy = - (1<< (FRAC-2));
      2: sy =   (1<< (FRAC-2));
      3: sy =   (1<< (FRAC-2));
      default: sy = 0;
    endcase
  endfunction

  integer s;
  // combinational evaluation: compute E for each edge and sample
  always @(*) begin
    cov_mask = 4'b0;
    for (s=0; s<4; s=s+1) begin
      // multiply a*ox and b*oy; scale adjust: ox,oy are Q.FRAC
      // products have extra FRAC bits; discard low FRAC bits by >>FRAC
      signed [COEFF_W+FRAC-1:0] e0 = a0 * sx(s) + b0 * sy(s) + (c0 << FRAC);
      signed [COEFF_W+FRAC-1:0] e1 = a1 * sx(s) + b1 * sy(s) + (c1 << FRAC);
      signed [COEFF_W+FRAC-1:0] e2 = a2 * sx(s) + b2 * sy(s) + (c2 << FRAC);
      // shift back by FRAC to align fixed-point; check sign bit
      if ((e0 >>> FRAC) >= 0 && (e1 >>> FRAC) >= 0 && (e2 >>> FRAC) >= 0)
        cov_mask[s] = 1'b1;
    end
  end
endmodule