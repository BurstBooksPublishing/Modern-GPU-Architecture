module msaa_resolve #(
  parameter integer SAMPLES = 4,
  parameter integer CHAN_W = 10  // per-channel width (e.g., 10-bit RGB)
)(
  input  wire [(SAMPLES*CHAN_W)-1:0] sample_r, // R channel packed
  input  wire [(SAMPLES*CHAN_W)-1:0] sample_g, // G channel packed
  input  wire [(SAMPLES*CHAN_W)-1:0] sample_b, // B channel packed
  input  wire [SAMPLES-1:0] sample_mask,       // 1=sample present
  output reg  [CHAN_W+2-1:0] out_r,            // extra bits for avg
  output reg  [CHAN_W+2-1:0] out_g,
  output reg  [CHAN_W+2-1:0] out_b
);
  integer i;
  reg [CHAN_W+4:0] sum_r, sum_g, sum_b; // accumulate with headroom
  reg [3:0] cnt; // supports up to 16 samples
  always @* begin
    sum_r = 0; sum_g = 0; sum_b = 0; cnt = 0;
    for (i=0;i
\section{Section 6: Framebuffer Organization}
\subsection{Item 1:  Linear and tiled layouts}
Following the render-target cache and color-compression discussion, the choice between linear and tiled framebuffer layouts directly determines how ROPs, TMUs and the ROP cache cooperate to maximize locality, enable efficient tile-based write combining, and exploit compression metadata. The layout is an architectural knob: it trades simple address arithmetic and CPU/GPU mapability against spatial locality and memory-system friendliness for high-bandwidth render workloads.

Linear (row-major) layout maps pixel $(x,y)$ to consecutive addresses across scanlines. It is simple for CPU DMA and display scanout but suffers when small rectangular regions are updated by many ROPs: cacheline fragmentation and noncontiguous writes increase DRAM transactions and reduce compression effectiveness. Tiled layouts subdivide the framebuffer into small blocks (tiles) that are stored contiguously; this matches tile-based rasterization and ROP cache behavior and enables:
\begin{itemize}
\item tile-local write combining (reducing write amplification),
\item per-tile compression metadata co-location,
\item reduced read-modify-write when blending or multisample resolves are local.
\end{itemize}

Operational mapping for a simple row-major tile layout ($\mathrm{tileSize} = T$, screen width $W$, bytes per pixel $\mathrm{bpp}$) computes a base tile index and an intra-tile offset. The physical byte address $A(x,y)$ can be written as
\begin{equation}[H]\label{eq:tile_addr}
A(x,y)=\Big(\Big\lfloor\frac{y}{T}\Big\rfloor\cdot\frac{W}{T}+\Big\lfloor\frac{x}{T}\Big\rfloor\Big)\cdot T^2\cdot \mathrm{bpp}
+\big((y\bmod T)\cdot T + (x\bmod T)\big)\cdot \mathrm{bpp}.
\end{equation}
This layout maximizes sequential DRAM bursts for intra-tile access patterns typical of shader stores and ROP writes. A tiled layout can be enhanced with a swizzle (Morton or Z-order) to interleave $x/y$ bits and reduce multi-bank contention and improve cache set distribution.

A common implementation computes a Morton index by bit-interleaving the tile-local coordinates; this remapping increases spatial locality along both axes and reduces worst-case strided accesses. The following C snippet shows a compact Morton-mapped address calculator used in a ROP cache controller or DMA engine:

\begin{lstlisting}[language=C,caption={Morton-mapped tile address calculation (bytes)},label={lst:mortonAddr}]
uint64_t mortonInterleave(uint32_t v){ // interleave lower 16 bits
  uint64_t x = v & 0xFFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

uint64_t computeAddr(uint32_t x, uint32_t y, uint32_t tileSize,
                     uint32_t screenW, uint32_t bpp){
  uint32_t tx = x / tileSize, ty = y / tileSize; // tile index
  uint32_t lx = x % tileSize, ly = y % tileSize; // intra-tile
  uint64_t morton = (mortonInterleave(ty) << 1) | mortonInterleave(tx); // tile Z-order
  uint64_t tileBytes = (uint64_t)tileSize * tileSize * bpp;
  return morton * tileBytes + (uint64_t)(ly*tileSize + lx) * bpp; // byte address
}