module rr_dispatch #(
  parameter N_CU = 4,
  parameter IDW = $clog2(N_CU)
)(
  input clk, input rst,
  input in_valid, input [31:0] in_pkt,
  output reg in_ready,
  output reg [N_CU-1:0] cu_valid,
  output reg [31:0] cu_pkt [0:N_CU-1],
  input  [N_CU-1:0] cu_ready
);
  reg [IDW-1:0] ptr;
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      ptr <= 0;
      in_ready <= 1;
      cu_valid <= 0;
    end else begin
      // find next ready CU starting from ptr
      in_ready <= 0;
      for (i=0;i
\subsection{Item 3:  Mobile GPUs (Mali, Adreno)}
Building on the previous case studies of desktop-class throughput optimizations and heterogeneous accelerator integration, mobile GPU designs shift emphasis to minimizing energy per frame and reducing DRAM traffic while retaining usable shader throughput for graphics and ML tasks.

Mobile vendors converge on three operational levers: on-chip tiling to exploit locality, aggressive bandwidth compression, and fine-grained power management. ARM Mali implementations center on a hardware tiler and tile-based deferred rendering (TBDR) that stages raster work into small on-chip tiles, allowing early-Z and blending to occur without round-trips to external DRAM. Qualcomm Adreno designs also use tiling and hierarchical binning but pair that tiler with low-latency ROP/TMU paths and proprietary bandwidth compression (e.g., UBWC) to sustain interactive frame rates. Both stacks integrate ISP/NPU interfaces so shader cores (SIMT-like units) can be repurposed for lightweight ML kernels, while NPUs handle heavier DNN inference.

Quantifying bandwidth requirements clarifies the trade-offs. For a framebuffer of width $W$, height $H$, bytes-per-pixel $b$, and target frame rate $f$, the raw bandwidth per second is approximately
\begin{equation}[H]\label{eq:bandwidth}
B_{\text{raw}} = W \cdot H \cdot b \cdot f.
\end{equation}
With an on-chip tiler and compression factor $c>1$, the external bandwidth requirement reduces to $B_{\text{ext}} \approx B_{\text{raw}}/c$, directly lowering system power at the cost of on-chip SRAM area and the logic for compression/decompression.

Architectural analysis and implementation considerations:
\begin{itemize}
\item Tile size selection trades on-chip SRAM versus parallelism: larger tiles increase cache reuse (good for complex shading, texture reuse) but require bigger tile buffers and incur higher latency for partial-frame updates.
\item Compression (AFBC, UBWC) reduces $B_{\text{ext}}$ but adds encode/decode latency in the TMU/ROP datapaths; designs must pipeline these stages to avoid starving SM/CU execution units.
\item Power management uses cluster-level power gating and DVFS; fast stitch points (tiler completion, early-Z rejects) create natural windows to gate TMU or shader clusters.
\end{itemize}

A compact, synthesizable RTL example of a tile buffer dual-port RAM used in a mobile tiler demonstrates an implementation pattern that balances area and single-cycle access for local ROP/TMU operations:

\begin{lstlisting}[language=Verilog,caption={Simple synthesizable tile buffer (dual-port RAM).},label={lst:tilebuf}]
module tile_buffer #(
  parameter TILE_PIXELS = 1024,
  parameter DATA_W = 32,
  parameter ADDR_W = $clog2(TILE_PIXELS)
)(
  input  wire                    clk,
  input  wire                    wr_en,      // write enable
  input  wire [ADDR_W-1:0]       wr_addr,
  input  wire [DATA_W-1:0]       wr_data,
  input  wire                    rd_en,      // read enable
  input  wire [ADDR_W-1:0]       rd_addr,
  output reg  [DATA_W-1:0]       rd_data
);
  // inferred dual-port BRAM (one port read, one port write)
  reg [DATA_W-1:0] mem [0:TILE_PIXELS-1];

  always @(posedge clk) begin
    if (wr_en) mem[wr_addr] <= wr_data;          // write port
    if (rd_en)  rd_data <= mem[rd_addr];         // read port (registered)
  end
endmodule