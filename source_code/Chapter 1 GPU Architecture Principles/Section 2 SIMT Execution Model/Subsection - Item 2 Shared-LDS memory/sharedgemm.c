__global__ void tiled_gemm(float *A, float *B, float *C,
                           int N) {
  const int TILE = 32;
  __shared__ float sA[TILE][TILE]; // no padding -> may conflict
  __shared__ float sB[TILE][TILE+1]; // +1 avoids bank conflicts
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx;
  float acc = 0.0f;
  for (int k0=0; k0
\subsection{Item 3:  L1, L2, and VRAM hierarchy}
The register file and shared/LDS layers provide the lowest-latency, highest-reuse storage inside an SM; the next levels extend capacity and visibility across SMs while trading latency for bandwidth and persistence. This subsection analyses how an L1 per-SM, a unified L2 slice network, and external VRAM (HBM/GDDR) compose a hierarchy that balances SM-level throughput with global data sharing for graphics and compute workloads.

Problem: SMs issue many fine-grained memory requests from warps; without hierarchical filtering, DRAM bandwidth and latency become the bottleneck for texture sampling, shader loads/stores, tensor accumulator spills, and BVH traversal hits.

Analysis:
\begin{itemize}
\item L1 caches are usually private or partitioned per SM (or per CU). They prioritize low-latency hits for scalar loads/stores and small streaming working sets from TMUs and shader cores. Typical functionality includes write-back, write-allocate policies and optional configuration as shared/LDS to trade capacity for explicit programmer-managed locality.
\item L2 is a larger, set-associative, unified cache slice array serving all SMs; it provides cross-SM coherence for read-modify-write atomics and reduces VRAM transaction count. L2 slices are banked and connected by an on-chip crossbar or NoC to match multiple concurrent SM miss streams.
\item VRAM (HBM/GDDR) supplies raw bandwidth but with higher latency and coarser minimum transaction granularity; memory controllers implement row-buffer management, bank interleaving, and QoS scheduling to maximize sustained throughput for ROP, compute, and tensor-core flows.
\end{itemize}

Quantitative model (average access latency): let $H_1,H_2$ be hit rates for L1 and L2, and $L_1,L_2,L_m$ latencies. Then
\begin{equation}[H]\label{eq:avg_latency}
A_{\text{avg}} \;=\; H_1 L_1 \;+\; (1-H_1)\bigl(H_2 L_2 + (1-H_2) L_m\bigr).
\end{equation}
Improving $H_1$ via coalescing and better thread-local reuse yields multiplicative reductions in average latency seen by the SM.

Implementation example: a synthesizable direct-mapped L1 tag-check module that demonstrates the simple control path between SM and L2. It is intentionally minimal to show interface signals and hit/miss handling.

\begin{lstlisting}[language=Verilog,caption={Minimal direct-mapped L1 tag and valid array; issues miss_req on miss},label={lst:l1_cache}]
module l1_direct_map #(
  parameter ADDR_WIDTH = 32,
  parameter LINE_BITS  = 6,   // 64B lines
  parameter INDEX_BITS = 8    // 256 lines
) (
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   req_valid,
  input  wire [ADDR_WIDTH-1:0]  req_addr,
  output reg                    hit,
  output reg                    miss_req,    // forward to L2
  output reg [LINE_BITS*8-1:0]  data_out
);
  // derived fields
  localparam TAG_WIDTH = ADDR_WIDTH - INDEX_BITS - LINE_BITS;
  wire [LINE_BITS-1:0]  _offset = req_addr[LINE_BITS-1:0];
  wire [INDEX_BITS-1:0] _index  = req_addr[LINE_BITS +: INDEX_BITS];
  wire [TAG_WIDTH-1:0]  _tag    = req_addr[LINE_BITS+INDEX_BITS +: TAG_WIDTH];

  // tag and valid arrays
  reg [TAG_WIDTH-1:0] tag_array [0:(1<
\subsection{Item 4:  Latency and bandwidth tradeoffs}
The prior subsections established how on-chip scratch (shared/LDS) and multi-level caches interact with high-bandwidth VRAM, and how small low-latency storage improves locality. This subsection quantifies the latency–bandwidth tradeoffs and shows a lightweight synthesizable model you can use to evaluate how many outstanding requests or warps are needed to saturate a given memory channel.

Problem: GPUs must choose between minimizing per-access latency and maximizing sustained throughput. SMs use many active warps to hide long DRAM round-trip latencies, while caches and shared memory reduce both latency and off-chip traffic. The central question is: given DRAM latency $L$ (cycles), request size $S$ (bytes), and core frequency $f$ (Hz), how many outstanding requests $N$ are required to achieve sustained bandwidth $B$?

Analysis: steady-state bandwidth equals request rate times request size. If the memory pipeline can return at most one request per cycle when filled, the request rate $R_{\mathrm{req}}$ (requests/cycle) in steady-state is
\begin{equation}[H]\label{eq:reqs}
R_{\mathrm{req}} \;=\; \frac{N}{L},
\end{equation}
since each outstanding request occupies $L$ cycles on the path. Converting to bytes/sec,
\begin{equation}[H]\label{eq:bandwidth}
B \;=\; R_{\mathrm{req}}\cdot S\cdot f \;=\; \frac{N}{L}\,S\,f.
\end{equation}
Rearrange to find required outstanding requests to hit target bandwidth $B$:
\begin{equation}[H]\label{eq:N_required}
N \;=\; \frac{B\;L}{S\,f}.
\end{equation}
Example: with $L=300$ cycles, $f=1\,\mathrm{GHz}$, $S=128$ bytes, achieving $B=500\,\mathrm{GB/s}$ requires $N = (500\times10^9 \times 300)/(128 \times 1\times10^9) \approx 1172$ outstanding requests. That maps directly to warp count when each warp issues one coalesced request per memory instruction.

Implementation: to explore these tradeoffs in RTL, you can model a memory pipe parametrically. The module below accepts requests with ready/valid handshake, limits outstanding requests, and returns a response after LATENCY cycles. Use it in simulations to sweep LATENCY, DEPTH, and request size to measure how SM occupancy maps to bandwidth.

\begin{lstlisting}[language=Verilog,caption={Parameterized memory pipeline modeling latency and outstanding depth.},label={lst:mempipe}]
module mem_pipe #(
  parameter LATENCY = 128,      // round-trip cycles
  parameter DEPTH   = 2048,     // max outstanding
  parameter DATA_W  = 128       // payload width in bits
)(
  input  wire                 clk,
  input  wire                 rst,
  // request channel
  input  wire                 req_valid,
  output wire                 req_ready,
  input  wire [DATA_W-1:0]    req_data,
  // response channel
  output reg                  resp_valid,
  output reg  [DATA_W-1:0]    resp_data,
  input  wire                 resp_ready
);
  // circular slot RAM to track outstanding requests and issue after LATENCY
  localparam SIZE = (LATENCY>0) ? LATENCY+8 : 8;
  reg [DATA_W-1:0] slot_mem [0:SIZE-1];
  reg              slot_v  [0:SIZE-1];
  integer head, tail, cnt;
  initial begin head=0; tail=0; cnt=0; resp_valid=0; end

  assign req_ready = (cnt < DEPTH);

  always @(posedge clk) begin
    if (rst) begin
      head <= 0; tail <= 0; cnt <= 0; resp_valid <= 0;
      integer i; for (i=0;i 0 and slot_v at head set, produce resp after LATENCY
      if (slot_v[head]) begin
        // after LATENCY cycles this slot becomes ready; we approximate by circular delay
        resp_valid <= 1; resp_data <= slot_mem[head];
        if (resp_valid && resp_ready) begin
          slot_v[head] <= 0; head <= (head + 1) % SIZE; cnt <= cnt - 1;
          resp_valid <= 0;
        end
      end else begin
        resp_valid <= 0;
      end
    end
  end
endmodule