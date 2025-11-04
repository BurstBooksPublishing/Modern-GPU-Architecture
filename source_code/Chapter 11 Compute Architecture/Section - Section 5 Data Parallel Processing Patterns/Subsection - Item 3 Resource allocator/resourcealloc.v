module resource_allocator #(
  parameter REG_BLOCKS = 512,     // number of reg blocks
  parameter SHM_BLOCKS = 256,     // number of shared-memory blocks
  parameter BLOCK_ID_W = 10       // ceil(log2(max(REG_BLOCKS,SHM_BLOCKS)))
) (
  input  wire                 clk,
  input  wire                 rstn,
  // request from dispatch
  input  wire                 req_valid,
  input  wire [15:0]          req_reg_blocks, // number of reg blocks requested
  input  wire [15:0]          req_shm_blocks, // number of shm blocks requested
  input  wire [15:0]          req_wgid,       // work-group id
  output reg                  req_ready,
  output reg                  grant,
  output reg [BLOCK_ID_W-1:0] grant_reg_base,
  output reg [BLOCK_ID_W-1:0] grant_shm_base,
  // release interface
  input  wire                 rel_valid,
  input  wire [15:0]          rel_reg_blocks, // blocks to release
  input  wire [15:0]          rel_shm_blocks,
  input  wire [15:0]          rel_base_reg,   // base indices for release
  input  wire [15:0]          rel_base_shm
);

  // allocation bitmaps
  reg [REG_BLOCKS-1:0] reg_bmap;
  reg [SHM_BLOCKS-1:0] shm_bmap;

  integer i, j;
  // simple single-cycle first-fit search (combinational)
  reg found_reg;
  reg [BLOCK_ID_W-1:0] candidate_reg;
  reg found_shm;
  reg [BLOCK_ID_W-1:0] candidate_shm;

  always @(*) begin
    found_reg = 1'b0;
    candidate_reg = {BLOCK_ID_W{1'b0}};
    // first-fit contiguous scan for reg blocks
    for (i=0; i<=REG_BLOCKS-1; i=i+1) begin
      if (!found_reg) begin
        if (i + req_reg_blocks <= REG_BLOCKS) begin
          // check all blocks free
          found_reg = 1'b1;
          for (j=0; j
\subsection{Item 4:  Barrier synchronization logic}
The work-group scheduler and resource allocator place and size work-groups and assign warps to SM slots; the barrier logic therefore must map to those allocations, track the number of participating warps, and interoperate with the SM's warp context queues and shared memory banks. In particular, the scheduler provides a work-group ID and an expected participant count; the allocator fixes warp-to-SM mappings that the barrier hardware uses to broadcast release events.

Barrier synchronization solves the problem of stalling multiple warps until every thread in a work-group reaches a synchronization point (for example, CUDA \_\_syncthreads). Analysis: implement a per-work-group barrier bank indexed by the scheduler-assigned slot. Each bank stores:
\begin{itemize}
\item expected\_count: number of participating warps $W$.
\item arrival\_count: running tally of warps that have signaled arrival.
\item gen: a generation bit used for sense-reversing to distinguish successive uses of the same barrier slot.
\end{itemize}

The release condition is simple and robust:
\begin{equation}[H]\label{eq:release}
\text{release when } arrival\_count = expected\_count.
\end{equation}
On release the bank toggles gen and clears arrival\_count to zero atomically so reused work-groups do not observe stale arrivals.

Implementation notes: warps assert a single-cycle pulse barrier\_req with their work\_group\_id and a warp\_active flag indicating participation (compilers ensure structured barriers so masked threads are accounted for). The barrier bank performs an atomic increment on arrival, compares to expected, and, on equality, drives a per-warp release vector or a release token that the SM distributes to the warps mapped to that work-group slot. A generation bit is returned with the release so that stalled warps can observe a change in generation rather than relying on resetting signals.

A synthesizable Verilog implementation follows. It implements a simple multi-slot barrier bank with parameterizable slots and maximum warps per group. Each slot handles increment, compare, and generation toggle. Comments explain signal roles.

\begin{lstlisting}[language=Verilog,caption={Work-group barrier bank (synthesizable).},label={lst:barrier_bank}]
module barrier_bank #(
  parameter SLOTS = 8,               // concurrent work-groups per SM
  parameter MAX_WARPS = 16,          // max warps per work-group
  parameter ID_W = 3                 // bits for slot id (log2(SLOTS))
) (
  input  wire                 clk,
  input  wire                 reset,
  // request: pulse when a warp reaches barrier
  input  wire                 req_valid,
  input  wire [ID_W-1:0]      req_slot,   // assigned work-group slot
  input  wire [$clog2(MAX_WARPS)-1:0] req_inc, // increment (1 if active warp)
  // write expected count at launch
  input  wire                 set_expected,
  input  wire [ID_W-1:0]      set_slot,
  input  wire [$clog2(MAX_WARPS)-1:0] set_expected_count,
  // outputs: release pulse per slot
  output reg                  release_valid,
  output reg  [ID_W-1:0]      release_slot
);

  // storage arrays
  reg [$clog2(MAX_WARPS)-1:0] expected_count [0:SLOTS-1];
  reg [$clog2(MAX_WARPS)-1:0] arrival_count  [0:SLOTS-1];
  reg                         gen_bit        [0:SLOTS-1];

  integer i;
  // init
  always @(posedge clk) begin
    if (reset) begin
      release_valid <= 1'b0;
      release_slot  <= {ID_W{1'b0}};
      for (i=0;i
\subsection{Item 5:  Compute kernel testbench}
The test scenario combines the resource allocation constraints and barrier synchronization behavior discussed previously, exercising how a dispatched work-group uses allocated register and shared-memory resources and stalls at barriers under contention.

A robust compute-kernel testbench must solve: how to deterministically drive the dispatch path, emulate a small SIMT execution stream, verify barrier release semantics, and measure occupancy-driven throughput. Analysis shows resident work-groups per SM is bounded by multiple resources; a compact expression is
\begin{equation}[H]\label{eq:resident_wg}
N_{\text{resident}} \;=\; \min\!\left(\left\lfloor\frac{R_{\text{SM}}}{R_{\text{wg}}}\right\rfloor,\;\left\lfloor\frac{S_{\text{SM}}}{S_{\text{wg}}}\right\rfloor,\;W_{\max}\right)
\end{equation}
where $R_{\text{SM}}$ is registers per SM, $R_{\text{wg}}$ registers per work-group, $S_{\text{SM}}$ shared memory per SM, $S_{\text{wg}}$ shared memory per work-group, and $W_{\max}$ is hardware warp-slot capacity. The testbench should vary $R_{\text{wg}}$ and $S_{\text{wg}}$ to validate allocation decisions, and inject thread divergence patterns to exercise reconvergences and barrier correctness.

Implementation below provides:
\begin{itemize}
\item a parameterized synthesizable compute kernel core that simulates a small SIMT workload (per-thread increment and a barrier);
\item a simple resource allocator modeling resident work-group counting and grant/return protocol;
\item a compact barrier module implementing arrival counting and release pulse; and
\item a testbench harness that launches multiple work-groups, toggles reset and clock for simulation, and reports completion timing.
\end{itemize}

\begin{lstlisting}[language=Verilog,caption={Compute kernel testbench harness with resource allocator and barrier},label={lst:compute_tb}]
module compute_kernel_tb;
  // simulation clock/reset
  reg clk = 0;
  reg rst_n = 0;
  always #5 clk = ~clk; // sim only

  parameter WG_SIZE = 8;
  parameter MAX_RESIDENT = 2;
  parameter NUM_WG_LAUNCH = 4;

  // simple resource allocator
  reg [$clog2(MAX_RESIDENT+1)-1:0] resident_cnt = 0;
  reg grant;
  reg release;

  // instantiate DUT (lightweight model)
  wire wg_start;
  reg [7:0] wg_id = 0;
  wire wg_done;

  // allocator FSM: grant when resident available and a launch pending
  reg launch_req = 0;
  always @(posedge clk) begin
    if (!rst_n) begin
      resident_cnt <= 0; grant <= 0; launch_req <= 0;
    end else begin
      grant <= 0;
      if (launch_req && resident_cnt < MAX_RESIDENT) begin
        grant <= 1; resident_cnt <= resident_cnt + 1;
        launch_req <= 0;
      end
      if (release) resident_cnt <= resident_cnt - 1;
    end
  end

  // simple barrier module
  reg [3:0] barrier_count = 0;
  reg barrier_release = 0;
  always @(posedge clk) begin
    if (!rst_n) begin barrier_count <= 0; barrier_release <= 0; end
    else begin
      barrier_release <= 0;
      if (wg_start) begin // simulated thread arrivals: WG_SIZE cycles
        barrier_count <= barrier_count + 1;
        if (barrier_count + 1 == WG_SIZE) begin
          barrier_release <= 1; barrier_count <= 0;
        end
      end
    end
  end

  // compute kernel core: simulates per-thread work, hits barrier once
  reg [3:0] thread_idx = 0;
  reg running = 0;
  reg done = 0;
  assign wg_done = done;
  assign wg_start = running; // single-cycle arrival per-thread

  always @(posedge clk) begin
    if (!rst_n) begin thread_idx <= 0; running <= 0; done <= 0; end
    else begin
      if (grant) begin running <= 1; thread_idx <= 0; done <= 0; end
      if (running) begin
        // simulate compute per-thread for one cycle then arrival
        if (barrier_release) begin
          // barrier passed, finish remaining threads
          if (thread_idx == WG_SIZE-1) begin running <= 0; done <= 1; end
          else thread_idx <= thread_idx + 1;
        end else begin
          // wait for barrier to accumulate arrivals
          if (thread_idx < WG_SIZE-1) thread_idx <= thread_idx + 1;
        end
      end
      if (done) release <= 1; else release <= 0;
    end
  end

  // launcher: sequence work-group launches
  integer i;
  initial begin
    #1 rst_n = 0;
    #20 rst_n = 1;
    for (i=0;i
\chapter{Chapter 12: Tensor and Matrix Acceleration}
\section{Section 1: Matrix Multiplication Fundamentals}
\subsection{Item 1:  GEMM operation principles}
Building on the compute dispatch and occupancy topics in Chapter 11, the GEMM discussion here ties those runtime concerns to the microarchitectural choices that maximize throughput on SMs and tensor cores. The following explains the canonical GEMM semantics, its computational and memory costs, and a practical tiled implementation for SIMT GPUs.

GEMM (general matrix multiply) is commonly expressed in linear-algebra libraries as
\begin{equation}[H]\label{eq:GEMM}
C \leftarrow \alpha\, A\,B + \beta\, C,
\end{equation}
where $A$ is $M\times K$, $B$ is $K\times N$, and $C$ is $M\times N$. For floating-point multiply–add (FMA) counted as two FLOPs, the total floating-point work is
\begin{equation}[H]\label{eq:flops}
\mathrm{FLOPs} = 2\,M\,N\,K.
\end{equation}
A naive implementation streams matrices from DRAM and achieves poor arithmetic intensity (FLOPs per byte). On modern GPUs, the goal is to raise arithmetic intensity by maximizing reuse in on-chip storage (registers, shared memory/LDS, and tile-level caches) and by mapping computations to specialized units (tensor cores) when precision and data layout allow.

Analysis steps:
\begin{enumerate}
\item Memory traffic model — for a blocked algorithm with block sizes $M_b\times N_b$ and inner tile $K_b$, each tile loads $M_b\times K_b$ elements of $A$ and $K_b\times N_b$ of $B$ once per accumulation into $M_b\times N_b$ outputs. Effective bytes moved per FLOP reduce with larger $K_b$ and reuse.
\item Compute/memory balance — arithmetic intensity $I$ can be approximated as
\begin{equation}[H]\label{eq:arith_intensity}
I \approx \frac{2\,M_b\,N_b\,K_b}{(M_b\,K_b + K_b\,N_b + M_b\,N_b)\times s},
\end{equation}
where $s$ is bytes per element (4 for FP32). Increasing $K_b$ increases $I$ until on-chip capacity limits reuse.
\item Mapping to GPU hardware — choose tile sizes to:
   \begin{itemize}
   \item Fill register file per warp for per-thread accumulators.
   \item Fit tiles of \lstinline|A| and \lstinline|B| into shared memory to avoid bank conflicts.
   \item Keep enough active warps for latency hiding without exceeding shared-memory or register limits.
   \end{itemize}
\end{enumerate}

Implementation: a standard tiled SIMT kernel using shared memory. This is intentionally portable FP32 code; replace with WMMA or vendor intrinsics to use tensor cores for mixed precision.

\begin{lstlisting}[language=C++,caption={Tiled FP32 GEMM kernel (CUDA) using shared memory},label={lst:gemm_cuda}]
__global__ void sgemm_tiled(int M,int N,int K,const float *A,const float *B,float *C,
                            float alpha,float beta){
  const int TILE = 32; // tile size; tune for SM register/shared limits
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y*TILE + threadIdx.y;
  int col = blockIdx.x*TILE + threadIdx.x;
  float acc = 0.0f; // per-thread accumulator in register
  for(int t=0;t<(K+TILE-1)/TILE;++t){
    int a_col = t*TILE + threadIdx.x;
    int b_row = t*TILE + threadIdx.y;
    // load tiles with bounds check
    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row*K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
    __syncthreads();
    // multiply-accumulate inner tile
    for(int k=0;k
\subsection{Item 2:  Blocking and tiling strategies}
The previous subsection established GEMM's operation count and the critical gap between compute and memory traffic; blocking and tiling directly attack that gap by trading on-chip storage to increase data reuse.  Here we analyze tiling parameters, show a practical CUDA-style blocked kernel, and derive the arithmetic-intensity benefit that drives SM and tensor-core utilization.

Problem. A naive row-column GEMM issues $\mathcal{O}(N^3)$ FLOPs while moving $\mathcal{O}(N^2)$ matrix elements repeatedly across high-latency VRAM, making the kernel memory-bound on modern GPUs with high SM throughput. Blocking reduces redundant loads by keeping small tiles resident in fast on-chip storage (registers or shared memory/LDS) so each loaded element participates in many MACs.

Analysis. For square tile size $b$ (computing a $b$-by-$b$ $C$ tile), a single $k$-step multiplies one $A$ tile and one $B$ tile, producing $\mathcal{O}(b^3)$ multiply-add operations while loading $\mathcal{O}(b^2)$ elements from each matrix once per step. If $C$ remains resident in registers across steps, the per-step arithmetic intensity (FLOPs per byte transferred) grows linearly with $b$. Formally, using element size $s$ bytes and counting one multiply-add as two FLOPs,
\begin{equation}[H]\label{eq:arith_intensity}
\text{AI}_{\text{per-step}} \approx \frac{2\,b^3}{2\,b^2 \cdot s} = \frac{b}{s}.
\end{equation}
Thus doubling tile dimension doubles AI, which improves the chance of being compute-bound and increases SM utilization and tensor-core occupancy. Practical limits on $b$ arise from shared memory capacity, register pressure, and bank conflicts.

Implementation. Below is a production-ready CUDA-style tiled GEMM kernel that demonstrates shared-memory blocking and $k$-loop tiling; it targets SM-level shared memory reuse and is amenable to tensor-core intrinsics with small changes.

\begin{lstlisting}[language=Cuda,caption={Shared-memory tiled GEMM kernel},label={lst:tiled_gemm}]
extern "C" __global__ void tiled_gemm(const float* A, const float* B, float* C,
                                      int M, int N, int K, int lda, int ldb, int ldc) {
  const int TILE = 32; // tile dimension tuned per-SM for shared mem and registers
  __shared__ float sA[TILE][TILE]; // shared memory tiles
  __shared__ float sB[TILE][TILE];
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE + ty;
  int col = bx * TILE + tx;
  float acc = 0.0f; // scalar accumulator in register
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // cooperative load with bounds check
    int aidx = row < M && (k0 + tx) < K ? row*lda + (k0 + tx) : -1;
    int bidx = (k0 + ty) < K && col < N ? (k0 + ty)*ldb + col : -1;
    sA[ty][tx] = (aidx >= 0) ? A[aidx] : 0.0f; // load A tile
    sB[ty][tx] = (bidx >= 0) ? B[bidx] : 0.0f; // load B tile
    __syncthreads();
    // compute inner product for tile
    #pragma unroll 8
    for (int t = 0; t < TILE; ++t) acc += sA[ty][t] * sB[t][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row*ldc + col] = acc; // writeback
}