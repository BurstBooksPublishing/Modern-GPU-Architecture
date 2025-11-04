module rr_arb #(
  parameter NUM_WARPS = 16
) (
  input  wire                    clk,
  input  wire                    rst,
  input  wire [NUM_WARPS-1:0]    ready_mask, // 1 means warp ready
  output reg  [NUM_WARPS-1:0]    grant       // one-hot grant
);
  reg [$clog2(NUM_WARPS)-1:0] ptr; // pointer for RR fairness
  integer i;
  wire [NUM_WARPS-1:0] masked = (ready_mask << ptr) | (ready_mask >> (NUM_WARPS - ptr));
  always @(*) begin
    grant = {NUM_WARPS{1'b0}};
    for (i=0;i
\subsection{Item 3:  Compute and graphics pipeline integration}
Building on the SM/CU functional breakdown and vendor comparison that highlighted differing resource partitioning and scheduler policies, we now examine how a single silicon fabric supports both graphics pipelines (raster, texturing, ROPs, RT cores) and general-purpose compute (SIMT kernels, tensor cores) without sacrificing throughput or latency for either domain.

The engineering problem is to multiplex heterogeneous workloads across shared SM/CU resources while keeping per-workload QoS acceptable. Analysis begins with a throughput bottleneck model: overall frame or kernel progress is bounded by the slowest critical resource, so for a mixed workload one can express sustained throughput as
\begin{equation}[H]\label{eq:throughput_balance}
T_{\mathrm{sys}} = \min\{T_{\mathrm{SM}},\, T_{\mathrm{TMU}},\, T_{\mathrm{ROP}},\, T_{\mathrm{MEM}}\},
\end{equation}
where $T_{\mathrm{SM}}$ is aggregate shader core compute throughput (including tensor cores), $T_{\mathrm{TMU}}$ texture fetch throughput, $T_{\mathrm{ROP}}$ raster output throughput, and $T_{\mathrm{MEM}}$ memory subsystem throughput. This operational equation drives design decisions: if texture traffic saturates $T_{\mathrm{MEM}}$ during complex shading, compute throughput must be throttled or partitioned.

Implementation patterns that emerge in modern GPUs:
\begin{itemize}
\item Unified front-end and command processor that accepts graphics command streams and compute dispatch packets, translating both into work-items (warps/wavefronts) queued to per-SM front-end buffers.
\item Hardware work distribution with separate priority queues: graphics tends to favor low-latency dispatch for fragments, compute favors high-throughput bulk execution. Arbitration is performed at dispatch to enforce fairness and latency budgets.
\item Resource accounting per-SM: registers, shared memory, and active warp slots are reserved dynamically; allocator rejects or stalls new work when occupancy thresholds would violate real-time constraints (e.g., present frame deadlines).
\item Cache and memory partitioning: L1/L2 QoS or way-partitioning prevents a heavy compute kernel from evicting texture working sets; TLB and pagewalk acceleration are shared with per-context quotas.
\item Fast-paths for raster: early-Z, hierarchical Z, and tile caches reduce memory pressure, enabling coexistence with compute without linear degradation.
\end{itemize}

A simple synthesizable arbitration example that grants SM dispatch slots between graphics and compute queues is shown below.

\begin{lstlisting}[language=Verilog,caption={Round-robin arbiter between graphics and compute dispatch requests},label={lst:arbiter}]
module gfx_compute_arbiter #(
  parameter QUEUES = 2 // 0:graphics, 1:compute
)(
  input  wire clk,
  input  wire rst_n,
  input  wire [QUEUES-1:0] req,   // request vectors
  output reg  [QUEUES-1:0] grant, // one-hot grant
  output reg  valid               // grant valid
);
  reg [$clog2(QUEUES)-1:0] ptr; // round-robin pointer

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptr <= 0;
      grant <= 0;
      valid <= 0;
    end else begin
      grant <= 0;
      valid <= 0;
      // simple round-robin search
      integer i;
      for (i=0; i
\subsection{Item 4:  Role of schedulers and dispatch units}
The preceding discussion of compute/graphics pipeline integration and the SM microarchitecture set the stage for understanding schedulers and dispatch units as the glue that maps queued work onto functional units inside an SM. Schedulers translate resource availability, dependency state, and QoS policies into issued warp-level instructions that feed ALUs, SFUs, tensor cores, TMUs, and ROP pipelines.

Problem: modern GPUs need to sustain thousands of threads per device while hiding long DRAM and cache latencies, yet they must also coordinate heterogeneous execution units and contention for register file ports and shared memory banks. Analysis: a scheduler's primary responsibilities are:
\begin{itemize}
\item select ready warps/wavefronts while enforcing scoreboarding dependencies and active-mask state;
\item arbitrate issued micro-ops to execution units respecting port width and functional-unit affinity;
\item implement fairness, priority, or QoS policies to avoid starvation and meet latency targets.
\end{itemize}
A simple capacity formula relates memory latency to the minimum active warps required to hide stalls:
\begin{equation}[H]\label{eq:occupancy}
N_{\mathrm{min}} \;=\; \left\lceil \frac{L_{\mathrm{mem}}}{I_{\mathrm{issue}}} \right\rceil
\end{equation}
where $L_{\mathrm{mem}}$ is average memory-access latency in cycles and $I_{\mathrm{issue}}$ is the average number of warp-issue opportunities per cycle.

Implementation: real SMs combine per-SM warp schedulers with a global dispatch and per-function-unit arbiters. Common techniques:
\begin{enumerate}
\item Round-robin or age-based warp selection to distribute issue opportunities and reduce head-of-line blocking.
\item Scoreboarding or CAM-style dependency tracking to mark registers and memory results busy until retire or writeback.
\item Multi-issue dispatch that maps scalar or vector micro-ops to ALU, SFU, or tensor core lanes with crossbars and backpressure signals.
\end{enumerate}

A compact, synthesizable Verilog example implements a rotating priority (round-robin) warp scheduler that takes a ready vector and grants one warp per cycle. It demonstrates the rotating pointer, mask rotation, and grant generation used in commercial SMs.

\begin{lstlisting}[language=Verilog,caption={Simple rotating-priority warp scheduler (synthesizable).},label={lst:warp_sched}]
module warp_scheduler
  #(parameter WARPS=16, LG_WARPS=$clog2(WARPS))
  (input  wire                 clk,
   input  wire                 rst_n,
   input  wire [WARPS-1:0]     ready_mask, // per-warp ready
   output reg  [WARPS-1:0]     grant,      // one-hot grant
   output reg  [LG_WARPS-1:0]  grant_id,
   output reg                  grant_valid);
  // rotating pointer
  reg [LG_WARPS-1:0] ptr;
  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptr <= 0;
      grant <= 0;
      grant_id <= 0;
      grant_valid <= 0;
    end else begin
      // rotate ready vector so pointer is MSB-aligned search
      reg [WARPS-1:0] rot;
      rot = {ready_mask[WARPS-1 -: WARPS]}; // rotation helper
      // scan for first set bit after pointer
      grant = 0;
      grant_valid = 0;
      for (i=0; i
\section{Section 4: Memory Hierarchy Snapshot}
\subsection{Item 1:  Register files and local storage}
Building on the SIMT execution model and the SM/CU internal organization discussed earlier, we now quantify how per-thread register allocation and local storage shape occupancy, latency hiding, and spill behavior inside a GPU core.

Problem: a finite per-SM register file must be partitioned among active threads; excessive per-thread register usage (register pressure) reduces active warps and can force spilling to slower local memory or L2/VRAM. Analysis begins with occupancy math that directly maps hardware capacity to software allocation. Let $R_{\mathrm{SM}}$ be total architected registers per SM, $r$ the registers allocated per thread, and $W$ the warp size (e.g., 32). The maximum resident warps per SM is
\begin{equation}[H]\label{eq:occupancy}
N_{\text{warps,max}}=\left\lfloor\frac{R_{\mathrm{SM}}}{r\cdot W}\right\rfloor,
\end{equation}
so the maximum resident threads equals $N_{\text{warps,max}}\cdot W$. Example: $R_{\mathrm{SM}}=65{,}536$, $r=64$, $W=32$ gives $N_{\text{warps,max}}=\lfloor65{,}536/(64\cdot32)\rfloor=32$ warps (1024 threads).

Implementation-level details that affect performance:
\begin{itemize}
\item Banked register files provide effective multi-porting without quadratic area cost. Banks $B$ reduce simultaneous-access contention; accesses from a warp to the same bank serialize and cost cycles.
\item Register renaming and instruction scoreboarding allow out-of-order write visibility within an SM while preserving SIMT semantics; however, latency of long-latency ops (memory, SFU) still requires many resident warps to hide stalls.
\item Spilling occurs when compiler/allocation exceeds available register file. Spilled variables are spilled to local memory which is typically backed by L1/L2/VRAM and can impose latencies from tens to hundreds of cycles depending on cache hits.
\end{itemize}

A practical kernel example that intentionally raises register pressure (and likely causes spilling) is shown below; profiling such a kernel with compiler flags reveals the register count and its impact on occupancy.

\begin{lstlisting}[language=Cuda,caption={CUDA kernel that increases per-thread register usage to illustrate spilling and occupancy effects},label={lst:spill_example}]
__global__ void pressure_kernel(float *out, const float *in, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Big local array likely forces spill (compiler may place on registers or local memory).
  float tmp[16]; // per-thread local array -> increases register usage / local memory.
  float acc = 0.0f;
  for (int i=0;i<16;i++) tmp[i] = in[idx+i]; // compute uses many temporaries.
  for (int i=0;i<16;i++) acc += tmp[i];
  out[idx] = acc;
}
// Host: compile with -Xptxas=-v to see register usage; use occupancy API to see active warps.