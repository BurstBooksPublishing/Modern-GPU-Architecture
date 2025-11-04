module warp_scheduler #(
  parameter NUM_WARPS = 32,
  parameter WARP_WIDTH = 32,
  parameter ID_W = $clog2(NUM_WARPS)
)(
  input  wire                   clk,
  input  wire                   rst,
  input  wire [NUM_WARPS-1:0]   warp_ready,    // ready from scoreboard
  input  wire [NUM_WARPS-1:0]   warp_any_active,// true if any lane active
  input  wire [NUM_WARPS*WARP_WIDTH-1:0] warp_mask_flat, // concatenated lane masks
  output reg  [ID_W-1:0]        grant_id,
  output reg                    grant_valid,
  output reg  [WARP_WIDTH-1:0]  grant_mask
);
  reg [ID_W-1:0] rr_ptr;
  integer i;
  reg [NUM_WARPS-1:0] candidate;
  always @(*) begin
    candidate = warp_ready & warp_any_active; // choose warps with live lanes and ready
    grant_valid = 1'b0;
    grant_id = {ID_W{1'b0}};
    // rotate-search first-one after rr_ptr
    for (i=0;i
\section{Section 3: Modern GPU Overview}
\subsection{Item 1:  NVIDIA SM, AMD CU, Intel Xe-core comparison}
Building on the SIMT scheduling and divergence-handling concepts introduced earlier, we now compare how three mainstream designs map threads, hide latency, and expose specialized hardware for graphics and compute workloads. The goal is to show which microarchitectural choices drive ML throughput, raster/texture performance, and compute efficiency.

Problem: given a kernel with a given arithmetic intensity, how do SM, CU, and Xe-core differ in extracting throughput and hiding memory latency? Analysis focuses on three vectors: thread grouping and scheduling, execution pipelines and special units, and memory/register resource partitioning.

\begin{itemize}
\item Thread grouping and scheduling:
  \begin{itemize}
  \item NVIDIA groups threads into 32-thread warps; SMs implement multiple warp schedulers and fine-grained preemption, which favors frequent context switches to hide global-memory latency.
  \item AMD CUs traditionally used 64-thread wavefronts, with RDNA introducing more flexible 32/64 wave sizes and SIMD lanes to reduce divergence penalties for graphics workloads.
  \item Intel Xe organizes execution into EUs grouped in subslices; scheduling targets fine-grained SIMD lanes and often emphasizes wide vectorization and instruction-level efficiency for media and compute.
  \end{itemize}
\item Execution pipeline and specialized units:
  \begin{itemize}
  \item NVIDIA SMs integrate scalar/FP pipelines plus tensor cores (systolic-like matrix MAC units) and dedicated RT cores for bounding-volume traversal and intersection acceleration.
  \item AMD CUs pair SIMD ALUs with scalar schedulers and, in RDNA2+, per-CU ray accelerators; AMD emphasizes flexible ALU resource sharing for variable-rate shading and compute.
  \item Intel Xe-core designs expose vector pipelines and, on server products, matrix engines (XMX) and ray tracing blocks; pipeline balance tends toward energy-efficient vector throughput for media.
  \end{itemize}
\item Resources, occupancy, and performance modeling:
  \begin{itemize}
  \item Peak floating-point throughput can be estimated by
    \begin{equation}[H]\label{eq:peak}
      \text{Peak FLOPS} = N_{\text{units}} \times W_{\text{SIMD}} \times O_{\text{FMA}} \times f_{\text{core}},
    \end{equation}
    where $N_{\text{units}}$ is number of execution subunits, $W_{\text{SIMD}}$ is SIMD width, $O_{\text{FMA}}$ is FMA ops per lane per cycle, and $f_{\text{core}}$ is frequency.
  \item Practical throughput is constrained by register file size $R$, shared memory $S$, and maximum resident thread groups $G$. Occupancy falls if $R$ or $S$ allocated per-thread exceed limits.
  \end{itemize}
\end{itemize}

Implementation example: a tiny CUDA occupancy query and kernel used to measure warp-limited latency hiding.

\begin{lstlisting}[language=CUDA,caption={Simple CUDA kernel and occupancy query (microbenchmark)},label={lst:cuda_occ}]
#include 
__global__ void saxpy(float a, float *x, float *y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i>>(2.0f,d_x,d_y,N);
  cudaDeviceSynchronize();
}