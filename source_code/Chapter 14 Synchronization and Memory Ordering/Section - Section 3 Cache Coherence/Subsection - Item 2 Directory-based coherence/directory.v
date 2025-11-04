module directory #(
  parameter NUM_CU = 16,
  parameter ADDR_W = 32,
  parameter ENTRY_N = 1024
)(
  input  wire                  clk,
  input  wire                  reset,
  // request: type 0=Read,1=Write
  input  wire                  req_valid,
  input  wire [1:0]            req_type,
  input  wire [ADDR_W-1:0]     req_addr,
  input  wire [$clog2(NUM_CU)-1:0] req_cu, // requester id
  output reg                   resp_valid,
  output reg [1:0]             resp_state,
  output reg [NUM_CU-1:0]      resp_sharers
);
  // Entry arrays
  reg [ADDR_W-1:0] tag_mem [0:ENTRY_N-1];
  reg [1:0]        state_mem [0:ENTRY_N-1]; // 0=I,1=S,2=M
  reg [NUM_CU-1:0] sharer_mem [0:ENTRY_N-1];
  integer i;
  // simple direct-mapped index
  wire [$clog2(ENTRY_N)-1:0] idx = req_addr[$clog2(ENTRY_N)+5:6]; // example bits
  always @(posedge clk) begin
    if (reset) begin
      resp_valid <= 0;
      for (i=0;i
\subsection{Item 3:  Cross-core consistency}
The previous subsections described directory-based coherence mechanisms and the latency/traffic trade-offs of write-invalidate protocols; here we examine how those mechanisms must enforce cross-core consistency so that SMs observe a usable global memory order for graphics and compute workloads. This subsection frames the problem, derives the necessary ordering invariants, shows an implementation pattern used in GPU kernels, and states concrete hardware trade-offs.

Problem. Multiple SMs (or CUs) with private \lstinline|L1| caches and a unified \lstinline|L2| must present two guarantees to programmers and runtime systems:
\begin{itemize}
\item Per-address coherence: writes to the same cache line must be serialized.
\item Cross-core visibility: a write becoming visible on one SM must be visible to another SM after defined synchronization.
\end{itemize}

Analysis. Coherence supplies per-address serialization; consistency binds that per-address order into program-visible happens-before relations. Formally, coherence requires a total order over writes to each address:
\begin{equation}[H]\label{eq:coherence}
\forall x,\ \forall w_1,w_2\in\text{Writes}(x),\ w_1\neq w_2\Rightarrow (w_1\prec_{\mathrm{coh}} w_2\ \lor\ w_2\prec_{\mathrm{coh}} w_1).
\end{equation}
Acquire-release (or fence) operations connect these per-address orders across addresses: if core A performs a store S with release semantics and core B performs a load L with acquire semantics that reads S, then S happens-before L and all prior stores on A become visible to B.

Implementation pattern. GPUs implement cross-core consistency using a combination of:
\begin{enumerate}
\item Directory or directory-cache slices that track owner/dirty bits and issue invalidations to remote \lstinline|L1|s.
\item Store buffers and write-combine units at each SM; these may delay visibility until flushed or acknowledged.
\item Memory-fence instructions that force writeback and drain of local buffers, and optionally broadcast a completion (system fence).
\end{enumerate}

Typical programmer pattern for inter-SM signaling uses a release-store plus a system-wide fence on the producer and an acquire-load on the consumer. Example CUDA-like kernel showing the required sequence (producer signals with a flag; consumer waits and then uses a fence to ensure visibility):

\begin{lstlisting}[language=C,caption={Inter-SM signaling pattern using atomic flag and system fence},label={lst:signal}]
__global__ void producer(int *data, int *flag) {
  data[0] = 42;                     // store data to global memory
  __threadfence_system();           // flush write buffers to system visibility
  atomicExch(flag, 1);              // release-like signal (atomic provides ordering)
}

__global__ void consumer(int *data, int *flag) {
  while (atomicAdd(flag, 0) == 0) ; // spin until flag set (acquire)
  __threadfence_system();           // ensure subsequent loads see producer stores
  int v = data[0];                  // now guaranteed to observe 42
}