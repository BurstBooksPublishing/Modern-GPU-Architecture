module banked_smem #(
  parameter int LANES = 32,
  parameter int NBANKS = 32,           // power-of-two
  parameter int WIDTH  = 32,           // data width bits
  parameter int ADDR_W = 10            // total address bits
) (
  input  logic                   clk, rst,
  input  logic [LANES-1:0]       req_valid,
  input  logic [LANES-1:0][ADDR_W-1:0] req_addr, // per-lane addr
  input  logic [LANES-1:0]       req_we,
  input  logic [LANES-1:0][WIDTH-1:0] req_wdata,
  output logic [LANES-1:0]       resp_valid,
  output logic [LANES-1:0][WIDTH-1:0] resp_rdata
);
  localparam int BANK_BITS = $clog2(NBANKS);
  localparam int ROW_BITS  = ADDR_W - BANK_BITS;
  localparam int DEPTH_PER_BANK = (1<
\subsection{Item 2:  Bank conflict resolution}
Building on the banked layout described in the previous subsection, we now examine how simultaneous SIMT accesses map onto banks and what mechanisms, both hardware and software, resolve or avoid conflicts. The goal is to quantify serialization cost and show practical code-level mitigations that improve throughput for graphics and ML kernels.

Problem statement and analysis:
\begin{itemize}
\item In banked shared memory, each physical bank can service one access per cycle; a warp of $W$ threads issues up to $W$ concurrent accesses. When multiple threads target addresses that map to the same bank but different words, accesses serialize; if all threads target the identical word within a bank, many GPU designs provide a read broadcast (no serialization) while writes are typically serialized or merged depending on semantics.
\item Let $W$ be warp size (commonly 32) and $u$ the number of distinct banks touched by a warp. The minimum service cycles required for that access pattern is
\begin{equation}[H]\label{eq:cycles_per_warp}
\mathrm{cycles\_per\_warp} \;=\; \left\lceil \frac{W}{u} \right\rceil .
\end{equation}
This assumes single-ported banks and ignores microarchitectural broadcasts and atomic hazards. Effective per-warp bandwidth scales inversely with $\mathrm{cycles\_per\_warp}$.
\end{itemize}

Operational relevance:
\begin{itemize}
\item For matrix transpose or strided reductions common in ML and graphics (texture tile accumulation, shared-memory reductions), a stride that is a multiple of the bank count causes severe conflicts. For a bank count $B$, a stride $s$ that satisfies $s \bmod B = 0$ maps every thread in a warp to the same bank, yielding $u=1$ and $\mathrm{cycles\_per\_warp}=W$.
\end{itemize}

Implementation techniques (software and hardware):
\begin{itemize}
\item Software: change data layout or insert padding to alter bank-index computation; use vectorized loads (64-bit or 128-bit) so each thread accesses multiple banks per transaction; perform a blockwise transpose so accesses become contiguous in shared memory.
\item Hardware: increase bank count, widen bank word size (reduces bank index granularity), implement address-XOR swizzle to distribute strides across banks, or provide multiple ports/multi-issue banking. Each hardware choice trades silicon area and energy for lower conflict probability.
\end{itemize}

Practical kernel pattern: pad a 2D tile to avoid conflicts by making each row stride non-multiple of $B$. Example CUDA snippet shows padding trick; the pad constant avoids bank aliasing for common $B=32$.

\begin{lstlisting}[language=Cuda,caption={Shared-memory padding to avoid bank conflicts},label={lst:bank_pad}]
// Each row uses PAD to break bank alignment; TILE is multiple of warp size.
#define TILE 128
#define PAD 33
__global__ void tiled_load(float *g, float *out) {
  __shared__ float tile[TILE]
\subsection{Item 3: Barrier synchronization}
The previous discussion on bank interleaving and conflict avoidance set the stage for synchronization mechanisms that must respect banked shared memory semantics and per-warp arrival patterns. Barrier hardware must therefore coordinate multiple warps accessing banked regions while avoiding the stalls and serializations that bank conflicts introduce.

Barrier synchronization in an SM (streaming multiprocessor) enforces that all threads in a work-group reach a rendezvous point before any proceed. Problem: naive software barriers cause long spin-waits in shared memory that occupy register and ALU resources in a SIMT warp, increasing latency and reducing throughput. Analysis shows two common hardware approaches: a centralized counter (atomic increment per warp) and hierarchical/tree-based aggregation. A centralized counter issues one atomic increment per arriving warp; the completion condition is count$=$$N$. That yields worst-case arrival-processing work proportional to $N$, so latency scales as
\begin{equation}[H]\label{eq:central_latency}
T_{\mathrm{central}} \approx N\cdot \tau_{\mathrm{atomic}} + \tau_{\mathrm{release}},
\end{equation}
where $\tau_{\mathrm{atomic}}$ is the per-atomic processing latency and $\tau_{\mathrm{release}}$ is the broadcast latency. A tree aggregator reduces the number of sequential updates to $O(\log_2 N)$, giving
\begin{equation}[H]\label{eq:tree_latency}
T_{\mathrm{tree}} \approx \tau_{\mathrm{setup}} + \tau_{\mathrm{node}}\log_2 N + \tau_{\mathrm{broadcast}}.
\end{equation}
Operationally, the tree design trades additional hardware (local buffers and small aggregation nodes) for lower barrier completion latency under high contention.

Implementation: the SM typically implements a block-local barrier unit that:
\begin{itemize}
\item accepts per-warp single-cycle \lstinline|arrive| pulses (one bit per warp),
\item maintains a per-epoch bitmask to avoid double-counting if a warp stalls,
\item issues a single-cycle \lstinline|release| pulse and advances an epoch counter so subsequent barriers reuse the same unit safely,
\item integrates with the shared memory coherence path so that any buffered writes are visible before release (i.e., a store-release to L1 or explicit fence).
\end{itemize}

Below is a synthesizable Verilog implementation of a centralized barrier controller intended for block-level use inside an SM. It assumes arriving warps assert a one-cycle pulse on \lstinline|arrive|; the module emits a one-cycle \lstinline|release| and an epoch tag for reuse.

\begin{lstlisting}[language=Verilog,caption={Centralized barrier controller (synthesizable)},label={lst:barrier_verilog}]
module barrier #(
  parameter NUM_WARPS = 32,
  parameter EPOCH_W = 8
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire [NUM_WARPS-1:0]     arrive,    // one-hot pulses from warps
  output reg                      release,   // one-cycle release pulse
  output reg [EPOCH_W-1:0]        epoch      // epoch tag for reuse
);
  // per-warp latch prevents double-counting within same epoch
  reg [NUM_WARPS-1:0] counted;
  reg [$clog2(NUM_WARPS+1)-1:0] count;

  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      counted <= {NUM_WARPS{1'b0}};
      count   <= 0;
      epoch   <= 0;
      release <= 1'b0;
    end else begin
      release <= 1'b0; // default
      // capture new arrivals and update count
      for (i=0;i
\subsection{Item 4:  Common programming patterns}
The previous discussion showed how barriers provide deterministic rendezvous and how bank remapping or padding removes intra-SMEM contention; these behaviors determine which programming patterns are practical and which require microarchitectural workarounds. Below we analyze common patterns, show their cost model, and present an implementation example that combines tiling, padding, and explicit synchronization.

Practical patterns and why they matter
\begin{itemize}
\item Tiling (blocked algorithms): Partition large problems into tiles that fit an SM's shared memory to maximize data reuse between threads on the same SM. Operational relevance: tiles reduce DRAM traffic and increase arithmetic intensity for TMUs, tensor cores, or FPUs.
\item Ping-pong buffering: Use two SMEM buffers to overlap loads from global memory with compute; useful when a load latency can be hidden by compute on the current tile.
\item Warp-aggregated atomics and warp shuffle: Use warp-level primitives to reduce contention on atomic units or to exchange values without touching SMEM, reducing ROP/atomic pipeline pressure.
\item In-place transpose and coalesced staging: Stage data into SMEM, perform a transpose with attention to bank mapping, then write back to global memory to achieve coalesced writes.
\item Reductions and scans: Use tree or warp-level hierarchies, combining per-thread registers, warp shuffles, and SMEM at block-level barriers for scalable reductions.
\end{itemize}

Bank indexing and throughput model
Accesses map to banks by thread lane and element size; for a word size $w$ bytes and $NB$ banks, bank index
\begin{equation}[H]\label{eq:bank}
\text{bank} = \left\lfloor\frac{\text{addr}}{w}\right\rfloor \bmod NB.
\end{equation}
If $k$ warps concurrently target the same bank, the effective per-access bandwidth for that bank scales as $B_{\text{bank}}/k$. For a tile of size $T$ and element size $w$, padding one element per row breaks stride-$NB$ aliases and reduces $k$, trading a small SMEM capacity for multiple-times better effective bandwidth.

Occupancy and resource trade-off
Let $S_{\text{block}}$ be shared memory per block and $S_{\text{SM}}$ the SM total. The max resident blocks per SM is
\begin{equation}[H]\label{eq:blocks}
B_{\max} = \left\lfloor\frac{S_{\text{SM}}}{S_{\text{block}}}\right\rfloor,
\end{equation}
which constrains latency hiding. Increasing $S_{\text{block}}$ (for larger tiles or padding) improves reuse but may reduce $B_{\max}$ and hurt SM-level latency hiding. Optimize $S_{\text{block}}$ for the target workload's arithmetic intensity.

Example: tiled GEMM kernel with SMEM padding and ping-pong buffering
\begin{lstlisting}[language=Cuda,caption={Tiled GEMM with SMEM padding and double buffering},label={lst:tile_gemm}]
__global__ void tiled_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  const int TILE = 32;
  __shared__ float sA[TILE][TILE];      // tile for A
  __shared__ float sB[TILE][TILE+1];    // pad +1 to avoid bank conflict
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t=0; t < (K+TILE-1)/TILE; ++t) {
    // load tile A and B (coalesced loads); bounds-checked
    int a_col = t*TILE + threadIdx.x;
    sA[threadIdx.y][threadIdx.x] =
      (row < M && a_col < K) ? A[row*K + a_col] : 0.0f;
    int b_row = t*TILE + threadIdx.y;
    sB[threadIdx.y][threadIdx.x] =
      (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
    __syncthreads(); // ensure tile is ready
    // compute inner product on tile
    for (int k=0; k
\section{Section 4: L1 Cache Design}
\subsection{Item 1:  Set-associative structure}
Building on the access-coalescing and banked-shared-memory behaviors discussed earlier, this subsection examines how a set-associative layout for the per-SM L1 cache maps memory streams from warps into physical lines and how that mapping affects SIMT throughput and latency. The analysis below connects associativity, indexing, and replacement to real GPU workload patterns such as strided tensor accesses and texture fetches.

A set-associative cache divides capacity into $A$ ways and $S$ sets; a request is indexed into one set and may occupy any of the $A$ ways in that set. For a cache with total size $C$ bytes and line size $L$ bytes:
\begin{equation}[H]\label{eq:sets}
S \;=\; \frac{C}{L\cdot A},\qquad
\mathrm{index\ bits} \;=\; \log_2 S.
\end{equation}
For example, a 64 KiB L1 with 128 B lines and 4-way associativity yields $S=512$ sets and 9 index bits. The index function is frequently the low-order bits of the block address, which makes predictable stride patterns vulnerable to set conflicts; a stride that is a multiple of $S\cdot L$ will map many addresses to the same set and can cause thrashing under SIMT-aligned accesses.

Design and analysis points:
\begin{itemize}
\item Conflict misses vs. access latency: increasing $A$ reduces conflict misses but raises tag storage, comparator fanout, and hit latency. For SMs running high-occupancy kernels, reducing misses improves arithmetic unit utilization; for latency-sensitive texture accesses through the TMU, minimal per-access latency is crucial.
\item Banking and parallelism: implement data arrays as multiple banks to sustain multiple concurrent cache-line fills from different warps. Bank indexing must avoid aliasing with set index to reduce contention at the physical data ports.
\item Replacement: true LRU yields the best miss-rate but scales poorly in area for large $A$. Practical designs use pseudo-LRU, tree-PLRU, or lightweight per-set round-robin to bound complexity.
\item Skewing and hashing: simple XOR of tag and index bits or hashed indexing mitigates pathological strides at the cost of slightly more complex address computation.
\item Interaction with higher-level caches: L1 design choices impact L2 pressure and coherence messaging frequency; a higher miss rate increases transactional load on the L2 crossbar and memory controller.
\end{itemize}

Implementation example: a synthesizable Verilog tag array with per-set round-robin replacement. The module is parameterized, supports synchronous tag read/update, and exposes a hit vector and selected way for eviction.

\begin{lstlisting}[language=Verilog,caption={Parameterized set-associative tag array with per-set round-robin replacement (synthesizable).},label={lst:setassoc_tag}]
module setassoc_tag_array #(
  parameter TAG_W = 20,             // tag width
  parameter LINE_W = 7,             // log2(line bytes)
  parameter WAYS = 4,
  parameter SETS = 512,
  parameter IDX_W = $clog2(SETS)
) (
  input  clk,
  input  rst,
  input  [IDX_W-1:0] index,         // set index
  input  [TAG_W-1:0] tag_in,        // tag from address
  input  rd,                        // read request
  input  wr,                        // write-on-allocate (store tag)
  output reg hit,
  output reg [$clog2(WAYS)-1:0] way_hit,
  output reg [$clog2(WAYS)-1:0] evict_way
);
  // tag storage: [way][set]
  reg [TAG_W-1:0] tag_mem [0:WAYS-1][0:SETS-1];
  reg valid     [0:WAYS-1][0:SETS-1];
  reg [$clog2(WAYS)-1:0] rr_ptr [0:SETS-1]; // per-set round-robin pointer

  integer w;
  always @(posedge clk) begin
    if (rst) begin
      for (w=0; w
\subsection{Item 2:  Write-back vs write-through}
Building on the set-associative indexing and tag management just discussed, we now examine how L1 write policies change traffic, coherence complexity, and latency hiding across SMs and CUs. The choice between write-back and write-through interacts with SIMT coalescing, store queues, and the L2/lower-level coherence domain, so quantify and implement accordingly.

Write-back stores modify cached lines and mark them dirty; lower levels are updated only on eviction. This minimizes off-chip write bandwidth for temporal reuse but requires:
\begin{itemize}
\item a dirty-bit per line and eviction logic to issue writebacks,
\item mechanisms to flush or drain dirty lines on context switch, page migration, or preemption,
\item coherence protocol support (write-invalidate or directory) to maintain cross-SM visibility.
\end{itemize}

Write-through immediately forwards writes to L2/DRAM while optionally updating L1. It simplifies coherence and enables simpler store ordering but increases bandwidth and can saturate memory channels under heavy store-streaming workloads, typical in some GPGPU kernels and compute-bound ML training phases.

Quantify effect with an expected off-chip write traffic model. Let $r_w$ be the fraction of memory operations that are stores, $h$ the L1 hit rate, and $p_d$ the probability a hit marks the line dirty. For write-back with write-allocate, expected off-chip writeback bandwidth per access approximates
\begin{equation}[H]\label{eq:wb_traffic}
B_{\mathrm{wb}} \approx r_w \cdot (1-h)\cdot S_{\mathrm{line}}\cdot p_{\mathrm{evict}},
\end{equation}
where $S_{\mathrm{line}}$ is cache line size and $p_{\mathrm{evict}}$ is eviction probability conditioned on miss. For write-through:
\begin{equation}[H]\label{eq:wt_traffic}
B_{\mathrm{wt}} \approx r_w \cdot S_{\mathrm{data}} + r_w\cdot h\cdot S_{\mathrm{data}},
\end{equation}
counting both immediate forwarded words $S_{\mathrm{data}}$ and cached updates if written. These show write-back reduces off-chip write volume when $h$ and temporal reuse are high, a common case for compute kernels that reuse working sets in shared memory and L1.

Implementation concerns translate into RTL and physical blocks:
\begin{itemize}
\item store coalescer and write buffer sizing must match memory controller latency to avoid stalls;
\item eviction paths require prioritized streaming to DRAM to avoid backpressure into the replacement pipeline;
\item atomic and ordered stores typically need serialization points; write-through simplifies atomic visibility.
\end{itemize}

Example synthesizable Verilog: a compact write-policy controller managing a dirty bit array and mem write issuance.

\begin{lstlisting}[language=Verilog,caption={L1 write-policy controller (parameterizable)},label={lst:wb_ctrl}]
module wb_wt_ctrl #(
  parameter INDEX_BITS = 6,
  parameter LINES = 1<
\subsection{Item 3:  Coherency mechanisms}
The previous discussion of set-associative layout and write-back policy established how line placement and dirty eviction change conflict behavior and off-chip traffic. Those choices directly shape which coherency mechanism is practical: snoop-based invalidation becomes expensive with many SMs and deep caches, while directory schemes trade area for scalability.

Coherency problem: multiple streaming multiprocessors (SMs) each with an L1 that may cache overlapping addresses must present a consistent view for shared-memory compute and graphics tasks, while sustaining SIMT throughput. Key constraints are:
\begin{itemize}
\item High concurrent request rate from many warps.
\item Wide memory transactions and bursty temporal locality (textures, framebuffers, activation tensors).
\item Weak GPU memory ordering where explicit fences and atomics bound visibility.
\end{itemize}

Architectural options and their operational effects:
\begin{enumerate}
\item Snooping (bus/broadcast): simple invalidation or update across SMs; low latency for few SMs but scales poorly because broadcast bandwidth grows $\mathrm{O}(N_{\mathrm{SM}})$.
\item Directory-based: per-line metadata records sharers or an owner. Scales to many SMs at the cost of storage and directory lookup latency.
\item Hybrid: small snoop domain inside a die cluster and directory across clusters.
\end{enumerate}

Coherence actions are usually write-invalidate for GPUs because write-update multiplies write bandwidth. Valid states follow MESI-style encodings, but many GPU designs simplify to:
\begin{itemize}
\item I (Invalid), S (Shared, read-only), M (Modified owner), O (Owned/shared with data in directory).
\end{itemize}
Write-back L1 implies dirty lines may be held in L1 until eviction, requiring directory tracking of owners.

Quantifying directory cost: with $S$ sets, $W$ ways, and $N$ SMs, a simple per-line sharer bitmask requires $N$ bits. Directory storage in bits is approximately
\begin{equation}[H]\label{eq:dir_cost}
B \approx S\cdot W \cdot N
\end{equation}
Additional tag and state bits increase the overhead; compressing sharer sets (coarse-grain directories, bloom filters) reduces $B$ at some false-positive cost.

Implementation sketch: a synthesizable per-line directory entry FSM that tracks tag, sharer mask, and owner. The module below is a minimal, production-style block suitable as a tile in a larger directory array.

\begin{lstlisting}[language=Verilog,caption={Per-line directory entry (parameterizable)},label={lst:dir_entry}]
module dir_entry #(
  parameter TAG_WIDTH = 20,
  parameter NUM_SMS = 8
)(
  input  wire                  clk,
  input  wire                  rst,
  input  wire                  req_v,       // request valid
  input  wire                  req_is_write,// 1=write,0=read
  input  wire [TAG_WIDTH-1:0]  req_tag,
  input  wire [$clog2(NUM_SMS)-1:0] req_sm,  // requester id
  output reg                   hit,
  output reg                   need_invalidate, // assert to invoke invalidates
  output reg [NUM_SMS-1:0]     sharers_mask
);
  // state: 0 = I, 1 = S, 2 = M (owner)
  reg [1:0] state;
  reg [TAG_WIDTH-1:0] tag;
  reg [$clog2(NUM_SMS)-1:0] owner;

  always @(posedge clk) begin
    if (rst) begin
      state <= 2'd0; tag <= {TAG_WIDTH{1'b0}}; sharers_mask <= {NUM_SMS{1'b0}};
      hit <= 1'b0; need_invalidate <= 1'b0; owner <= {($clog2(NUM_SMS)){1'b0}};
    end else begin
      hit <= 1'b0; need_invalidate <= 1'b0;
      if (req_v && req_tag == tag && state != 2'd0) begin
        hit <= 1'b1;
        if (!req_is_write) begin
          // read hit: add sharer
          sharers_mask[req_sm] <= 1'b1;
        end else begin
          // write hit: if single sharer and same SM, upgrade; else invalidate others
          if (state == 2'd2 && owner == req_sm) begin
            // already owner, keep M
            ; 
          end else begin
            need_invalidate <= 1'b1;
            owner <= req_sm;
            sharers_mask <= {NUM_SMS{1'b0}}; // will be rebuilt after invalidates
            state <= 2'd2; // M
          end
        end
      end
      // on miss, policy handled externally: allocate and set tag/state accordingly
    end
  end
endmodule