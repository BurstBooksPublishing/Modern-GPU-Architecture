module drr_arbiter #(
  parameter int N = 8,                   // number of SM requesters
  parameter int QW = 8                   // quantum width
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic [N-1:0]         req,      // one-hot active requests
  input  logic [QW-1:0]        quantum [N],// per-client quantum (weight)
  output logic [N-1:0]         grant     // one-hot grant
);
  // per-client deficit counters
  logic [QW+8-1:0] deficit [N]; // extra bits for accumulation
  int ptr; // rotation pointer

  // reset
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptr <= 0;
      for (int i=0;i
\subsection{Item 3:  Virtual channels}
The previous discussion characterized SM-to-memory partition links and bandwidth scheduling policies that shape per-partition demand; virtual channels (VCs) are the complementary mechanism inside the crossbar that multiplex logical flows over physical links while enabling QoS, deadlock avoidance, and fine-grained arbitration across SMs, TMUs, ROPs and memory controllers.

Virtual channels (VCs) are lightweight logical lanes multiplexed over a single physical link. They separate competing request classes (e.g., high-priority read responses, writebacks, texture fetches) into independent FIFO buffers and arbitration domains. Problem: without VCs, head-of-line (HOL) blocking by a long transaction stalls unrelated flows, reducing SM throughput and increasing warp stall time. Analysis shows that VCs mitigate HOL blocking and enable credit-based flow control and virtual-channel allocation to avoid cyclic resource dependencies that cause deadlock.

Key operational elements:
\begin{itemize}
\item Per-input buffering: each ingress port maintains $N_{\mathrm{vc}}$ separate FIFOs; a packet is tagged with \lstinline|vc_id| at allocation time.
\item Allocation and arbitration: VC allocation occurs at packet enqueue and crossbar arbitration chooses a granted pair (input VC, output port) based on QoS weights.
\item Flow control: credit counters per VC prevent buffer overflow downstream; credits equal free entries.
\item Deadlock avoidance: assign VCs so that at least one escape VC follows a strict acyclic ordering, or use virtual-channel ordering enforced by the allocator.
\end{itemize}

Sizing and latency: buffer depth must absorb link round-trip-time (RTT) and burstiness. For a link of raw bandwidth $B$ (bytes/s), flit size $F$ (bytes), and round-trip latency $T_{\mathrm{rt}}$ (s), per-VC minimum free-slots D to avoid backpressure-induced stalls satisfies:
\begin{equation}[H]\label{eq:vc_depth}
D \;\ge\; \left\lceil \frac{B \cdot T_{\mathrm{rt}}}{F \cdot N_{\mathrm{vc}}^{\text{active}}} \right\rceil
\end{equation}
where $N_{\mathrm{vc}}^{\text{active}}$ is the number of concurrently active VCs sharing the link. Equation (1) shows depth grows with bandwidth and RTT; smaller flits reduce required depth but increase control overhead.

Implementation sketch: a synthesizable Verilog module below provides parameterized per-VC FIFOs with credit return and full/empty flags. It is intentionally minimal but synthesizable and suitable for integration into a crossbar input stage.

\begin{lstlisting}[language=Verilog,caption={VC FIFO array with credit interface},label={lst:vcfifo}]
module vc_fifo_array #(parameter N_VC=4, DEPTH=8, WIDTH=64, PTR_W=$clog2(DEPTH))
(
  input  wire                   clk,
  input  wire                   rst,
  input  wire [N_VC-1:0]        push,        // push per VC (one-hot allowed)
  input  wire [N_VC-1:0]        pop,         // pop per VC
  input  wire [N_VC-1:0]        push_valid,  // valid qualifier per VC
  input  wire [WIDTH-1:0]       din [N_VC-1:0], // data per VC (bundled)
  output reg  [N_VC-1:0]        full,
  output reg  [N_VC-1:0]        empty,
  output wire [WIDTH-1:0]       dout [N_VC-1:0], // output per VC
  output reg  [PTR_W:0]         credits [N_VC-1:0] // free slots per VC
);
  // Per-VC RAMs and pointers
  reg [WIDTH-1:0] mem [0:N_VC-1][0:DEPTH-1];
  reg [PTR_W-1:0] wptr [0:N_VC-1];
  reg [PTR_W-1:0] rptr [0:N_VC-1];
  integer i;
  // Initialization
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i
\subsection{Item 4:  QoS enforcement}
The previous discussion established how virtual channels isolate traffic classes and how bandwidth schedulers map those classes onto memory partitions; QoS enforcement is the mechanism that turns those policies into concrete, time- and buffer-bounded behaviour at the crossbar. Here we analyze enforcement objectives, derive simple bounding math, present an implementable scheduler primitive, and state practical trade-offs.

Problem statement: multiple SMs, TMUs, and RT/tensor engines generate heterogeneous flows that share limited crossbar and DRAM bandwidth. Goals are to (1) guarantee minimum throughput for latency-sensitive flows, (2) prevent starvation of low-priority classes, and (3) bound worst-case queuing delay for real-time consumers such as display and RT cores.

Analysis proceeds from two building blocks commonly implemented in silicon:
\begin{itemize}
\item rate reservation (token-bucket or credit-based), which limits average service to a configured rate; and
\item deficit/weight-based arbitration (WRR/DRR), which enforces proportional shares while avoiding head-of-line blocking.
\end{itemize}

If class $i$ is assigned weight $w_i$ and the crossbar can sustain aggregate service $B_{\text{total}}$, the long-term bandwidth guarantee is
\begin{equation}[H]\label{eq:share}
\text{share}_i \;=\; \frac{w_i}{\sum_j w_j}\;B_{\text{total}}.
\end{equation}
For token-bucket enforcement, let $\text{tokens}_i(t)$ be tokens at time $t$, refill rate $r_i$, and bucket depth $B$. The discrete update per clock $T_{\text{clk}}$ is
\begin{equation}[H]\label{eq:token}
\text{tokens}_i[t+1] = \min\!\big(B,\;\text{tokens}_i[t] + r_i T_{\text{clk}} - s_i[t]\big),
\end{equation}
where $s_i[t]$ is service consumed. Equation (2) yields a maximum burst allowance equal to $B$, which directly bounds worst-case queuing if the recipient can drain at link rate.

Implementation: a hybrid design uses per-class token buckets at the aggregator inputs, plus a central WRR arbiter that consults token availability and virtual-channel occupancy. The following synthesizable SystemVerilog module implements a small, hardware-friendly WRR QoS arbiter. It maintains deficit counters and advances a round-robin pointer to avoid starvation. We assume unit-service quanta per cycle; weights scale quanta added each round.

\begin{lstlisting}[language=Verilog,caption={Synthesizable WRR QoS arbiter (SystemVerilog) for N classes.},label={lst:wrr_qos}]
module wrr_qos #(parameter int CLASSES=4, W=8) (
  input  logic                 clk, rst,
  input  logic [CLASSES-1:0]   req,                // request per class
  input  logic [W-1:0]         weight [CLASSES],   // configured weights
  output logic [CLASSES-1:0]   grant               // one-hot grant
);
  logic [W+8-1:0] deficit [CLASSES];               // accumulate quanta
  int unsigned ptr;

  always_ff @(posedge clk) begin
    if (rst) begin
      ptr <= 0;
      grant <= '0;
      for (int i=0;i
\subsection{Item 4:  Error handling}
The previous subsections showed how DMA engines create high-throughput, ordered PCIe TLP streams and how peer-to-peer mappings can bypass OS-side error containment. Building on those ideas, this subsection treats detection, classification, and hardware response for PCIe errors that affect GPU DMA and P2P transfers.

Problem statement and analysis: undetected or late-detected errors on PCIe links cause data corruption, silent poisoning of GPU memory, or long stalls when DMA descriptors time out. Errors fall into three operational classes:
\begin{itemize}
\item Correctable link/endpoint errors (recoverable via resynchronization).
\item Non-fatal transaction errors (ECRC or completion timeouts; require retry or completion handling).
\item Fatal errors (link down, device malfunction; require isolation).
\end{itemize}

Quantitative criterion for retry budgeting comes from the transfer error probability. For a transfer of $N$ bits and bit error rate $p$, the probability of a clean transfer is approximately
\begin{equation}[H]\label{eq:ber}
P_{\text{ok}} \approx (1-p)^{N} \approx \mathrm{e}^{-pN}.
\end{equation}
For large $N$, expected retries per transfer $\approx (1-P_{\text{ok}})/P_{\text{ok}}$. Use this to dimension replay buffers and to set retry budgets that avoid thrashing.

Implementation analysis: hardware must combine three mechanisms:
\begin{enumerate}
\item Lightweight per-TLP/DMA descriptor tracking so the DMA engine can detect missing completions and reissue. Track descriptor ID, length, timestamp, and a poison bit set by completion with ECRC or user-level error.
\item A timeout unit calibrated to worst-case PCIe round-trip latency plus SM scheduling jitter. If outstanding bytes $W$ and link bandwidth $B$, a minimal timeout $T_{\text{min}}$ satisfies $T_{\text{min}} \geq W/B + t_{\text{link}} + t_{\text{host\_proc}}$.
\item AEC/AER reporting fabric that escalates to firmware for non-fatal or fatal errors, and injects poison markers into GPU page tables to avoid silent corruption for P2P accesses.
\end{enumerate}

Example synthesizable Verilog for a compact error handler that tracks outstanding DMA descriptors, times out, counts errors, and asserts an AER request is below. Adjust parameters to system scale.

\begin{lstlisting}[language=Verilog,caption={Compact PCIe error tracker and AER generator},label={lst:pcie_err}]
module pcie_err_handler #(parameter DEPTH=16, IDW=8, LENW=16, TMW=24)
(
  input  wire                 clk,
  input  wire                 rst,
  // incoming DMA issue
  input  wire                 issue_valid,
  input  wire [IDW-1:0]       issue_id,
  input  wire [LENW-1:0]      issue_len,
  // completion indication
  input  wire                 comp_valid,
  input  wire [IDW-1:0]       comp_id,
  input  wire                 comp_err, // 1 = error (ECRC/poison)
  // configuration and outputs
  input  wire [TMW-1:0]       tmo_thresh,
  output reg                  aer_request,
  output reg [31:0]           fatal_count
);

  // simple tracker arrays
  reg                  valid [0:DEPTH-1];
  reg [IDW-1:0]        id    [0:DEPTH-1];
  reg [LENW-1:0]       len   [0:DEPTH-1];
  reg [TMW-1:0]        age   [0:DEPTH-1];

  integer i;
  // allocation pointer (wrap)
  reg [$clog2(DEPTH)-1:0] alloc_ptr;

  // reset
  always @(posedge clk) begin
    if (rst) begin
      alloc_ptr <= 0;
      aer_request <= 0;
      fatal_count <= 0;
      for (i=0;i= tmo_thresh) begin
            // escalate as fatal if beyond threshold twice (simple policy)
            fatal_count <= fatal_count + 1;
            aer_request <= 1'b1;
            valid[i] <= 1'b0; // retire entry to prevent resource leak
          end
        end
      end

      // allocate new descriptor
      if (issue_valid) begin
        // find free slot (simple next-fit)
        if (!valid[alloc_ptr]) begin
          valid[alloc_ptr] <= 1'b1;
          id[alloc_ptr] <= issue_id;
          len[alloc_ptr] <= issue_len;
          age[alloc_ptr] <= {TMW{1'b0}};
          alloc_ptr <= alloc_ptr + 1'b1;
        end else begin
          // allocation collision: treat as fatal to avoid deadlock
          fatal_count <= fatal_count + 1;
          aer_request <= 1'b1;
        end
      end

      // process completion
      if (comp_valid) begin
        // simple linear search (small DEPTH) for matching id
        for (i=0;i
\section{Section 4: High-Speed Serial Links}
\subsection{Item 1:  NVLink and Infinity Fabric}
The previous discussion of on-chip routers and PHY equalization set the context for off-chip high-speed fabrics by stressing topology, flow-control, and signal-integrity trade-offs that carry over to inter-GPU links. This subsection analyzes architectural choices and performance trade-offs between two dominant implementations used in modern GPU systems: NVIDIA's NVLink and AMD's Infinity Fabric.

Problem: provide low-latency, high-bandwidth, coherent or loosely-coherent connectivity across dies and boards to enable scalable multi-GPU and CPU–GPU systems. Analysis focuses on protocol layering, aggregation, flow control, and application-level effect on graphics and ML workloads.

Main exposition.

\begin{itemize}
\item Protocol and topology: both fabrics are packetized, lane-aggregated serial links with link training, per-lane equalization, and CRC-based integrity. They expose:
\begin{enumerate}
\item Multiple virtual channels to avoid head-of-line blocking for traffic classes (memory, coherency, message).
\item Credit-based flow control to bound buffering and prevent overruns across SM/CU endpoints.
\item Topologies ranging from point-to-point meshes to multi-rail fat-tree fabrics; topology choice impacts worst-case bisection bandwidth and collective operation latency.
\end{enumerate}
\item Coherency and memory model: newer NVLink generations support GPU memory coherency across devices within a unified address space, enabling direct load/store semantics and reducing software synchronization cost. Infinity Fabric typically integrates coherency between CPU caches and chiplet dies in AMD designs, enabling tighter CPU–GPU sharing in heterogeneous systems.
\item Performance modeling: design and analysis commonly use an aggregated-bandwidth model. For $N$ lanes at per-lane line rate $R$ (bytes/s) and protocol efficiency $\eta$ ($0<\eta\leq1$), effective bandwidth is:
\begin{equation}[H]\label{eq:agg_bw}
B_{\mathrm{eff}} = N \cdot R \cdot \eta.
\end{equation}
Latency for an injected packet of payload $P$ (bytes) includes serialization, switch traversal, and queuing; serialization dominant term is $T_{\mathrm{ser}} = P/(N\cdot R)$ assuming striping across lanes.
\end{itemize}

Implementation: simple calculator and microbenchmark estimator in Python to explore trade-offs (serialize overhead, link efficiency, and round-trip cost).

\begin{lstlisting}[language=Python,caption={Link bandwidth and latency estimator},label={lst:link_est}]
def eff_bandwidth(lanes, lane_rate_bytes, efficiency):  # compute B_eff
    return lanes * lane_rate_bytes * efficiency

def round_trip_latency(payload_bytes, lanes, lane_rate_bytes, switch_hops,
                       per_hop_ns=20, fixed_ns=200, efficiency=0.9):
    # serialization + switch latency + fixed overhead
    ser_ns = payload_bytes / (lanes * lane_rate_bytes) * 1e9
    return ser_ns + switch_hops * per_hop_ns + fixed_ns

# example usage
lanes=8; lane_rate=3e9  # 3 GB/s per lane
B=eff_bandwidth(lanes,lane_rate,0.92)  # bytes/sec
rtt=round_trip_latency(64,lanes,lane_rate,2)  # ns
print(B, rtt)  # bytes/sec, ns