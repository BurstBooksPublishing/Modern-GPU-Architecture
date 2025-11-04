module victim_cache #(
  parameter N = 8,
  parameter TAGW = 20,
  parameter DATAW = 128
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 rd_req,           // probe request
  input  wire [TAGW-1:0]      rd_tag,
  output reg                  rd_hit,
  output reg  [DATAW-1:0]     rd_data,
  input  wire                 wr_req,           // insert evicted line
  input  wire [TAGW-1:0]      wr_tag,
  input  wire [DATAW-1:0]     wr_data,
  output reg  [TAGW-1:0]      evicted_tag,      // old entry on insert
  output reg  [DATAW-1:0]     evicted_data
);
  // storage
  reg [TAGW-1:0] tags [0:N-1];
  reg [DATAW-1:0] data_mem [0:N-1];
  reg valid [0:N-1];
  integer i;
  reg [$clog2(N)-1:0] rr_ptr;

  // combinational probe
  reg hit_comb;
  reg [$clog2(N)-1:0] hit_idx;
  always @(*) begin
    hit_comb = 1'b0; hit_idx = 0;
    for (i=0;i
\subsection{Item 4:  Cache slice arbitration}
Building on how victim buffering reduces slice-to-DRAM pressure and how partitioning plus a crossbar confines request paths, arbitration at each L2 slice becomes the final arbiter of concurrent demand from multiple SMs and internal pipelines.

The problem: many simultaneous memory requests (loads, stores, writebacks, atomics, texture fetches) converge on a single cache slice, and unmanaged contention causes unfair delays, head-of-line blocking, and throughput loss in SIMT workloads. Analysis therefore must quantify wait and throughput bounds and provide an implementable arbiter that enforces fairness, priorities for latency-sensitive traffic, and backpressure-compatible handshakes.

Analysis and simple bounds. Assume $N$ requestors and a slice that issues $s$ grants per cycle (commonly $s=1$). Under a fair round-robin policy and uniform request arrival, expected wait
\begin{equation}[H]\label{eq:avg_wait}
E[W] = \frac{N-1}{2s}\ \text{cycles},
\end{equation}
and worst-case wait is
\begin{equation}[H]\label{eq:worst_wait}
W_{\max} = \frac{N-1}{s}\ \text{cycles}.
\end{equation}
Throughput is bounded by $T \le s$ requests/cycle; effective utilization drops if requests are blocked downstream by MSHR or DRAM backpressure.

Implementation approach. Practical L2 slice arbiters combine:
\begin{itemize}
\item Round-robin fairness to avoid starvation for bursty SM traffic.
\item Priority escalation for atomics and response-path writebacks to maintain correctness and reduce tail-latency for critical operations.
\item Credit-aware admission: track per-requestor outstanding grants to respect MSHR limits and avoid allocate storms.
\item Virtual-channel (VC) aware selection so responses and high-priority VCs bypass bulk texture fetch VCs.
\end{itemize}

Below is a synthesizable, parameterized Verilog round-robin arbiter with optional priority override input. It supports \lstinline|req| handshake inputs and one-hot \lstinline|grant| outputs, and preserves round-robin state across cycles.

\begin{lstlisting}[language=Verilog,caption={Parameterized round-robin arbiter with priority override},label={lst:rr_arb}]
module rr_arbiter #(
  parameter N = 8
)(
  input               clk,
  input               rst_n,
  input  [N-1:0]      req,        // request vector
  input  [N-1:0]      pri;        // priority override (one-hot higher priority)
  output reg [N-1:0]  grant,      // one-hot grant
  output              grant_valid
);
  // rotation pointer
  reg [$clog2(N)-1:0] ptr;
  integer i;
  wire [N-1:0] masked_req;
  // priority if any; otherwise use rotated round-robin mask
  assign grant_valid = |grant;

  // construct rotated mask starting at ptr
  function [N-1:0] rotate_mask(input [$clog2(N)-1:0] start);
    integer j;
    begin
      rotate_mask = {N{1'b0}};
      for (j=0;j
\section{Section 6: Memory Controller}
\subsection{Item 1:  DRAM command generation}
Building on the L2 cache slice arbitration and transaction flow discussed previously, the memory controller must convert those cache-level transactions into time-ordered DRAM primitives while preserving QoS and maximizing bandwidth. This subsection analyzes the mapping from high-level requests to timed DRAM commands, presents a compact synthesizable implementation sketch for a command generator, and highlights the trade-offs for GPU workloads (graphics, ML, ray tracing).

The central problem is enforcing the DRAM command timing and bank/row semantics while keeping SMs busy. Key elements are:
\begin{itemize}
\item Bank state machine: each bank is tracked as IDLE, ROW_OPEN, or PRECHARGING with the open-row address.
\item Timing constraints: per-chip timing parameters such as $t_{\mathrm{RCD}}$ (ACT-to-READ), $t_{\mathrm{RP}}$ (PRE-to-ACT), $t_{\mathrm{RAS}}$ (minimum ACT-to-PRE), $t_{\mathrm{RC}}$ (row cycle), and $t_{\mathrm{CCD}}$ (column-to-column).
\item Command ordering and reordering: controller may reorder to exploit bank parallelism or keep a hot row open for locality while respecting QoS.
\end{itemize}

Analysis: For a single bank issuing back-to-back accesses that require an ACT and PRE per access, the minimum service interval per access is approximately $t_{\mathrm{RC}}$ cycles. If each access transfers BL bytes per burst (burst length) at an external clock frequency $f$, the per-bank sustainable bandwidth is
\begin{equation}[H]\label{eq:bank_bw}
\mathrm{BW_{bank}} \approx \frac{\mathrm{BL\ bytes}}{t_{\mathrm{RC}}/f} = \mathrm{BL\ bytes}\cdot\frac{f}{t_{\mathrm{RC}}}.
\end{equation}
Parallelism across $N_b$ independent banks raises aggregate bandwidth toward $N_b\cdot\mathrm{BW_{bank}}$, but bank conflicts and address mapping can reduce realized throughput.

Implementation: the command generator must:
\begin{enumerate}
\item Inspect \lstinline|req_addr| to determine \lstinline|bank| and \lstinline|row|.
\item If row is open and timing timers allow, issue READ/WRITE immediately (subject to $t_{\mathrm{CCD}}$).
\item If closed, issue ACT then wait $t_{\mathrm{RCD}}$ before issuing READ/WRITE; schedule PRE when required to meet $t_{\mathrm{RAS}}/t_{\mathrm{RP}}$.
\item Maintain per-bank timers for $t_{\mathrm{RCD}}$, $t_{\mathrm{RAS}}$, $t_{\mathrm{RP}}$ and a global $t_{\mathrm{CCD}}$ counter between column commands.
\end{enumerate}

A compact synthesizable Verilog module implementing these rules is shown; it is intentionally conservative (issue in-order per-bank, reorders across banks opportunistically) to be synthesis-friendly and clear.

\begin{lstlisting}[language=Verilog,caption={Simple DRAM command generator (synthesizable).},label={lst:dram_cmd_gen}]
module dram_cmd_gen #(
  parameter BANKS = 16, parameter ROW_BITS = 16,
  parameter tRCD = 4, parameter tRP = 4, parameter tRAS = 8, parameter tCCD = 2
)(
  input  wire                   clk, reset,
  // incoming request
  input  wire                   req_valid,
  input  wire [31:0]            req_addr,
  input  wire                   req_is_write,
  output reg                    req_ready,
  // command output
  output reg                    cmd_valid,
  output reg [1:0]              cmd_type, // 0=ACT,1=PRE,2=READ,3=WRITE
  output reg [3:0]              cmd_bank,
  output reg [ROW_BITS-1:0]     cmd_row
);
  // simple address decode
  wire [3:0] bank = req_addr[6:3];
  wire [ROW_BITS-1:0] row = req_addr[22:7];

  // per-bank state
  reg [ROW_BITS-1:0] open_row [0:BANKS-1];
  reg                row_open  [0:BANKS-1];
  reg [7:0]          timer     [0:BANKS-1]; // cycles remaining for timing

  integer i;
  always @(posedge clk) begin
    if (reset) begin
      cmd_valid <= 0; req_ready <= 1;
      for (i=0;i0) timer[i]<=timer[i]-1;
      if (req_valid && req_ready) begin
        // service same-bank hot row
        if (row_open[bank] && open_row[bank]==row && timer[bank]==0) begin
          // obey tCCD globally (simple: stall if last column within tCCD)
          cmd_valid <= 1; cmd_type <= req_is_write ? 3 : 2;
          cmd_bank <= bank; cmd_row <= row;
          // after column command, set CCD cooldown on all banks
          for (i=0;i
\subsection{Item 2:  Row buffer management}
Building on the DRAM command generation policies described earlier, row buffer management decides whether to leave a row open (reducing future ACT cost) or to precharge (freeing the bank for other rows), and thus directly shapes the sequence and timing of ACT/READ/WRITE/PRE commands produced by the controller.

Efficient row buffer management problem → analysis:
\begin{itemize}
\item Problem: SIMT workloads exhibit mixed spatial locality; texture and matrix tiles produce high row locality while random-access scatter/gather from compute shaders produce low locality. The controller must maximize row-buffer hit rate without starving other banks or increasing bank-occupancy latency.
\item Metrics: define row-buffer hit probability $p_{\mathrm{hit}}$ and per-operation latencies $t_{\mathrm{hit}}$ (read/write to open row) and $t_{\mathrm{miss}}$ (requires ACT ± PRE). The expected memory access latency is
\begin{equation}[H]\label{eq:avg_latency}
\mathcal{L}_{\mathrm{avg}} = p_{\mathrm{hit}}\,t_{\mathrm{hit}} + (1-p_{\mathrm{hit}})\,t_{\mathrm{miss}}.
\end{equation}
Reducing $t_{\mathrm{miss}}$ or increasing $p_{\mathrm{hit}}$ lowers $\mathcal{L}_{\mathrm{avg}}$; the controller trades longer occupancy (to retain a row) against multi-bank concurrency.
\end{itemize}

Implementation analysis and policies:
\begin{enumerate}
\item Open-page (leave row open): maximizes hit latency for spatially local kernels (texture filtering, tiled GEMM). Risk: long bank hold times block other warps that map to same bank.
\item Close-page (auto precharge): favors low locality or mixed workloads; reduces worst-case head-of-line blocking and improves fairness.
\item Adaptive policies: maintain per-bank counters (recent hits, access stride) and switch between open/close or issue precharge when predicted miss probability exceeds threshold.
\item Row-buffer aware scheduling: prefer issuing subsequent transactions to banks with open rows (crossbar/queue prioritization), but bound maximum occupancy to avoid starvation.
\end{enumerate}

Implementation (synthesizable Verilog): lightweight per-bank row tag, a small FSM that issues ACT if row mismatch, otherwise issues READ/WRITE; a configurable auto-precharge (close-page) mode or adaptive thresholding counter.
\begin{lstlisting}[language=Verilog,caption={Row buffer manager (per-bank tag + simple FSM)},label={lst:rbm}]
module row_buffer_manager #(
  parameter NUM_BANKS = 8,
  parameter ROW_BITS  = 16,
  parameter BANK_BITS = 3
)(
  input  wire clk,
  input  wire rst_n,
  // request interface
  input  wire req_valid,
  input  wire req_rw,               // 0=read,1=write
  input  wire [BANK_BITS-1:0] req_bank,
  input  wire [ROW_BITS-1:0]  req_row,
  // command output
  output reg  cmd_valid,
  output reg  [2:0] cmd_type,      // 0=NOP,1=ACT,2=READ,3=WRITE,4=PRE
  output reg  [BANK_BITS-1:0] cmd_bank,
  output reg  [ROW_BITS-1:0]  cmd_row
);
  // per-bank open-row tag and valid bit
  reg [ROW_BITS-1:0] row_tag [0:NUM_BANKS-1];
  reg                row_valid [0:NUM_BANKS-1];

  // simple FSM states
  localparam S_IDLE = 2'd0, S_ACT = 2'd1, S_RW = 2'd2, S_PRE = 2'd3;
  reg [1:0] state, next_state;
  reg [BANK_BITS-1:0] cur_bank;
  reg [ROW_BITS-1:0]  cur_row;
  reg                 cur_rw;

  integer i;
  // synchronous state
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      cmd_valid <= 1'b0;
      for (i=0;i
\subsection{Item 3:  Bank interleaving}
The previous subsections established how row-buffer hits and command timing ($t_{\mathrm{RCD}}$, $t_{\mathrm{CAS}}$) interact with scheduling; bank interleaving is the next lever to exploit parallelism across physical banks and channels while respecting those timing constraints.

Bank interleaving problem: many SMs generate bursts of coalesced and strided accesses that concentrate on a small set of DRAM banks, producing bank conflicts and long serialization. Analysis shows two orthogonal goals: (1) maximize outstanding independent activations to hide $t_{\mathrm{RCD}}$ and CAS latency, and (2) avoid violating DRAM activation-rate limits ($t_{\mathrm{RRD}}$, $t_{\mathrm{FAW}}$). A simple capacity model gives the maximum useful concurrency as the minimum of physical resources and timing windows. Let $B_{\mathrm{total}}$ be total banks across channels and ranks, $C$ the number of channels, and define $k$ as the maximum activations allowed in a $t_{\mathrm{FAW}}$ window:
\begin{equation}[H]\label{eq:peff}
k = \left\lfloor\frac{t_{\mathrm{FAW}}}{t_{\mathrm{RRD}}}\right\rfloor,\qquad
P_{\mathrm{effective}}=\min\{B_{\mathrm{total}},\,C\times\text{ranks},\,k\}.
\end{equation}
This shows bank count alone does not guarantee higher parallelism; controller scheduling must respect the $t_{\mathrm{FAW}}$ envelope.

Design patterns that reduce conflicts:
\begin{itemize}
\item Low-order interleaving maps consecutive cache-line addresses to different banks and channels so warp-coalesced streams spread naturally.
\item XOR (bitwise swizzle) of lower and higher address bits reduces pathological strides (power-of-two strides) that concentrate accesses.
\item Page-granular mappings keep rows aligned to increase row-buffer hits while still distributing banks.
\end{itemize}

Implementation: a parameterized bank mapper in the memory controller implements selectable interleave modes (identity or XOR swizzle), exposing bank, row, and channel fields for the DRAM scheduler. The mapper is feed-forward and cheap; it provides deterministic mapping so the coalescing unit and scheduler can predict bank targets and avoid conflicts.

\begin{lstlisting}[language=Verilog,caption={Parameterizable bank, row, and channel mapper (synthesizable).},label={lst:bankmap}]
module bank_mapper
 #(parameter ADDR_W=32, BANK_BITS=4, CHANNEL_BITS=1, LINE_BITS=6)
 (input  wire [ADDR_W-1:0] addr,
  output wire [BANK_BITS-1:0] bank,
  output wire [CHANNEL_BITS-1:0] channel,
  output wire [ADDR_W-LINE_BITS-BANK_BITS-CHANNEL_BITS-1:0] row);
  // Derived widths
  localparam ROW_W = ADDR_W - LINE_BITS - BANK_BITS - CHANNEL_BITS;
  wire [LINE_BITS-1:0] col = addr[LINE_BITS-1:0]; // byte-in-line
  wire [BANK_BITS-1:0] bank_low = addr[LINE_BITS+BANK_BITS-1:LINE_BITS];
  wire [ROW_W-1:0] row_bits = addr[LINE_BITS+BANK_BITS +: ROW_W];
  // Optional XOR swizzle: use higher row bits if available
  wire [BANK_BITS-1:0] high_bits;
  generate
    if (ROW_W >= BANK_BITS) begin
      assign high_bits = row_bits[ROW_W-1 -: BANK_BITS];
    end else begin
      assign high_bits = {BANK_BITS{1'b0}};
    end
  endgenerate
  assign bank = bank_low ^ high_bits;                 // XOR swizzle
  assign row  = row_bits;                             // row for RAS
  assign channel = addr[LINE_BITS+BANK_BITS+ROW_W +: CHANNEL_BITS]; // channel select
endmodule