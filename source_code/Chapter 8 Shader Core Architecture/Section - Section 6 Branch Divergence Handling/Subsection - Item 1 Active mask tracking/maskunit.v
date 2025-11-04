module active_mask_unit #(
  parameter W = 32
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire [W-1:0]         cur_mask,      // current active lanes
  input  wire [W-1:0]         taken_mask,    // lanes deciding taken path
  input  wire                 commit,        // commit pulse to update mask
  output reg  [W-1:0]         next_mask,     // updated active mask
  output reg  [$clog2(W+1)-1:0] active_count  // popcount result
);
  // compute the next mask: lanes that are both active and follow taken path
  wire [W-1:0] masked_taken = cur_mask & taken_mask;

  // parallel popcount via adder tree (generate)
  wire [$clog2(W+1)-1:0] partial_count;
  genvar i;
  // simple reduction using integer loop synthesizable in many flows
  integer idx;
  reg [$clog2(W+1)-1:0] sum;
  always @(*) begin
    sum = 0;
    for (idx=0; idx
\subsection{Item 2:  Divergence stack logic}
Continuing from active-mask maintenance, the divergence stack operates on those per-warp masks to record suspended execution paths and their reconvergence points; the stack is the hardware mechanism that converts boolean lane-level decisions into serializable SIMT control sequences. This subsection treats the problem, derives the key bitwise operations and depth bound, provides a synthesizeable Verilog implementation, and states concrete design trade-offs.

Problem: when a branch causes some lanes to take the branch target and others to fall through, the SM must \begin{enumerate} \item continue executing one sub-path with a reduced active mask, \item remember the other sub-path and its reconvergence PC, and \item restore that path when the current sub-path is finished. \end{enumerate} Failure to handle nested divergence correctly either wastes cycles (by serializing too early) or produces incorrect results.

Analysis: let $M_a$ be the current active mask and $C$ be the per-lane condition mask for the branch; both are $W$-bit vectors where $W$ is warp width. The two resulting masks are
\begin{equation}[H]\label{eq:mask_split}
M_{\text{taken}} = M_a \land C,\qquad M_{\text{not}} = M_a \land \lnot C.
\end{equation}
If both $M_{\text{taken}}$ and $M_{\text{not}}$ are non-zero, the stack receives an entry describing the suspended sub-path (mask and reconvergence PC), then the core sets $M_a$ to one of the masks and branches to the corresponding PC. The worst-case hardware stack depth occurs when each lane requires a unique suspended path; thus a safe upper bound is
\begin{equation}[H]\label{eq:depth_bound}
D_{\max} \le W - 1,
\end{equation}
since at most $W-1$ pushes can be outstanding before only one lane remains active. Practical implementations cap $D$ (e.g. 8 entries) and must handle overflow via serialization, kernel replay, or compiler-rewritten control flow.

Implementation: the Verilog module below provides a parameterized, synchronous divergence stack storing $W$-bit masks and PC values; it supports push when both masks non-zero, pop on reconvergence, and exposes \lstinline|empty|/\lstinline|full| signals. The design uses a simple pointer and registers for small-area, low-latency operation typical inside an SM.

\begin{lstlisting}[language=Verilog,caption={Synthesizable divergence stack for warp-level masks},label={lst:div_stack}]
module div_stack #(
  parameter WARP = 32,
  parameter PC_W = 32,
  parameter DEPTH = 8
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 push,            // push valid entry
  input  wire [WARP-1:0]      push_mask,
  input  wire [PC_W-1:0]      push_pc,
  input  wire                 pop,             // pop on reconv
  output reg  [WARP-1:0]      top_mask,
  output reg  [PC_W-1:0]      top_pc,
  output wire                 empty,
  output wire                 full,
  output reg  [$clog2(DEPTH+1)-1:0] depth
);
  // storage arrays
  reg [WARP-1:0] mask_mem [0:DEPTH-1];
  reg [PC_W-1:0] pc_mem   [0:DEPTH-1];
  reg [$clog2(DEPTH)-1:0] sp; // stack pointer (points to next free)

  integer i;
  always @(posedge clk) begin
    if (rst) begin
      sp <= 0;
      depth <= 0;
      top_mask <= {WARP{1'b0}};
      top_pc <= {PC_W{1'b0}};
      for (i=0;i
\subsection{Item 3:  Reconvergence hardware}
The previous discussion of stack-based control records and per-lane active-mask updates motivates a dedicated reconvergence unit: it must store deferred execution contexts created by mask splits and present the correct active mask when a warp reaches its reconvergence program counter. The reconvergence hardware bridges per-branch bookkeeping with the warp scheduler so the SM can resume the complementary path efficiently.

Problem: when a conditional divides a warp, lanes follow different dynamic targets but the SIMT model requires serializing those paths while preserving per-lane state. Analysis shows the hardware must:
\begin{itemize}
\item capture the complement mask and a reconvergence PC (the post-dominator or explicit target),
\item support nested divergence (LIFO semantics for structured code) or more general reconv points if compiler provides post-dominators,
\item expose the currently active mask to the warp scheduler and signal when to pop.
\end{itemize}

A common entry layout:
\begin{itemize}
\item reconv\_pc: return address where the saved mask must be reactivated,
\item saved\_mask: bitmask of lanes deferred,
\item branch\_taken\_pc: PC for currently executing subset (optional),
\item valid flag.
\end{itemize}

Mask split math for a binary branch with per-lane predicate mask $P$ and current active mask $A$:
\begin{equation}[H]\label{eq:mask_split}
A_{t} = A \;\&\; P,\qquad A_{n} = A \;\&\; \neg P,
\end{equation}
where $A_{t}$ executes immediately if branch taken, and $A_{n}$ is pushed with reconv PC. On return, the scheduler loads $A_{n}$ as the active mask.

Implementation: a per-warp reconvergence stack as synthesizable Verilog. It provides push when both $A_{t}$ and $A_{n}$ non-zero, pop when the warp PC equals top.reconv\_pc and no other pending condition. The stack depth parameter bounds nested condition handling; compilers generally keep depth small for structured kernels.

\begin{lstlisting}[language=Verilog,caption={Per-warp reconvergence stack (parameterized, synthesizable)},label={lst:reconv_stack}]
module reconv_stack #(
  parameter WARPSIZE = 32,
  parameter DEPTH = 8,
  parameter PC_WIDTH = 16
)(
  input  wire clk,
  input  wire rst,
  // push interface
  input  wire push,                      // push new entry
  input  wire [PC_WIDTH-1:0] push_reconv_pc,
  input  wire [WARPSIZE-1:0] push_mask,
  // pop condition (e.g., warp PC equals top.reconv_pc)
  input  wire pop,
  // status / peek
  output reg  [WARPSIZE-1:0] top_mask,
  output reg  [PC_WIDTH-1:0] top_reconv_pc,
  output reg  empty
);
  // storage arrays
  reg [PC_WIDTH-1:0] st_reconv_pc [0:DEPTH-1];
  reg [WARPSIZE-1:0] st_mask      [0:DEPTH-1];
  reg [$clog2(DEPTH+1)-1:0] sp; // stack pointer (number of entries)
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      sp <= 0;
      empty <= 1'b1;
      top_mask <= {WARPSIZE{1'b0}};
      top_reconv_pc <= {PC_WIDTH{1'b0}};
      for (i=0;i
\subsection{Item 4:  Performance impact analysis}
The reconvergence hardware and the divergence stack logic described previously set the mechanisms by which a warp's control flow is serialized back to a common PC; this subsection quantifies how those mechanisms translate into lost cycles and reduced SM throughput. We now model divergence-induced slowdown, show how to measure it in hardware, and discuss practical trade-offs for scheduler and reconvergence design.

Problem and analysis. Under SIMT, a branch that partitions a warp into $m$ serially-executed subgroups increases per-warp execution time roughly by a factor $m$ when the same instruction sequence follows each path. For a program with branch frequency $p_b$ (fraction of dynamic instructions that are branch points of interest) and an average number of serialized segments per encountered branch $\mathbb{E}[m]$, the first-order slowdown $S$ (ratio of elapsed cycles to an ideal, no-divergence baseline) is
\begin{equation}[H]\label{eq:slowdown}
S \approx 1 + p_b\big(\mathbb{E}[m]-1\big).
\end{equation}
For the common two-target branch (taken / not-taken) with per-thread taken probability $q$ and warp width $W$, the probability both outcomes are present in a warp is
\begin{equation}[H]\label{eq:Pboth}
P_{\mathrm{both}} = 1 - q^{W} - (1-q)^{W},
\end{equation}
so $\mathbb{E}[m]=1+P_{\mathrm{both}}$ and therefore $S\approx 1 + p_b P_{\mathrm{both}}$. With $W=32$ and $q=0.5$, $P_{\mathrm{both}}\approx 1$, implying each divergent branch almost certainly serializes into two passes and contributes directly to slowdown proportional to its frequency.

Implementation: on-chip profiling. Use a lightweight synthesizable profiler in the SM to count branch events, detect two-way divergence based on the per-lane taken bitmap, and accumulate an estimated cycles-lost metric by adding an expected extra segment latency (instruction-sequence length estimate). The Verilog below is synthesizable and intended for insertion inside the warp-scheduler instrumentation path.

\begin{lstlisting}[language=Verilog,caption={Warp-level divergence profiler (synthesizable).},label={lst:divprof}]
module warp_divergence_profiler #(
  parameter W = 32,
  parameter CNT_W = 32
)(
  input  wire                  clk,
  input  wire                  rst,
  input  wire                  branch_valid,               // branch resolved this cycle
  input  wire [W-1:0]          active_mask,                // lanes active in warp
  input  wire [W-1:0]          taken_mask,                 // lanes taking target
  input  wire [15:0]           instr_len,                 // estimated path length (cycles)
  output reg  [CNT_W-1:0]      branch_count,
  output reg  [CNT_W-1:0]      divergent_count,
  output reg  [63:0]           cycles_lost
);
  wire divergence_detect = branch_valid && (taken_mask != 0) && (taken_mask != active_mask);

  always @(posedge clk) begin
    if (rst) begin
      branch_count    <= 0;
      divergent_count <= 0;
      cycles_lost     <= 0;
    end else begin
      if (branch_valid) branch_count <= branch_count + 1;
      if (divergence_detect) begin
        divergent_count <= divergent_count + 1;
        cycles_lost     <= cycles_lost + instr_len; // approximate extra segment cost
      end
    end
  end
endmodule