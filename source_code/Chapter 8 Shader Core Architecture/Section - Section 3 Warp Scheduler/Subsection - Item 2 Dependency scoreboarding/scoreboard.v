module scoreboard #(
  parameter NUM_WARPS = 32,
  parameter NUM_REGS  = 128,
  parameter REG_ID_W  = 7
)(
  input  wire                     clk,
  input  wire                     rst,
  // issue interface
  input  wire                     issue_valid,
  input  wire [$clog2(NUM_WARPS)-1:0] issue_warp, // warp id
  input  wire [REG_ID_W-1:0]      issue_src0,
  input  wire [REG_ID_W-1:0]      issue_src1,
  input  wire [REG_ID_W-1:0]      issue_dst,
  output wire                     issue_ready,
  // commit/writeback interface
  input  wire                     commit_valid,
  input  wire [$clog2(NUM_WARPS)-1:0] commit_warp,
  input  wire [REG_ID_W-1:0]      commit_dst
);

  // per-warp busy bitmaps
  reg [NUM_REGS-1:0] busy_bits [0:NUM_WARPS-1];

  integer i;
  // reset
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i
\subsection{Item 3:  Issue policies and priorities}
Dependency tracking from the previous subsection supplies per-warp ready/blocked signals, and the earlier discussion of warp/wavefront execution established that SIMT constraints and divergence shape which lanes are eligible. Those signals feed the issue policy: the arbiter must balance fairness, latency-hiding, and execution-unit affinity while respecting scoreboarding hazards.

Problem: multiple warps often become simultaneously ready but contend for a limited number of issue slots and heterogeneous execution units (scalar ALUs, TF32 tensor cores, TMUs for texturing). A practical policy must maximize utilization while preventing starvation and honoring dependency scoreboards. Key metrics are:
\begin{itemize}
\item warp age (how long it has been ready),
\item memory-wait penalty (waiting on loads/stores),
\item unit affinity (matches to ALU, TMU, tensor core),
\item instruction-level readiness (scoreboard says registers available).
\end{itemize}

A compact, hardware-friendly way to combine these is a linear priority score computed each cycle and compared across warps. Define the score for warp $i$ as
\begin{equation}\label{eq:priority_score}
P_i = \alpha\cdot \mathrm{age}_i \;+\; \beta\cdot \mathrm{ready}_i \;-\; \gamma\cdot \mathrm{memwait}_i \;+\; \delta\cdot \mathrm{affinity}_{i,u},
\end{equation}
where $\mathrm{ready}_i\in\{0,1\}$ indicates instruction-ready (no WAR/WAW/RAW hazards), $\mathrm{memwait}_i$ is 1 if outstanding memory dependence exists, and $\mathrm{affinity}_{i,u}$ is a small integer bonus when the requested unit $u$ matches the issue port. Coefficients $\alpha,\beta,\gamma,\delta$ are integers chosen to fit bit widths so comparison is single-cycle. This score approximates "oldest-first with unit-aware bias" while being simple to implement.

Implementation: the arbiter computes $P_i$ in parallel and performs a max-reduction to select the winning warp. The Verilog below is a synthesizable combinational arbiter for parameterized warp count and simple integer scoring; it assumes external scoreboards gate \lstinline|ready| and \lstinline|memwait| inputs. The output is a one-hot grant and index.

\begin{lstlisting}[language=Verilog,caption={Simple warp arbiter with age and affinity scoring},label={lst:warp_arbiter}]
module warp_arbiter #(
  parameter N_WARPS = 32,
  parameter AGE_W = 8,            // bits for age counters
  parameter SCORE_W = 12          // bits for score
)(
  input  wire [N_WARPS-1:0] ready,    // per-warp ready
  input  wire [N_WARPS-1:0] memwait,  // per-warp mem-wait flag
  input  wire [N_WARPS*AGE_W-1:0] age_vec, // concatenated ages
  input  wire [N_WARPS-1:0] affinity, // affinity bit for this unit
  output reg  [N_WARPS-1:0] grant,    // one-hot grant
  output reg  [$clog2(N_WARPS)-1:0] grant_idx
);
  integer i;
  reg [SCORE_W-1:0] scores [0:N_WARPS-1];
  reg [SCORE_W-1:0] best_score;
  integer best_i;

  always @* begin
    best_score = {SCORE_W{1'b0}};
    best_i = 0;
    for (i=0; i= best_score) begin
        best_score = scores[i];
        best_i = i;
      end
    end
    // produce one-hot grant
    grant = {N_WARPS{1'b0}};
    grant[best_i] = ready[best_i]; // grant only if ready
    grant_idx = best_i[$clog2(N_WARPS)-1:0];
  end
endmodule