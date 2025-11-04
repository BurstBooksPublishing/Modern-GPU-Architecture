module fp_exc_aggregator #(
  parameter integer LANES = 32
)(
  input  wire                  clk,
  input  wire                  rst_n,
  input  wire                  in_valid,
  input  wire [LANES*5-1:0]    in_flags, // 5 flags per lane: {invalid,divzero,overflow,underflow,inexact}
  input  wire                  out_ready,
  output reg                   out_valid,
  output reg  [4:0]            out_agg_flags // aggregated sticky flags
);
  // internal sticky accumulation
  reg [4:0] sticky;
  integer i;
  wire [4:0] lane_flags [0:LANES-1];
  // unpack lanes
  generate
    genvar gi;
    for (gi=0; gi
\section{Section 5: Verification Basics}
\subsection{Item 1:  Testbench architecture}
Building on the floating-point corner-case verification techniques, a robust testbench architecture defines the harness and transactors that exercise GPU-specific datapaths such as SM pipelines, TMU request streams, tensor-core MAC arrays, and ROP writeback under realistic SIMT and memory stress. The testbench must translate high-level workloads into cycle-accurate stimuli, collect golden-reference outputs, and measure metrics that map directly to microarchitectural performance goals.

Problem: validate complex, highly parallel modules while preserving observability and controllability. Analysis shows three orthogonal responsibilities:
\begin{enumerate}
\item Stimulus generation and sequencing — convert kernel-level behavior into packetized transactions and warp-aligned memory accesses.
\item Checking and coverage — perform functional equivalence checks, temporal ordering checks across multiple clock domains, and measure coverage of corner cases (denormals, NaN propagation, divergence reconvergence).
\item Measurement and profiling — compute throughput, latency distributions, and resource utilization counters for roofline analysis.
\end{enumerate}

Implementation patterns:
\begin{itemize}
\item Use synthesizable transactors for emulation environments and class-based drivers for fast RTL simulation. Transactors translate between APB/AXI-like bus transfers and internal valid-ready flits.
\item A scoreboard maintains per-warp outstanding counts and checks writeback order, enabling detection of mis-ordered ROP commits caused by pipeline bypass or writeback hazards.
\item Inject controlled contention: concurrent TMU read bursts, shared-memory bank conflicts, and tensor-core mixed-precision operand streams to expose timing-dependent bugs.
\end{itemize}

Operational relevance before math: throughput and utilization metrics from the testbench are inputs to architectural trade-offs and must be computed deterministically from captured counters. For a stream of successful transactions counted as $N$ over $C$ cycles at clock frequency $f_{\mathrm{clk}}$, sustained throughput in transactions per second is
\begin{equation}[H]\label{eq:throughput}
T = \frac{N}{C}\cdot f_{\mathrm{clk}}.
\end{equation}
Latency distribution is computed by sampling enqueue and dequeue cycle stamps per warp lane, then reporting percentiles.

A compact, synthesizable monitor useful in emulation validates valid-ready channels and counts transactions. It can be instantiated inside FPGA builds for hardware-assisted verification:

\begin{lstlisting}[language=Verilog,caption={Synthesizable valid-ready channel monitor for emulation},label={lst:vr_monitor}]
module vr_monitor #(
  parameter ID_WIDTH = 8
)(
  input  wire        clk,
  input  wire        rst_n,
  input  wire        valid,    // producer valid
  input  wire        ready,    // consumer ready
  input  wire [ID_WIDTH-1:0] id,// transaction id
  output reg  [31:0] tx_count, // committed transactions
  output reg         error     // ordering or protocol error
);
  reg [ID_WIDTH-1:0] last_id;
  always @(posedge clk) begin
    if (!rst_n) begin
      tx_count <= 0;
      last_id  <= 0;
      error    <= 0;
    end else begin
      // commit on handshake
      if (valid && ready) begin
        tx_count <= tx_count + 1;
        // simple ordering check: ids should be non-decreasing for a stream
        if (id < last_id) error <= 1;
        last_id <= id;
      end
    end
  end
endmodule