module crossbar #(
  parameter N = 4,               // number of inputs
  parameter M = 4,               // number of outputs
  parameter WIDTH = 64           // data width
)(
  input  wire                   clk,
  input  wire                   rstn,
  input  wire [N-1:0]           in_valid,
  input  wire [N*WIDTH-1:0]     in_data,
  output reg  [N-1:0]           in_ready,
  output reg  [M-1:0]           out_valid,
  output reg  [M*WIDTH-1:0]     out_data,
  input  wire [M-1:0]           out_ready
);
  integer i,j;
  // internal one-hot selects per output
  reg [N-1:0] sel [0:M-1];

  always @(*) begin
    // default
    for (i=0;i
\subsection{Item 2:  Router architecture and buffering}
Continuing from the topology discussion, router microarchitecture and buffering choices realize the logical links of a mesh or the centralized paths of a crossbar; topology shapes where congestion accumulates and thus where buffering must be provisioned. The following describes the router pipeline, buffering strategies appropriate for GPU workloads, and a compact Verilog implementation of a credit-aware input FIFO used in input-buffered routers.

A practical router datapath separates concerns into stages:
\begin{enumerate}
\item Routing computation (RC): determine next hop from packet header.
\item Virtual-channel allocation (VCA): assign a free virtual channel to avoid head-of-line (HOL) blocking.
\item Switch allocation (SA): arbitrate contenders for crossbar ports.
\item Switch traversal (ST): pass flit through crossbar fabric.
\item Link traversal (LT): transmit flit across physical link and update credits.
\end{enumerate}
For GPU workloads—texture fetch bursts from TMUs, multicast-like shader message patterns, and sustained memory responses—buffer occupancy spikes are both bursty and correlated across SMs. Two dominant buffering strategies are used: input-buffered routers with virtual channels and credit-based flow control, and more area-costly buffered-crossbar or output-buffered designs used in small-scale crossbars.

Buffer sizing must cover round-trip flow-control latency and peak burstiness. Operationally, to avoid link stalls the minimum buffer depth per VC (in flits) should satisfy the link bandwidth times the round-trip credit latency:
\begin{equation}[H]\label{eq:buffer_depth}
$D_{\min} \approx B_{\text{link}} \cdot RTT_{\text{cycles}}$,
\end{equation}
where $B_{\text{link}}$ is flits per cycle. For example, a 1-flit-per-cycle link with a 20-cycle RTT needs at least 20 flits per VC to sustain full throughput without pipeline bubbles. Add a burst margin factor $\alpha$ (1.2--2.0) to tolerate instantaneous multi-SM bursts and on-chip memory controller jitter.

Virtual channels mitigate HOL blocking and improve throughput; however, they increase meta-data (per-VC pointers, occupancy bits). Credit-based flow control simplifies deadlock avoidance when combined with escape VCs and deterministic routing in meshes. Wormhole routing keeps router latency low by streaming flits, but increases sensitivity to VC availability and thus buffer sizing.

Practical implementation: a synthesizable input FIFO that returns credits when a flit departs. This module can be instantiated per VC in an input-buffered router.

\begin{lstlisting}[language=Verilog,caption={Credit-aware input FIFO for a router VC.},label={lst:fifo_vc}]
module vc_fifo #(
  parameter FLIT_WIDTH = 128,
  parameter DEPTH = 32,          // must be power of two
  parameter PTR_W = $clog2(DEPTH)
)(
  input  wire                   clk,
  input  wire                   rst_n,
  input  wire                   push,      // write enable
  input  wire [FLIT_WIDTH-1:0]  din,       // flit input
  input  wire                   pop,       // read enable (on egress)
  output wire [FLIT_WIDTH-1:0]  dout,      // flit output
  output wire                   full,
  output wire                   empty,
  output wire [PTR_W:0]         credit_out // credits freed on pop
);
  // BRAM-based circular buffer with pointers and occupancy counter
  reg [FLIT_WIDTH-1:0] mem [0:DEPTH-1];
  reg [PTR_W-1:0] wptr, rptr;
  reg [PTR_W:0] occ;              // occupancy counter
  assign full  = (occ == DEPTH);
  assign empty = (occ == 0);
  assign dout  = mem[rptr];
  assign credit_out = (pop && !empty) ? 1 : 0; // return one credit per flit read

  // synchronous write
  always @(posedge clk) begin
    if (!rst_n) begin
      wptr <= 0; rptr <= 0; occ <= 0;
    end else begin
      if (push && !full) begin
        mem[wptr] <= din;
        wptr <= wptr + 1;
        occ <= occ + 1;
      end
      if (pop && !empty) begin
        rptr <= rptr + 1;
        occ <= occ - 1;
      end
    end
  end
endmodule