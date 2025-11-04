module crossbar #(
  parameter N = 8,                // number of inputs (SM links)
  parameter M = 4,                // number of outputs (memory partitions)
  parameter W = 64                // data width
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [N-1:0]         in_valid,
  input  wire [N*W-1:0]       in_data,
  input  wire [$clog2(M)-1:0] in_dest [N-1:0], // per-input destination
  output reg  [N-1:0]         in_ready,
  output reg  [M-1:0]         out_valid,
  output reg  [M*W-1:0]       out_data,
  input  wire [M-1:0]         out_ready
);

  // per-output priority pointer and combinational grant generation
  reg [$clog2(N)-1:0] prio_ptr [M-1:0];
  integer i,j;

  // build request matrix: req[j][i] == input i requests output j
  wire req [M-1:0][N-1:0];
  generate
    for (i=0;i
\subsection{Item 2:  Round-robin arbiter}
This module follows the crossbar switch discussion by specifying the per-port arbitration needed at each crossbar input so that competing SM/CU requestors (texture units, TMUs, ROPs, or memory controllers) receive fair service without head-of-line starvation. The round-robin arbiter presented below trades minimal logic for bounded latency and predictable fairness across many SIMT-driven request streams.

Problem: multiple initiators present one-hot requests; the crossbar can accept a single grant per cycle. Goals: (1) no starvation, (2) simple rotate-based priority for fairness, (3) parameterizable width for SM, L2 slice, or PCIe request fans.

Analysis:
\begin{itemize}
\item A canonical round-robin maintains a rotating pointer $P$ that gives priority to requestor $P$, then $P+1$, ..., wrapping to $P-1$.
\item Worst-case wait for any requester in a non-blocking single-grant arbiter is
\begin{equation}[H]\label{eq:worst_wait}
L_{\max} = N - 1,
\end{equation}
where $N$ is the number of requestors; average wait under uniform load is $\approx (N-1)/2$ cycles.
\item Throughput is one grant per cycle when requests are saturated; latency impact on GPU pipelines must be evaluated against SM issue/backpressure and memory controller buffering.
\end{itemize}

Implementation: the Verilog below is synthesizable and parameterized by WIDTH. It finds the first set bit after the pointer by iterating and produces a one-hot grant. The pointer advances to the next position after a successful grant, ensuring strict round-robin fairness.

\begin{lstlisting}[language=Verilog,caption={Parameterizable round-robin arbiter (synthesizable).},label={lst:round_robin}]
module round_robin_arbiter #(parameter WIDTH = 8) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire [WIDTH-1:0]     req,    // one-hot request vector
    output reg  [WIDTH-1:0]     grant   // one-hot grant vector
);
    // pointer width (computed at elaboration)
    function integer clog2(input integer x); integer i; begin i=0; x=x-1; while(x>0) begin x=x>>1; i=i+1; end clog2=i; end endfunction
    localparam PTR_W = (WIDTH<=1) ? 1 : clog2(WIDTH);

    reg [PTR_W-1:0] pointer;            // current priority base
    integer i;
    integer sel_idx;
    reg found;

    // combinational selection: first request at or after pointer
    always @(*) begin
        grant = {WIDTH{1'b0}};
        found = 1'b0;
        sel_idx = 0;
        for (i = 0; i < WIDTH; i = i + 1) begin
            // compute candidate index with wrap-around
            integer idx;
            idx = (pointer + i) % WIDTH;
            if (!found && req[idx]) begin
                grant[idx] = 1'b1;      // one-hot grant
                sel_idx = idx;
                found = 1'b1;
            end
        end
    end

    // pointer update and grant register (synchronous)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pointer <= {PTR_W{1'b0}};
            // grant reset handled by combinational block above; keep stable on reset
        end else begin
            if (|grant) begin
                // advance pointer to next position after granted index
                pointer <= (sel_idx + 1) % WIDTH;
            end
        end
    end
endmodule