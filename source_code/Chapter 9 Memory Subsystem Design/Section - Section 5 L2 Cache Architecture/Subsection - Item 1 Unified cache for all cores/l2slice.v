module l2_slice #(
  parameter ADDR_WIDTH=32, DATA_WIDTH=128, TAG_WIDTH=20,
  parameter NUM_LINES=1024
)(
  input  wire                     clk,
  input  wire                     rst,
  // request interface (multi-source arbitrated externally)
  input  wire                     req_valid,
  input  wire [ADDR_WIDTH-1:0]    req_addr,
  input  wire                     req_rw,      // 0=read,1=write
  input  wire [DATA_WIDTH-1:0]    req_wdata,
  output reg                      resp_valid,
  output reg  [DATA_WIDTH-1:0]    resp_data,
  // refill interface to memory controller
  output reg                      mem_req_valid,
  output reg  [ADDR_WIDTH-1:0]    mem_req_addr,
  input  wire                     mem_resp_valid,
  input  wire [DATA_WIDTH-1:0]    mem_resp_data
);
  localparam INDEX_BITS = $clog2(NUM_LINES);
  // tag and data arrays (simple synthesizable RAMs)
  reg [TAG_WIDTH-1:0] tag_ram [0:NUM_LINES-1];
  reg [DATA_WIDTH-1:0] data_ram [0:NUM_LINES-1];
  reg valid_bit [0:NUM_LINES-1];

  wire [INDEX_BITS-1:0] idx = req_addr[INDEX_BITS+3:4]; // 16-byte lines
  wire [TAG_WIDTH-1:0]   tag = req_addr[ADDR_WIDTH-1:ADDR_WIDTH-TAG_WIDTH];

  always @(posedge clk) begin
    if (rst) begin
      resp_valid <= 0;
      mem_req_valid <= 0;
      // invalidate lines
      integer i;
      for (i=0;i
\subsection{Item 2:  Partitioning and crossbar interconnect}
The previous discussion established L2 as a unified, shared level that must balance many SMs' demands; partitioning and the interconnect determine whether that sharing becomes a scalable high-throughput subsystem or a contention hotspot. Here we quantify the problem, examine partitioning strategies and crossbar topologies, give an implementation-ready arbitration primitive, and draw concrete hardware trade-offs.

Problem and analysis. Multiple SMs (or CUs) issue cache-line requests to an array of L2 slices and then to DRAM channels. The crossbar must (1) provide enough aggregate bandwidth and (2) limit head-of-line blocking and latency variance. If $N$ masters each produce average bandwidth $b_i$ and the L2-to-DRAM path per-slice peak bandwidth is $B_{\text{slice}}$, a necessary capacity condition is
\begin{equation}[H]\label{eq:bandwidth}
B_{x} \;=\; M \cdot B_{\text{slice}} \;\ge\; \sum_{i=1}^{N} b_i,
\end{equation}
where $M$ is the number of independent slices. However capacity alone is insufficient: transient bursts produce queuing. Using an M/M/1 approximation for a slice's request queue with arrival rate $\lambda$ and service rate $\mu$ (service rate tied to slice and DRAM arbitration), the utilization $\rho=\lambda/\mu$ gives mean queue length $L=\rho/(1-\rho)$. As $\rho$ approaches 1, latency explodes; hence partitioning and admission control aim to keep per-slice $\rho$ well below saturation.

Partitioning strategies and topologies.
\begin{itemize}
\item Static address-based partitioning: map address ranges to slices to maximize locality from texture/compute kernels with regular accesses. Low metadata overhead, but suffers from hot-spotting under skewed workloads.
\item Dynamic hashing/swish: block addresses are hashed across slices to smooth load; improves average utilization but raises coherence and prefetch complexity.
\item Crossbar topologies:
\begin{itemize}
\item Full crossbar: any SM to any slice; simple logically but $O(NM)$ switch complexity and wire pitch; high area and power.
\item Segmented/banked crossbar: group SMs to switch segments with limited connectivity to subsets of slices, reducing switch radix and wiring while providing most locality benefits.
\item Banyan/clos networks: multistage switching reduces per-stage radix at the cost of in-network arbitration and potential blocking; good tradeoff for large $N$, $M$.
\end{itemize}
\end{itemize}

Implementation primitive. A lightweight, synthesizable $N$-way round-robin arbiter is a building block for per-slice request arbitration. The module below grants one request per cycle in a fair rotating order.

\begin{lstlisting}[language=Verilog,caption={N-way round-robin arbiter for L2-slice input ports},label={lst:rr_arb}]
module rr_arbiter #(
  parameter N = 8
) (
  input  wire           clk,
  input  wire           rst_n,
  input  wire [N-1:0]   req,       // request one-hot or multi-hot
  output reg  [N-1:0]   grant,     // one-hot grant
  output wire           grant_valid
);
  // rotate pointer
  reg [$clog2(N)-1:0] ptr;
  integer i;
  wire [N-1:0] rotated;
  // generate rotated request vector
  assign rotated = (req << ptr) | (req >> (N - ptr));
  // grant valid when any request
  assign grant_valid = |req;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ptr <= 0;
      grant <= 0;
    end else begin
      grant <= 0;
      // find first set bit in rotated vector
      for (i = 0; i < N; i = i + 1) begin
        if (rotated[i] && grant == 0) begin
          // unrotate grant back to original index
          grant <= (1'b1 << ((i + ptr) % N));
          ptr <= ((i + ptr + 1) % N); // next priority
        end
      end
    end
  end
endmodule