module channel_dispatcher #(
  parameter ADDR_W = 64,
  parameter DATA_W = 128,
  parameter CH = 8,               // number of channels (power of two)
  parameter CH_BITS = 3
)(
  input  wire                   clk,
  input  wire                   rstn,
  // incoming request
  input  wire                   in_valid,
  input  wire [ADDR_W-1:0]      in_addr,
  input  wire [DATA_W-1:0]      in_data,
  output reg                    in_ready,
  // per-channel outputs
  output reg  [CH-1:0]          ch_valid,
  output reg  [CH-1:0]          ch_ready, // back-pressure from channels
  output reg  [ADDR_W-1:0]      ch_addr [0:CH-1],
  output reg  [DATA_W-1:0]      ch_data [0:CH-1],
  input  wire [CH-1:0]          ch_full    // channel cannot accept
);
  integer i;
  reg [CH_BITS-1:0] preferred;
  reg allocated;
  always @(*) begin
    preferred = in_addr[CH_BITS-1:0]; // low-order interleave
    ch_valid = {CH{1'b0}};
    in_ready = 1'b0;
    allocated = 1'b0;
    // probe up to CH channels starting at preferred
    for (i=0;i
\subsection{Item 4:  Performance balancing}
The previous subsections established how parallel memory channels increase aggregate bandwidth and how designers trade latency against throughput when sizing caches and controllers. Building on that, this subsection addresses the mechanisms required to balance performance across multiple channels, SMs, and functional units so that the theoretical channel capacity is realized in practice.

Problem: GPUs present concurrent request streams from SMs, TMUs, ROPs and DMA engines with differing arrival rates and QoS requirements. Without careful balancing, hot spots (bank conflicts, channel saturation) cause queuing and head-of-line delays that reduce effective throughput and increase tail latency for latency-sensitive workloads such as real-time graphics or interactive ML inference.

Analysis: Model each traffic class $i$ as an arrival rate $\lambda_i$ and each channel $j$ as a service rate $\mu_j$. Stability requires
\begin{equation}[H]\label{eq:stability}
\sum_i \lambda_i \;<\; \sum_j \mu_j .
\end{equation}
For a single-channel M/M/1 approximation the mean queueing delay is $W = 1/(\mu-\lambda)$; as $\lambda \rightarrow \mu$ latency grows quickly, so balancing must avoid sustaining any channel near saturation. A practical target is to shape allocations so expected per-channel bandwidth $B_c$ approximates
\begin{equation}[H]\label{eq:band-per-channel}
B_c \approx \frac{B_{\text{total}}}{C},
\end{equation}
where $C$ is the number of independent channels. Weighted fairness and priority can be encoded with weights $w_i$ giving share $s_i = w_i/\sum_j w_j$.

Implementation: hardware techniques that achieve these targets include:
\begin{itemize}
\item Channel interleaving (address swizzling) to map contiguous blocks across channels and reduce conflict amplification.
\item Per-channel credit/occupancy tracking to prevent overcommitment and steer new requests to underutilized channels.
\item Weighted round-robin or deficit round-robin schedulers in the memory controller to honor QoS while preserving throughput.
\item Small out-of-order request steering within the L2-to-channel crossbar to fill DRAM row buffers and increase effective service $\mu_j$.
\end{itemize}

The following synthesizable Verilog implements a simple XOR-based address swizzle used in many GPUs to spread cacheline accesses across channels; it is cheap, deterministic, and reduces pathological strides that map repeatedly to a single channel.

\begin{lstlisting}[language=Verilog,caption={Simple channel swizzle selector (synthesizable).},label={lst:channel_swizzle}]
module channel_swizzle #(
  parameter integer CHANS     = 8,               // number of channels
  parameter integer CHAN_BITS = 3,               // log2(CHANS)
  parameter integer LINE_BITS = 6                // cacheline size (64B -> 6)
)(
  input  wire [63:0] addr,                       // physical address
  output wire [CHAN_BITS-1:0] chan               // selected channel index
);
  // low bits pick the natural interleave; high bits XOR to reduce stride conflicts
  wire [CHAN_BITS-1:0] low  = addr[LINE_BITS+CHAN_BITS-1:LINE_BITS];
  wire [CHAN_BITS-1:0] high = addr[LINE_BITS+2*CHAN_BITS-1:LINE_BITS+CHAN_BITS];
  assign chan = low ^ high;                      // XOR-swizzle mapping
endmodule