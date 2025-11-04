module noc_router #(
  parameter FLIT_W = 32,
  parameter DEPTH = 4
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [3:0]           in_valid,
  input  wire [3:0][FLIT_W-1:0] in_data,
  output reg  [3:0]           in_ready,    // credits available
  output reg  [3:0]           out_valid,
  output reg  [3:0][FLIT_W-1:0] out_data,
  input  wire [3:0]           out_ready    // downstream backpressure
);

  // Simple shift-register FIFOs (single VC) per input
  reg [FLIT_W-1:0] fifo [3:0][0:DEPTH-1];
  reg [2:0]         cnt  [3:0]; // count per FIFO

  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i<4;i=i+1) begin cnt[i]<=0; in_ready[i]<=1; end
      out_valid<=0; out_data<=0;
    end else begin
      // enqueue
      for (i=0;i<4;i=i+1) begin
        if (in_valid[i] && in_ready[i]) begin
          fifo[i][cnt[i]] <= in_data[i];
          cnt[i] <= cnt[i] + 1;
        end
        // update in_ready based on depth
        in_ready[i] <= (cnt[i] < DEPTH);
      end

      // per-output round-robin arbitration (one output per port)
      // simple mapping: output j arbitrates among inputs i where destination matches j
      // here assume header encodes dest in upper 2 bits of flit
      for (i=0;i<4;i=i+1) begin
        out_valid[i] <= 0;
        out_data[i]  <= 0;
      end

      // naive round-robin across inputs with fairness pointer
      // pointer rotate per output (small state)
      // For brevity, implement fixed priority: choose lowest-index pending
      integer j;
      for (j=0;j<4;j=j+1) begin
        integer src;
        for (src=0;src<4;src=src+1) begin
          if (cnt[src]>0) begin
            // peek header to decide destination; here simple mapping: low 2 bits
            if (fifo[src][0][1:0] == j && out_ready[j]) begin
              out_valid[j] <= 1;
              out_data[j]  <= fifo[src][0];
              // dequeue shift
              integer k;
              for (k=0;k
\chapter{Chapter 18: Performance Analysis and Optimization}
\section{Section 1: Performance Metrics}
\subsection{Item 1:  Throughput (GFLOPS and TFLOPS)}
This subsection follows the chapter's metric taxonomy and moves from qualitative descriptions to the concrete arithmetic used to quantify GPU compute throughput and its relation to memory behavior. The exposition focuses on how to compute peak FLOPS, compare it to sustained measurements, and use those numbers to reason about architecture-level trade-offs.

Throughput for floating-point workloads is reported in FLOPS and commonly scaled to GFLOPS or TFLOPS. Architects distinguish peak theoretical throughput from sustained, application-level throughput. Peak throughput is an architectural property derived from the number of execution lanes, FMA (fused multiply-add) capability, and clock frequency. Define variables precisely and then give the canonical peak formula:
\begin{equation}[H]\label{eq:peak}
\text{Peak FLOPS} = N_{\mathrm{SM}}\times L\times R\times F
\end{equation}
where $N_{\mathrm{SM}}$ is the count of streaming multiprocessors (SMs or CUs), $L$ is the number of FP lanes (SIMD width) per SM, $R$ is the number of floating-point results produced per lane per cycle (e.g., $R=1$ for a single FMA), and $F$ is clock frequency in flops/s per result. Note that an FMA typically counts as two floating-point operations (multiply + add), so $F$ often includes this factor by setting $R\leftarrow R\times 2$.

Operational relevance: comparing peak to measured throughput identifies inefficiencies (memory stalls, divergence, poor ILP). Use the roofline arithmetic intensity threshold to classify bottlenecks:
\begin{equation}[H]\label{eq:arint}
I_{\mathrm{crit}}=\frac{\text{Peak FLOPS}}{B_{\mathrm{mem}}}
\end{equation}
where $B_{\mathrm{mem}}$ is sustained memory bandwidth in bytes/s. Workloads with arithmetic intensity $I
\subsection{Item 2:  Bandwidth utilization}
Continuing from throughput metrics such as GFLOPS and TFLOPS, bandwidth utilization ties raw computational capability to the memory system's ability to feed execution units (SMs, tensor cores, TMUs, ROPs) without starvation. Measuring and optimizing utilization exposes whether a workload is memory-bound or compute-bound and points to concrete microarchitectural fixes.

Bandwidth utilization quantifies how effectively the memory subsystem is used relative to its peak. Operationally this matters because many graphics and ML kernels are limited by data movement: texture fetches, scatter-gather for sparse tensors, or large GEMM input loads. Key contributors to utilization are access coalescing, request parallelism across memory channels and L2 slices, on-chip compression (DCC), and write-combine behavior at the ROP and DMA engines.

Analysis proceeds from counters exposed by PMUs: per-channel issued bytes, DRAM read/write cycles, L2 miss rate, and in-flight transaction counts. A compact metric is achieved bandwidth divided by theoretical peak:
\begin{equation}[H]\label{eq:bandwidth_util}
U \;=\; \frac{B_{\text{achieved}}}{B_{\text{peak}}}
\;=\; \frac{\sum_{i} \text{bytes\_issued}_i}{B_{\text{peak}}\cdot t}
\end{equation}
where $i$ indexes channels or memory controllers and $t$ is the sampling interval. For example, if an HBM3 stack delivers $1.2\ \mathrm{TB/s}$ peak and counters show $600\ \mathrm{GB}$ transferred in one second, $U=0.5$.

Practical measurement must correct for traffic amplification: cacheline fetches for narrow stores, read-for-ownership writebacks, and metadata from compression. Use derived counters to compute effective payload bytes versus protocol overhead. Latency hiding also affects interpretation: high utilization with long queue depths can indicate queuing bottlenecks rather than ideal streaming.

Implementation: sample a minimal set of counters and compute utilization and effective request size. The snippet below shows a simple analysis loop intended for on-host profiling tools that read hardware counters and report per-slice utilization and average request size.

\begin{lstlisting}[language=Python,caption={Bandwidth-utilization calculation from PMU counters},label={lst:bandwidth_calc}]
# read counters: bytes_issued_per_channel, cycles, peak_bw_per_channel
# counters returned as dictionaries keyed by channel id.
def compute_util(bytes_issued, sample_time_s, peak_bw_per_ch):
    util = {}
    for ch, b in bytes_issued.items():
        peak = peak_bw_per_ch[ch]  # bytes/sec
        util[ch] = b / (peak * sample_time_s)  # fraction
    total_util = sum(bytes_issued.values()) / (sum(peak_bw_per_ch.values()) * sample_time_s)
    return util, total_util

# Example: bytes_issued={'ch0':3e11,'ch1':3e11}, sample_time_s=1.0