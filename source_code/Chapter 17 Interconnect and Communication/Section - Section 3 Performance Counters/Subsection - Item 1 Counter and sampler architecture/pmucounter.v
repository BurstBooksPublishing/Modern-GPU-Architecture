module pmu_counter_sampler #(
  parameter NUM_EVENTS = 64,
  parameter NUM_CNTRS  = 8,
  parameter WIDTH      = 48
)(
  input  wire                 clk,
  input  wire                 rstn,
  input  wire [NUM_EVENTS-1:0] event_bus,   // one-hot event signals
  // config interface (simple register write)
  input  wire                 cfg_we,
  input  wire [7:0]           cfg_addr,     // addr space: 0..NUM_CNTRS-1 => event sel, 128 => sample_period
  input  wire [WIDTH-1:0]     cfg_wdata,
  // sampler output
  output reg                  sample_valid,
  output reg [NUM_CNTRS*WIDTH-1:0] sample_data,
  input  wire                 sample_ack
);

  // per-counter event select (encoded index into event_bus)
  reg [$clog2(NUM_EVENTS)-1:0] evt_sel [0:NUM_CNTRS-1];
  reg [WIDTH-1:0] counter [0:NUM_CNTRS-1];
  reg [NUM_CNTRS-1:0] overflow;

  // sampling period register (cycles)
  reg [31:0] sample_period;
  reg [31:0] sample_cnt;

  integer i;
  // config writes
  always @(posedge clk) begin
    if (!rstn) begin
      sample_period <= 32'd1000;
      for (i=0;i increment.
      for (i=0;i= log2(max_event_rate * T_s). Overflow flags must be monitored for configuration adjustments.
\subsection{Item 2: Multiplexing techniques}
Building on the sampler and counter buffering described previously, we now examine practical multiplexing strategies that let a limited set of hardware counters measure a much larger set of SM-, TMU-, ROP-, tensor-core and memory events without overwhelming area or crossbar bandwidth.

Multiplexing problem: modern GPUs expose hundreds of distinct events per SM but cannot afford one full-width physical counter per event. The goal is to trade temporal resolution and software complexity for hardware resource efficiency while preserving statistical usefulness for bottleneck analysis. Techniques fall into three operational categories:
\begin{itemize}
\item Time-division (round-robin) multiplexing: cycle a physical counter through selected event lines, allocating a fixed timeslot per event. This minimizes logic by reusing one counter but introduces quantization error and potential aliasing for bursty events.
\item Event-triggered sampling: latch counters only when a trigger condition occurs (e.g., warp retire rate drop). This reduces average overhead but needs programmable triggers and edge-detection hardware.
\item Hierarchical aggregation: local per-SM short-term counters are multiplexed into an SM aggregator which periodically sends compact summaries to a global PMU fabric, reducing cross-chip bandwidth.
\end{itemize}

Analysis: For round-robin multiplexing with $N$ events and a timeslot of $T$ cycles per event at clock frequency $f_{\mathrm{clk}}$, the per-event sampling frequency equals
\begin{equation}[H]\label{eq:sample_rate}
f_{\mathrm{sample}}=\frac{f_{\mathrm{clk}}}{N\cdot T}.
\end{equation}
Lowering $T$ increases temporal resolution but increases overhead on the crossbar and counter writebacks. Bursty SIMT behavior (e.g., divergence spikes) requires $T$ small enough to avoid undercounting short-lived phenomena.

Implementation example: a synthesizable Verilog PMU multiplexer that performs configurable round-robin sampling and writes per-event shadow counters. It supports a software-readable shadow array and a programmable timeslot width.

\begin{lstlisting}[language=Verilog,caption={PMU round-robin multiplexer — synthesizable},label={lst:pmu_mux}]
module pmu_mux #(
  parameter NUM_EVENTS = 32,
  parameter CNT_WIDTH  = 48,
  parameter T_BITS     = 8            // timeslot size = 2^T_BITS cycles
)(
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire [NUM_EVENTS-1:0]   event_pulse,   // one-cycle pulses per source
  input  wire [T_BITS-1:0]       timeslot_log,  // programmable timeslot exponent
  input  wire [$clog2(NUM_EVENTS)-1:0] rd_idx,   // read index
  output reg  [CNT_WIDTH-1:0]    rd_data,
  output reg                     rd_valid
);
  // per-event shadow counters (compact storage)
  reg [CNT_WIDTH-1:0] counters [0:NUM_EVENTS-1];
  reg [$clog2(NUM_EVENTS)-1:0] sel;                // current selected event
  reg [T_BITS-1:0] timeslot_cnt;

  integer i;
  // initialize counters
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sel <= 0;
      timeslot_cnt <= 0;
      rd_data <= 0;
      rd_valid <= 0;
      for (i=0;i
\subsection{Item 3:  Key hardware metrics}
Building on the sampler architecture and multiplexing strategies discussed earlier, we now map those mechanisms to the concrete metrics you should collect and how each metric ties to a specific bottleneck in an SM/CU. This subsection translates abstract counter design into actionable measures for graphics and ML workloads.

Problem: choose a compact set of counters that reveals whether a kernel is compute-bound, memory-bound, or limited by control flow or resources. Analysis: counters must capture (a) instruction and functional-unit utilization, (b) memory-system behavior across hierarchies, (c) control and warp-level inefficiencies, and (d) thermal/power headroom. Key hardware metrics to expose:

\begin{itemize}
\item Compute throughput:
\begin{enumerate}
\item SM active cycles and SM warp-issued ($\mathrm{warp\_issue\_count}$) — show SIMT issue utilization.
\item FP32/FP16/INT operations retired per cycle and tensor-core usage — directly measures arithmetic throughput.
\item ALU/TMU/RT/ROP occupancy breakdown — isolates unit-specific saturation.
\end{enumerate}

\item Memory and cache:
\begin{enumerate}
\item DRAM bytes read/write and PCIe transfers.
\item L1, L2 hit/miss counters and request latencies.
\item Texture (TMU) and ROP bandwidth counters.
\item Shared memory bank conflict events.
\end{enumerate}

\item Scheduling and control:
\begin{enumerate}
\item Active warps per SM and occupancy (fraction of maximum warps active).
\item Warp divergence events, reconverge latency.
\item Stalled cycles by reason: memory, execution, dispatch, or synchronization.
\end{enumerate}

\item System-level:
\begin{enumerate}
\item Power draw and thermal throttling samples (NVML/CUPTI).
\item Context-switch and kernel-launch overhead counters.
\end{enumerate}
\end{itemize}

Operational relevance before equations: occupancy and roofline arithmetic-intensity thresholds are essential for diagnosing whether adding more compute units or higher bandwidth yields performance.

Let $\mathrm{RegsPerSM}$, $\mathrm{SharedPerSM}$, and $\mathrm{MaxBlocksPerSM}$ be hardware limits; for a kernel with $\mathrm{RegsPerThread}$, $\mathrm{SharedPerBlock}$, $\mathrm{ThreadsPerBlock}$, the active blocks per SM is
\begin{equation}[H]\label{eq:active_blocks}
B_{\mathrm{active}}=\min\left(\left\lfloor\frac{\mathrm{RegsPerSM}}{\mathrm{RegsPerThread}\cdot\mathrm{ThreadsPerBlock}}\right\rfloor,\left\lfloor\frac{\mathrm{SharedPerSM}}{\mathrm{SharedPerBlock}}\right\rfloor,\mathrm{MaxBlocksPerSM}\right).
\end{equation}
Occupancy is then
\begin{equation}[H]\label{eq:occupancy}
\mathrm{Occupancy}=\frac{B_{\mathrm{active}}\cdot\mathrm{WarpsPerBlock}}{\mathrm{MaxWarpsPerSM}}.
\end{equation}

Roofline tie-in: sustained $\mathrm{GFLOPS} = \min(\mathrm{peak\_GFLOPS}, \mathrm{BW\_bytes/s} \cdot \mathrm{arithmetic\_intensity})$. Use counters to compute $\mathrm{arithmetic\_intensity} = \mathrm{FLOPs} / \mathrm{DRAM\_bytes}$.

Implementation: sample both high-rate streaming counters (SM cycles, issued warps) and multiplexed detailed counters (L1 misses, tensor-core events). Use coarse sampling for power/thermal and high-rate ring buffers for per-kernel counters. Example: quick NVML poll for SM/GPU utilization and memory throughput.

\begin{lstlisting}[language=Python,caption={Poll lightweight GPU metrics via NVML},label={lst:nvml}]
import pynvml, time
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
for _ in range(10):
    util = pynvml.nvmlDeviceGetUtilizationRates(h)     # GPU and memory util
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)            # total/used/free
    power = pynvml.nvmlDeviceGetPowerUsage(h)         # milliwatts
    print(util.gpu, util.memory, mem.used, power)     # simple inline comments
    time.sleep(0.2)
pynvml.nvmlShutdown()