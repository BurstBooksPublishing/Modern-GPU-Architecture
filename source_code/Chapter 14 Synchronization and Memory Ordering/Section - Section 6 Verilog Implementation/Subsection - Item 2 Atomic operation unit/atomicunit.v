module atomic_unit #(
  parameter ADDR_W = 40,
  parameter DATA_W = 64,
  parameter TAG_W  = 8,
  parameter RESV_N = 8,
  parameter FIFO_D  = 16
)(
  input  wire                    clk,
  input  wire                    rst,
  // request: valid/ready, single-beat
  input  wire                    req_valid,
  output reg                     req_ready,
  input  wire [ADDR_W-1:0]       req_addr,
  input  wire [2:0]              req_op,    // 0=read,1=add,2=and,3=or,4=xor,5=cas
  input  wire [DATA_W-1:0]       req_data,  // for RMW or CAS (expected)
  input  wire [TAG_W-1:0]        req_tag,
  // response
  output reg                     resp_valid,
  output reg  [DATA_W-1:0]       resp_data,
  output reg  [TAG_W-1:0]        resp_tag,
  // memory interface (simple request/response)
  output reg                     mem_req_valid,
  input  wire                    mem_req_ready,
  output reg  [ADDR_W-1:0]       mem_req_addr,
  output reg  [DATA_W-1:0]       mem_req_wdata,
  output reg  [2:0]              mem_req_op, // 0=read,1=write,2=atomic
  input  wire                    mem_resp_valid,
  input  wire [DATA_W-1:0]       mem_resp_rdata
);

// Simple FIFO for incoming requests (circular buffer)
reg [ADDR_W-1:0] fifo_addr [0:FIFO_D-1];
reg [DATA_W-1:0] fifo_data [0:FIFO_D-1];
reg [2:0]        fifo_op   [0:FIFO_D-1];
reg [TAG_W-1:0]  fifo_tag  [0:FIFO_D-1];
reg [4:0]        fifo_wr_ptr, fifo_rd_ptr; // depth up to 32
reg [5:0]        fifo_cnt;

always @(posedge clk) begin
  if (rst) begin
    fifo_wr_ptr <= 0; fifo_rd_ptr <= 0; fifo_cnt <= 0; req_ready <= 1;
  end else begin
    // enqueue
    if (req_valid && req_ready) begin
      fifo_addr[fifo_wr_ptr] <= req_addr;
      fifo_data[fifo_wr_ptr] <= req_data;
      fifo_op[fifo_wr_ptr]   <= req_op;
      fifo_tag[fifo_wr_ptr]  <= req_tag;
      fifo_wr_ptr <= fifo_wr_ptr + 1;
      fifo_cnt <= fifo_cnt + 1;
    end
    // backpressure when full
    req_ready <= (fifo_cnt < FIFO_D-1);
    // dequeue happens below when issuing mem or matching reservation
  end
end

// Reservation table (linear search associative)
reg valid_r [0:RESV_N-1];
reg [ADDR_W-1:0] addr_r [0:RESV_N-1];
reg [DATA_W-1:0] accum_r [0:RESV_N-1];
reg [TAG_W-1:0]  head_tag [0:RESV_N-1];

integer i;
always @(posedge clk) begin
  if (rst) begin
    for (i=0;i0) begin
      reg [ADDR_W-1:0] a; reg [DATA_W-1:0] d; reg [2:0] op; reg [TAG_W-1:0] t;
      a = fifo_addr[fifo_rd_ptr]; d = fifo_data[fifo_rd_ptr]; op = fifo_op[fifo_rd_ptr]; t = fifo_tag[fifo_rd_ptr];
      // find matching reservation
      integer hit; hit = -1;
      for (i=0;i
\subsection{Item 3:  Memory fence controller}
The fence controller follows naturally from the atomic unit's ability to order single read-modify-write primitives and the barrier module's group rendezvous: it must convert software fence requests into observable memory-quiescent points across caches and outstanding transactions. The following explains the problem, analyzes scope-locality tradeoffs, and presents a synthesizable Verilog implementation that integrates with transaction trackers and cache-drain handshakes.

Problem and analysis. GPUs require multiple fence scopes: within-SM ($L_0$/shared memory), $L_1$/$L_2$ (chip-local), and device/system (cross-GPU or CPU-visible). A fence must complete only after (a) all prior memory operations at the issuing scope have globally completed and (b) any store buffers, write-combines, and cache lines pending eviction are drained. Measuring completion requires counting outstanding transactions and coordinating with cache controllers. We track:
\begin{itemize}
\item per-SM outstanding operations for local fences,
\item per-scope outstanding counts for broader scopes,
\item a FIFO of pending fence requests to serialize completion.
\end{itemize}

If $N$ outstanding transactions each have worst-case latency $L$ cycles, a conservative bound on drain latency is
\begin{equation}[H]\label{eq:drain_latency}
T_{\text{drain}} \approx N \cdot L,
\end{equation}
which motivates both reducing $N$ by early coalescing and limiting fence frequency in tight loops.

Implementation strategy.
\begin{enumerate}
\item Accept fence requests with scope and SM identifier.
\item Maintain counters: per-SM and per-scope. Increment/decrement via explicit transaction\_inc/transaction\_dec signals from load/store pipelines and cache controllers.
\item For each queued fence, assert a cache-drain request and wait until the relevant counters are zero and cache\_drained is returned.
\item Provide fence\_done and ack handshake to the issuing SM.
\end{enumerate}

The following Verilog is synthesizable and parameterized for NUM\_SM and FIFO\_DEPTH. It assumes external modules assert transaction\_inc/dec with matching scope bits.

\begin{lstlisting}[language=Verilog,caption={Memory fence controller (synthesizable)},label={lst:fence_ctrl}]
module mem_fence_ctrl #(
    parameter NUM_SM = 16,
    parameter NUM_SCOPES = 4,          // 0:SM,1:L1,2:L2,3:DEVICE
    parameter CNT_WIDTH = 12,
    parameter FIFO_DEPTH = 8,
    parameter SM_ID_WIDTH = 4,
    parameter PTR_WIDTH = $clog2(FIFO_DEPTH)
) (
    input  wire clk,
    input  wire rst,

    // Fence request from SM
    input  wire                fence_req_valid,
    input  wire [SM_ID_WIDTH-1:0] fence_req_sm,
    input  wire [1:0]          fence_req_scope, // encoded scope
    output reg                 fence_req_ack,
    output reg                 fence_done,      // asserted when fence completes
    output reg [SM_ID_WIDTH-1:0] fence_done_sm,

    // Transaction tracker interface (from load/store/caches)
    input  wire                tx_inc_valid,
    input  wire [1:0]          tx_inc_scope,
    input  wire [SM_ID_WIDTH-1:0] tx_inc_sm,
    input  wire                tx_dec_valid,
    input  wire [1:0]          tx_dec_scope,
    input  wire [SM_ID_WIDTH-1:0] tx_dec_sm,

    // Cache drain handshake for scopes (one hot on scope)
    output reg  [NUM_SCOPES-1:0] drain_req,
    input  wire [NUM_SCOPES-1:0] drain_done
);

    // Outstanding counters per scope and per-SM (synthesizable arrays)
    reg [CNT_WIDTH-1:0] scope_count [0:NUM_SCOPES-1];
    reg [CNT_WIDTH-1:0] sm_count    [0:NUM_SM-1];

    // Simple FIFO for fence requests
    reg [SM_ID_WIDTH-1:0] fifo_sm [0:FIFO_DEPTH-1];
    reg [1:0]             fifo_scope [0:FIFO_DEPTH-1];
    reg [PTR_WIDTH-1:0]   fifo_head, fifo_tail;
    reg [PTR_WIDTH:0]     fifo_cnt;

    integer i;
    // counters updates
    always @(posedge clk) begin
        if (rst) begin
            for (i=0;i
\subsection{Item 4:  Synchronization testbench}
The testbench below continues verification of the memory fence controller and the atomic operation unit by exercising ordering, contention, and barrier completion across SM threads and divergent warp-like sequences. It focuses on realistic GPU behaviors: concurrent fetch_add/CAS sequences, acquire/release fence ordering, and barrier completion under interleaved arrivals.

Problem and analysis: we must prove that (1) atomic RMWs serialize correctly against concurrent accesses, (2) fences enforce the intended visibility windows across loads/stores, and (3) the barrier releases all waiting threads exactly once per epoch. The formal barrier condition for a block of size $B$ is
\begin{equation}[H]\label{eq:barrier}
\sum_{i=0}^{B-1} arrived_i = B \quad\Longrightarrow\quad \text{release\_pulse}
\end{equation}
where each $arrived_i$ is a single-shot arrival indicator for thread $i$. The testbench must saturate the atomic unit with contending updates while interposing fences to verify global ordering effects.

Implementation: the testbench instantiates compact, synthesizable models of a barrier and an atomic unit, then creates a configurable number of stimulus threads which:
\begin{enumerate}
\item issue randomized atomic operations (fetch_add and CAS) to shared addresses to produce contention and expose lost-update bugs;
\item perform load/store sequences with optional acquire or release fence semantics to check write visibility ordering;
\item assert barrier arrival at arbitrary cycles to validate the counter-reset and single-cycle release pulse.
\end{enumerate}

The Verilog below is a complete simulation harness and synthesizable DUT models for small-scale FPGA emulation. Comments inside code are concise.

\begin{lstlisting}[language=Verilog,caption={Synchronization testbench exercising barrier, fences, and atomic unit},label={lst:sync_tb}]
`timescale 1ns/1ps
module barrier #(parameter B=8)(input clk, input rst,
    input [clog2(B)-1:0] tid, input arrive, output reg release);
  // simple counter barrier: when count==B assert release one cycle
  integer i;
  reg [clog2(B):0] count;
  reg arrived_flag [0:B-1];
  always @(posedge clk) begin
    if (rst) begin
      count <= 0; release <= 0;
      for (i=0;i
\chapter{Chapter 15: Advanced Rendering Features}
\section{Section 1: Tessellation Pipeline}
\subsection{Item 1:  Hull shader and control points}
Building on vertex processing and primitive assembly principles, the hull stage converts patches and their control points into tessellation factors that the fixed-function tessellator consumes. The following explains operational behavior, latency and resource effects, and a reference HLSL implementation for a triangle patch.

The problem: expose sufficient geometric detail via hardware tessellation while minimizing per-patch overhead and preserving SIMT efficiency. Analysis: a hull shader has two phases. The control-point phase runs once per control point per patch (InvocationCount equals the output control-point count), producing per-control-point outputs stored in an OutputPatch. The patch-constant function runs once per patch and emits tessellation factors: outer factors (for edges) and inner factors (for interiors). Hardware then rasterizes based on the tessellator's partitioning mode (integer, fractional\_odd, fractional\_even) and domain type (isolines, tri, quad).

Key operational points:
\begin{itemize}
\item Execution model: the control-point invocations map to adjacent threads within a thread-group (SIMT lane grouping). Shared local memory (LDS/shared) or register spilling holds the OutputPatch until the constant function executes.
\item Synchronization: the hull stage implicitly requires all control-point invocations to complete before the patch-constant function reads the patch. HLSL guarantees this ordering; on the hardware side this implies a small barrier and potential thread divergence cost if InvocationCount is large.
\item Tessellation factor semantics: outer factors feed edge subdivision counts; inner factors feed interior subdivision. For triangle domain, positions in the domain shader are computed by barycentric interpolation of control-point outputs:
\end{itemize}
\begin{equation}\label{eq:dom_pos}
P(\alpha,\beta,\gamma)=\alpha P_0+\beta P_1+\gamma P_2,\quad \alpha+\beta+\gamma=1.
\end{equation}
For isolines, a common mapping for edge segments is $n_{\text{segments}}=\max(1,\lceil T_{\text{edge}}\rceil)$, clamped to hardware maximum.

Reference HLSL hull shader (triangle patch, pass-through control points, adaptive tessellation factor based on edge length):
\begin{lstlisting}[language=HLSL,caption={Triangle hull shader with patch constant function},label={lst:hs_example}]
struct VS_OUT { float3 pos : POSITION; }; // vertex shader output

[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("HSConst")] 
HS_OUT HSMain(InputPatch patch, uint cpId : SV_OutputControlPointID)
{
    HS_OUT outCP;
    outCP.pos = patch[cpId].pos; // pass-through per-control-point
    return outCP;
}

struct HS_PatchConst { float edges[3] : SV_TessFactor; float inside : SV_InsideTessFactor; };

HS_PatchConst HSConst(InputPatch patch)
{
    HS_PatchConst pc;
    // compute geometric edge lengths in patch space (cheap L2)
    float3 p0 = patch[0].pos, p1 = patch[1].pos, p2 = patch[2].pos;
    float e0 = length(p1 - p0); // edge 0
    float e1 = length(p2 - p1); // edge 1
    float e2 = length(p0 - p2); // edge 2
    // map length to tess factor, clamp to hardware limits (1..64)
    pc.edges[0] = clamp(e0 * 8.0, 1.0, 64.0);
    pc.edges[1] = clamp(e1 * 8.0, 1.0, 64.0);
    pc.edges[2] = clamp(e2 * 8.0, 1.0, 64.0);
    pc.inside   = clamp((e0+e1+e2)/3.0 * 4.0, 1.0, 64.0);
    return pc;
}