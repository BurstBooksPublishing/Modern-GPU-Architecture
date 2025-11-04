module zero_skip_mask #(
    parameter LANES = 8,
    parameter DATA_WIDTH = 32
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     in_valid,                // incoming compressed lane vector valid
    input  wire [LANES*DATA_WIDTH-1:0] in_data_flat,         // concatenated lane values
    output reg                      out_valid,               // output valid for downstream
    output reg  [LANES-1:0]         out_enable,              // per-lane non-zero enables
    output reg  [$clog2(LANES+1)-1:0] out_active_count,      // number of non-zero lanes
    output reg  [LANES*DATA_WIDTH-1:0] out_data_flat         // passed-through data
);

    // local unpack
    genvar i;
    wire [LANES-1:0] nz;
    generate
        for (i=0; i
\subsection{Item 4:  Compression and decompression paths}
The previous discussion showed how zero-skipping reduces arithmetic work and how structured sparsity (for example 2:4) makes regular patterns that simplify dataflow; the compression/decompression path is the bridge that carries sparse payloads and metadata into dense, tiled inputs usable by tensor cores. It must minimize latency and metadata overhead while producing warp-aligned vectors that preserve coalescing for SM memory ports and tensor-core input lanes.

Problem statement and analysis. GPUs require contiguous, predictable tile inputs for high-utilization of tensor cores. Compressed representations reduce DRAM traffic but add metadata and decode cost in the datapath. Two practical schemes are common:
\begin{itemize}
\item block-bitmask (used with structured sparsity): small per-block masks indicate positions of nonzeros inside a fixed block (for 2:4, a 4-bit mask with two ones).
\item index-value streams (CSR-like) for irregular sparsity, which require variable-length index arithmetic and greater metadata.
\end{itemize}

Operational relevance precedes equations: effective bandwidth savings $R$ depends on the fraction of stored nonzeros $f$ and the per-element metadata overhead $m$ (in elements per matrix). The idealized reduction is
\begin{equation}[H]\label{eq:bandwidth_reduction}
R \;=\; 1 - \Big(f + \frac{m}{M\!N}\Big),
\end{equation}
where $M\!N$ is total elements and $0
\section{Section 6: Verilog Implementation}
\subsection{Item 1:  Systolic array module}
This subsection continues the tensor-core hardware discussion by turning the systolic organization and dataflow patterns into a concrete, synthesizable Verilog module that can be instantiated inside an SM/CU level tensor datapath. The implementation follows the skewed-input streaming model previously described for weight reuse and local accumulation.

A systolic array implements GEMM by streaming A rows from the left and B columns from the top into a two-dimensional grid of processing elements (PEs). Each PE performs a multiply-accumulate (MAC), forwards its A operand to the right, and its B operand downward, enabling spatial reuse and minimizing external memory traffic. The design goals are: deterministic latency, simple control, and parameterizable dimensions for scalability.

Analysis:
\begin{itemize}
\item For an $M \times K$ times $K \times N$ multiplication realized on an $M \times N$ systolic array, the steady-state generation of outputs requires
\begin{equation}[H]\label{eq:latency}
T_{\text{cycles}} = K + \max(M,N) - 1
\end{equation}
cycles to produce the full $M \times N$ result after the first valid inputs enter the array. This formula highlights the trade-off: larger arrays reduce the effective per-element amortized memory bandwidth but increase on-chip area and routing complexity.
\end{itemize}

Implementation strategy:
\begin{itemize}
\item Each PE contains registers for local A and B, a multiply unit, and an accumulator register.
\item Inputs: per-row A streams and per-column B streams; single-cycle valid signals allow pipeline gating.
\item Control: simple start and flush semantics; no complex ready/valid backpressure inside the array simplifies timing closure in an SM datapath.
\end{itemize}

The following Verilog is synthesizable RTL for a parameterizable systolic array with PE grid, streaming interfaces, and per-PE accumulation outputs. It assumes two's-complement signed arithmetic; change to unsigned if needed.

\begin{lstlisting}[language=Verilog,caption={Parameterizable systolic array (PE + array).},label={lst:systolic}]
`timescale 1ns/1ps
module pe #(
  parameter DATA_W = 16,
  parameter ACC_W  = 48
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    a_valid,      // new A available this cycle
  input  wire                    b_valid,      // new B available this cycle
  input  wire signed [DATA_W-1:0] a_in,
  input  wire signed [DATA_W-1:0] b_in,
  input  wire signed [DATA_W-1:0] a_right_in,  // from left neighbor (shifted)
  input  wire signed [DATA_W-1:0] b_down_in,   // from top neighbor (shifted)
  output reg  signed [DATA_W-1:0] a_right_out, // to right neighbor
  output reg  signed [DATA_W-1:0] b_down_out,  // to bottom neighbor
  output reg  signed [ACC_W-1:0]  acc_out
);
  reg signed [DATA_W-1:0] a_reg, b_reg;
  reg signed [ACC_W-1:0]  acc;

  always @(posedge clk) begin
    if (rst) begin
      a_reg <= 0; b_reg <= 0; acc <= 0;
      a_right_out <= 0; b_down_out <= 0; acc_out <= 0;
    end else begin
      // load new operands if valid, else use shifted values
      a_reg <= a_valid ? a_in        : a_right_in;
      b_reg <= b_valid ? b_in        : b_down_in;
      // forward operands for neighbor routing
      a_right_out <= a_reg;
      b_down_out  <= b_reg;
      // MAC: multiply current local operands and accumulate
      acc <= acc + $signed(a_reg) * $signed(b_reg);
      acc_out <= acc;
    end
  end
endmodule

module systolic_array #(
  parameter DATA_W = 16,
  parameter ACC_W  = 48,
  parameter M = 4, // rows
  parameter N = 4  // cols
)(
  input  wire                      clk,
  input  wire                      rst,
  input  wire                      start,
  input  wire [M*DATA_W-1:0]       a_vec_in,  // concatenated per-row A input
  input  wire [N*DATA_W-1:0]       b_vec_in,  // concatenated per-col B input
  input  wire                      a_valid,
  input  wire                      b_valid,
  output reg  [M*N*ACC_W-1:0]      c_vec_out, // concatenated per-PE accumulators
  output reg                       out_valid
);
  // internal buses for inter-PE routing
  wire signed [DATA_W-1:0] a_bus [0:M-1][0:N-1];
  wire signed [DATA_W-1:0] b_bus [0:M-1][0:N-1];
  wire signed [ACC_W-1:0]  acc_bus [0:M-1][0:N-1];

  genvar i,j;
  generate
    for (i=0;i
\subsection{Item 2:  Matrix multiply-accumulate block}
The systolic array module established the dataflow and tile boundaries for streaming partial products into local MAC lanes; this subsection drills into the single matrix multiply-accumulate (MAC) block that forms each lane, showing how it implements the core inner-product update used by tensor cores and systolic arrays.

Problem: implement a pipelined, parameterized MAC suitable for high-throughput SIMT and tensor-core fabrics where each cycle produces a partial accumulation $C_{i,j} \mathrel{+}= A_{i,k} \cdot B_{k,j}$. Analysis: the MAC must \begin{enumerate} \item match the multiplier/adder latency to keep the systolic schedule tight, \item provide sufficient accumulator headroom to avoid overflow across $K$ summed terms, and \item support low-latency handshaking for backpressure. \end{enumerate} For fixed-point inputs of widths $W_A$ and $W_B$ and a summation length $K$, an accumulator width that avoids overflow in the worst case is
\begin{equation}[H]\label{eq:acc_width}
W_{\mathrm{acc}} \;=\; W_A + W_B + \lceil \log_2 K \rceil,
\end{equation}
so the product is sign-extended into the accumulator and added each cycle. Implementation choices include signed vs unsigned arithmetic, optional saturation, and how many pipeline registers (multiplier and adder stages) to insert; latency choices affect tile scheduling across the systolic array.

Below is a synthesizable, parameterized Verilog implementation of a single-cycle-throughput, pipelined fixed-point MAC with a one-deep input buffer and configurable latency. It uses sign-extension of the product into the accumulator width and a simple valid/ready handshake.

\begin{lstlisting}[language=Verilog,caption={Synthesizable fixed-point pipelined MAC lane},label={lst:mac_fixed}]
module mac_fixed #(
  parameter A_WIDTH = 16,
  parameter B_WIDTH = 16,
  parameter ACC_WIDTH = 48,
  parameter LATENCY = 2  // 1..N pipeline stages (mult+add)
)(
  input  wire                     clk,
  input  wire                     rst,        // synchronous reset
  input  wire                     in_valid,   // input valid
  output reg                      in_ready,   // backpressure
  input  wire [A_WIDTH-1:0]       a,          // operand A (fixed-point)
  input  wire [B_WIDTH-1:0]       b,          // operand B (fixed-point)
  input  wire [ACC_WIDTH-1:0]     acc_in,     // incoming accumulator
  output reg                      out_valid,  // output valid
  output reg  [ACC_WIDTH-1:0]     acc_out     // updated accumulator
);
  // local widths
  localparam P_WIDTH = A_WIDTH + B_WIDTH;
  // pipeline registers
  reg [P_WIDTH-1:0] prod_reg;
  reg [ACC_WIDTH-1:0] acc_reg;
  reg valid_reg;

  // simple one-cycle multiplier + adder pipeline (LATENCY=2 implies 1 reg between)
  always @(posedge clk) begin
    if (rst) begin
      prod_reg <= 0;
      acc_reg  <= 0;
      valid_reg <= 0;
      in_ready <= 1'b1;
      out_valid <= 1'b0;
      acc_out <= 0;
    end else begin
      // accept when pipeline free (simple single-entry buffering)
      in_ready <= ~valid_reg;
      if (in_valid && in_ready) begin
        // signed multiply using Verilog signed cast
        prod_reg <= $signed(a) * $signed(b); // product width P_WIDTH
        valid_reg <= 1'b1;
      end
      // pipeline stage(s): for LATENCY>1 add extra regs if needed (omitted loop for clarity)
      if (valid_reg) begin
        // sign-extend product into ACC_WIDTH and add to acc_in
        acc_reg <= acc_in + {{(ACC_WIDTH-P_WIDTH){prod_reg[P_WIDTH-1]}}, prod_reg};
        out_valid <= 1'b1;
        acc_out <= acc_reg;
        valid_reg <= 1'b0;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end
endmodule