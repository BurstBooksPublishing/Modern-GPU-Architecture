module adder_tree #(
  parameter WIDTH = 32,
  parameter N = 8  // must be power of two
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 in_valid,
  input  wire [N*WIDTH-1:0]   in_data,  // packed N inputs
  output reg                  out_valid,
  output reg  [WIDTH+$clog2(N)-1:0] out_sum
);
  // stage registers: stage s has N/(2^s) lanes
  localparam STAGES = $clog2(N);
  // unpack inputs
  wire [WIDTH-1:0] in_vec [0:N-1];
  genvar i, s;
  generate for (i=0;i
\subsection{Item 4:  Precision and throughput balance}
The previous subsections established systolic dataflow and the microarchitecture of matrix multiply-accumulate units, so this subsection examines how operand precision choices map to tensor-core throughput and accumulator sizing. We connect dataflow capacity and MAC density to both numerical fidelity and hardware area, then give a small synthesizable example illustrating tradeoffs.

Precision decisions directly influence MAC unit area, clock-handling, and required accumulator width, which in turn set achievable GFLOPS/TFLOPS for an SM or CU. For fixed silicon area, a simple engineering model treats multiplier area as growing approximately with the square of operand bitwidth; therefore the count of MACs per tile scales inversely:
\begin{equation}[H]\label{eq:throughput_scaling}
T(b) \approx \frac{K}{b^{2}},
\end{equation}
where $T$ is relative throughput, $b$ is operand bitwidth in bits, and $K$ is a proportionality constant that folds frequency and area budget. This explains why lowering from FP32 to FP16 or INT8 yields superlinear increases in MAC count per tile.

Accumulator sizing is a distinct constraint. For integer fixed-point accumulation of $N$ terms each $b$ bits, exact sum bit growth requires
\begin{equation}[H]\label{eq:acc_bits}
w_{\text{acc}} = b + \lceil \log_{2} N \rceil.
\end{equation}
Floating-point accumulators behave differently: rounding error accumulates rather than simple overflow, so GPUs commonly use wider accumulators (e.g., FP32 or TF32 accumulators for FP16 inputs) to bound relative error.

Operational implications for tensor-core pipelines:
\begin{itemize}
\item Mixed precision: Use narrow input multipliers with wider accumulators to maximize MAC density while retaining convergence for ML workloads. This pattern appears in practice: FP16 inputs with FP32 accumulation often hit similar model quality as full FP32.
\item Quantized inference: INT8/INT4 pipelines can pack more parallel MACs and use 32-bit accumulators to avoid overflow in deep reductions.
\item Frequency and pipeline depth: Narrower datapaths allow shorter combinational delays, enabling higher clock frequency or fewer pipeline stages for lower latency.
\end{itemize}

To illustrate a synthesizable hardware primitive useful in accelerator tile design, below is a parameterized fixed-point MAC implemented for synthesis. It shows how changing \lstinline|INPUT_WIDTH| and \lstinline|ACC_WIDTH| affects storage and datapath widths, modeling area/perf tradeoffs in RTL.

\begin{lstlisting}[language=Verilog,caption={Parameterized fixed-point MAC with selectable accumulator width},label={lst:mac_verilog}]
module mac_sync #(
  parameter INPUT_WIDTH = 8,   // operand width (bits)
  parameter ACC_WIDTH   = 32   // accumulator width (bits)
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     in_valid,
  input  wire signed [INPUT_WIDTH-1:0] a, // multiplicand
  input  wire signed [INPUT_WIDTH-1:0] b, // multiplier
  input  wire signed [ACC_WIDTH-1:0]     acc_in, // current accumulator
  output reg  signed [ACC_WIDTH-1:0]     acc_out,
  output reg                      out_valid
);
  // pipelined multiply (combinational then registered)
  wire signed [2*INPUT_WIDTH-1:0] product = a * b;
  reg  signed [2*INPUT_WIDTH-1:0] product_r;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      product_r <= 0;
      acc_out   <= 0;
      out_valid <= 0;
    end else begin
      product_r <= product; // stage 1 register
      if (in_valid) begin
        // sign-extend product into accumulator width before add
        acc_out <= acc_in + $signed({{(ACC_WIDTH-2*INPUT_WIDTH){product_r[2*INPUT_WIDTH-1]}}, product_r});
        out_valid <= 1'b1;
      end else begin
        out_valid <= 1'b0;
      end
    end
  end
endmodule