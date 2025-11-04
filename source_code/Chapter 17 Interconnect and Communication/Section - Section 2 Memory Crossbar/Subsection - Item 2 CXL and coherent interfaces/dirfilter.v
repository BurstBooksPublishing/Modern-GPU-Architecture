module cxl_dir_filter #(
  parameter ENTRIES = 16,
  parameter TAGW    = 48,
  parameter IDXW    = 4
)(
  input                 clk,
  input                 rst,
  input                 req_valid,    // request present
  input                 req_rw,       // 0=read,1=write
  input  [TAGW-1:0]     req_tag,      // line tag
  output reg            hit,
  output reg            grant_valid,  // grant to L2 or fabric
  output reg [IDXW-1:0] grant_idx     // matched/alloc index
);
  reg [TAGW-1:0] tags [0:ENTRIES-1];
  reg           val  [0:ENTRIES-1];
  integer i;
  reg [IDXW-1:0] alloc_ptr;

  // parallel compare (combinational)
  wire [ENTRIES-1:0] match;
  genvar gi;
  generate for (gi=0; gi
\subsection{Item 3:  PHY design and equalization}
The previous discussion established system-level link training and coherence semantics used by host fabrics and die-to-die links; here we focus on the PHY-level mechanisms that realize those logical guarantees by shaping signals and recovering data on noisy, frequency-dependent channels.

PHY design must solve inter-symbol interference (ISI), frequency-dependent loss, and time-varying crosstalk while preserving latency and throughput targets for graphics and ML traffic. Practical PHY blocks include transmitter pre-emphasis (FFE), continuous-time linear equalizers (CTLE) in the RX front end, and decision feedback equalizers (DFE) or adaptive FFE in the digital domain. The engineering problem is to trade silicon area, power, and convergence time against residual bit-error-rate (BER) at target data rates and modulation orders (NRZ or PAM-N). Convergence and stability are typically handled by an adaptive algorithm; a simple, robust choice is the least-mean-squares (LMS) rule used in decision-directed or training modes. Operationally, training mode uses a known sequence for initial convergence, then switches to decision-directed LMS for tracking.

Before showing an implementation sketch, the LMS update for tap vector $w$ reads:
\begin{equation}[H]\label{eq:lms}
w_{n+1} = w_n + \mu \; e_n \; x_n
\end{equation}
where $e_n$ is the error (decision minus equalized output), $x_n$ is the tap input sample vector, and $\mu$ is a small positive step-size controlling stability and convergence speed. For fixed-point hardware, $\mu$ is implemented as a power-of-two reciprocal (shift) to avoid costly multiplies.

A compact synthesizable Verilog module below implements a fixed-point multi-tap LMS equalizer suitable for a GPU link PHY training engine. It uses signed arithmetic, shift-based step size, and a serial-shift sample buffer. This module targets the digital equalizer stage that follows a CTLE and ADC; it is deliberately parameterized for tap count and bit widths.

\begin{lstlisting}[language=Verilog,caption={Fixed-point multi-tap LMS equalizer (synthesizable).},label={lst:phy_lms}]
module phy_eq_lms #(
  parameter integer TAPS = 5,
  parameter integer SAMPLE_W = 10,
  parameter integer TAP_W = 16,
  parameter integer MU_SHIFT = 4  // mu = 1/2^MU_SHIFT
)(
  input  wire clk,
  input  wire rst,                 // sync reset
  input  wire train_en,            // 1 during training or tracking
  input  wire signed [SAMPLE_W-1:0] rx_sample, // new ADC sample
  input  wire signed [SAMPLE_W-1:0] ref_symbol, // known symbol in training or decision
  input  wire decision_mode,       // 1 = decision-directed, 0 = training
  output reg  signed [SAMPLE_W-1:0] eq_out
);
  // tap storage
  reg signed [TAP_W-1:0] taps [0:TAPS-1];
  reg signed [SAMPLE_W-1:0] shift_reg [0:TAPS-1];
  integer i;
  // compute equalized output y = sum taps * x (accumulate with extra width)
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i0;i=i-1) shift_reg[i] <= shift_reg[i-1];
      shift_reg[0] <= rx_sample;
      // compute Y
      reg signed [TAP_W+SAMPLE_W+4:0] acc;
      acc = 0;
      for (i=0;i>> (TAP_W-1); // simple normalization
      // determine error e = reference - y (training) or decision - y (DD)
      reg signed [SAMPLE_W-1:0] ref;
      ref = (decision_mode) ? ref_symbol : ref_symbol; // placeholder; device supplies decision when DD
      reg signed [SAMPLE_W+8:0] err;
      err = $signed(ref) - $signed(eq_out);
      // LMS update when allowed
      if (train_en) begin
        for (i=0;i> MU_SHIFT
          reg signed [TAP_W+SAMPLE_W+8:0] delta;
          delta = (err * $signed(shift_reg[i])) >>> MU_SHIFT;
          taps[i] <= taps[i] + delta[TAP_W-1:0];
        end
      end
    end
  end
endmodule