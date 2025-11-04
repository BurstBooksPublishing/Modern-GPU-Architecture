module display_dma_arbiter #(
  parameter N_CH = 4,
  parameter DEAD_W = 32
)(
  input  wire                     clk,
  input  wire                     rst,
  input  wire [N_CH-1:0]          req_valid,             // per-head request valid
  input  wire [N_CH*DEAD_W-1:0]   req_deadline_flat,     // concatenated deadlines MSB...LSB
  output reg  [N_CH-1:0]          grant                  // one-hot grant
);
  integer i;
  reg [DEAD_W-1:0] deadlines [0:N_CH-1];
  reg [DEAD_W-1:0] min_dead;
  reg [N_CH-1:0]   min_idx;
  reg [N_CH-1:0]   tie_mask;
  // unpack deadlines
  always @* begin
    for (i=0;i
\section{Section 2: Display Compression}
\subsection{Item 1:  Display Stream Compression (DSC)}
Following the display timing and pixel-pipeline considerations discussed in the display controller subsection, the Display Stream Compression (DSC) stage sits between the GPU's pixel pipeline and the external link (DisplayPort/HDMI PHY or internal panel interface). DSC's role is to reduce required link bandwidth while preserving visual fidelity, enabling higher resolutions, higher refresh rates, or fewer physical lanes in SM/MMIO-constrained implementations.

Problem — link bandwidth versus pixel throughput: a raw RGB frame at resolution $R_x \times R_y$ and pixel clock $f_{\mathrm{pix}}$ with $N$ bits per color channel requires raw link throughput
\begin{equation}[H]\label{eq:raw_bw}
B_{\mathrm{raw}} = f_{\mathrm{pix}}\cdot R_{\mathrm{bpp}} \quad\text{where}\quad R_{\mathrm{bpp}} = N_{\mathrm{chan}}\cdot N_{\mathrm{bits/channel}}.
\end{equation}
DSC reduces bandwidth by a content-dependent compression ratio CR, so the compressed bandwidth target is approximately $B_{\mathrm{comp}} \approx B_{\mathrm{raw}}/ \mathrm{CR}$, plus slice-header and packetization overhead: when encoding is slice-based, per-slice header costs inflate effective bits-per-pixel by $H_{\mathrm{slice}}/\mathrm{pixels\_per\_slice}$.

Analysis — algorithmic and architectural constraints: practical DSC implementations in GPUs must meet constant-rate constraints imposed by the serial link and maintain low, bounded latency for scanout. To accomplish that, hardware DSC encoders operate on fixed-size slices (rows or groups of pixels), perform a reversible color-space transform (to decorrelate channels), apply prediction to generate residuals, quantize and entropy-code the residuals, and finally pack slice headers and payload into transport packets. The encoder also runs an on-chip rate-control loop to meet the target bits-per-pixel (bpp) for each slice; this can be modeled as ensuring
\begin{equation}[H]\label{eq:rate_control}
\sum_{i\in\text{slice}} b_i \leq \text{pixels}_{\text{slice}}\cdot \text{bpp}_{\text{target}} - H_{\text{slice}}
\end{equation}
where $b_i$ are the coded symbol lengths. The GPU must provide sufficient FIFO depth and a deterministic token-budget to absorb short-term entropy bursts without underruning the link.

Implementation — datapath and controller: integrate the DSC encoder as a late-stage pixel pipeline block, immediately upstream of the display timing generator. Key hardware blocks:
\begin{itemize}
\item Slice input buffer (line buffering sized for worst-case slice pixels),
\item Color transform and predictor pipelines (pipelined for the pixel clock),
\item Quantizer and entropy coder (variable-latency; typically constrained to a few cycles via parallelism and pipelining),
\item Slice packetizer with constant-rate shaper feeding the PHY.
\end{itemize}
To illustrate a small synthesizable component of the encoder subsystem — a token-bucket rate shaper that gates variable-length codewords into fixed-bit link words — see the Verilog module below.

\begin{lstlisting}[language=Verilog,caption={Token-bucket based DSC rate shaper (synthesizable).},label={lst:rate_shaper}]
module dsc_rate_shaper #(
  parameter integer PIXEL_CLK_MHZ = 148,       // pixel clock in MHz
  parameter integer TARGET_BPP = 24,           // target bits per pixel (integer bpp*1)
  parameter integer SLICE_PIXELS = 3840        // pixels per slice
)(
  input  wire clk,
  input  wire reset_n,
  // input: encoded symbol length in bits for current pixel (variable-length)
  input  wire        sym_valid,
  input  wire [15:0] sym_bits,    // bits used to encode symbol (<= 16k)
  output reg         sym_accept,  // accepted for transmission this cycle
  // output: link word ready handshake (simplified)
  output reg         link_word_valid,
  input  wire        link_word_ready
);
  // budget accumulator in bits, fixed-point with microbit granularity
  localparam integer BUDGET_MAX = 32'd100000000;
  reg [31:0] budget_bits; // current token budget
  // initialize budget: pixels per second * bpp gives bits per second; scaled per cycle
  // For simplicity, add fixed budget per cycle = TARGET_BPP (approx) per pixel tick.
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      budget_bits <= 0;
      sym_accept  <= 0;
      link_word_valid <= 0;
    end else begin
      // replenish budget (simplified): add TARGET_BPP bits per pixel clock tick
      // In real design, use fractional accumulator synchronized to pixel clock domain.
      if (budget_bits + TARGET_BPP < BUDGET_MAX)
        budget_bits <= budget_bits + TARGET_BPP;
      else
        budget_bits <= BUDGET_MAX;
      // accept symbol only if budget suffices
      if (sym_valid && sym_bits <= budget_bits) begin
        sym_accept <= 1'b1;
        budget_bits <= budget_bits - sym_bits;
        // produce link word (simplified packetization)
        if (link_word_ready) link_word_valid <= 1'b1;
        else link_word_valid <= 1'b0;
      end else begin
        sym_accept <= 1'b0;
        link_word_valid <= 1'b0;
      end
    end
  end
endmodule