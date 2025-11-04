module delta_rle_enc #(
  parameter PIXW = 10, // bits per pixel channel
  parameter BUSW = 16  // output bus width
)(
  input  clk, input rstn,
  input  valid_in, input [PIXW-1:0] px_in,
  output reg ready_in,
  output reg valid_out, output reg [BUSW-1:0] data_out
);
  // simple delta predictor and run-length (counts up to 2^BUSW-1)
  reg [PIXW-1:0] prev;
  reg [BUSW-1:0] run;
  reg state;
  always @(posedge clk) begin
    if (!rstn) begin prev <= 0; run <= 0; state <= 0; valid_out <= 0; ready_in <= 1; end
    else begin
      valid_out <= 0;
      if (valid_in && ready_in) begin
        if (px_in == prev) begin
          run <= run + 1;
        end else begin
          // flush run if any
          if (run != 0) begin valid_out <= 1; data_out <= {1'b1, run[BUSW-2:0]}; run <= 0; end
          // emit delta token: leading 0 then signed diff
          valid_out <= 1; data_out <= {1'b0, px_in[PIXW-2:0]}; // simplified
          prev <= px_in;
        end
      end
    end
  end
endmodule

module delta_rle_dec #(
  parameter PIXW = 10, parameter BUSW = 16
)(
  input clk, input rstn,
  input valid_in, input [BUSW-1:0] data_in, output reg ready_in,
  output reg valid_out, output reg [PIXW-1:0] px_out
);
  reg [PIXW-1:0] prev;
  always @(posedge clk) begin
    if (!rstn) begin prev<=0; valid_out<=0; ready_in<=1; end
    else begin
      valid_out <= 0;
      if (valid_in && ready_in) begin
        if (data_in[BUSW-1]) begin // run token
          integer i;
          for (i=0;i
\subsection{Item 3:  Bandwidth reduction analysis}
The previous subsections established DSC's microblock prediction, encoder/decoder latency constraints, and practical hardware partitioning for real-time display streams. Here we quantify how those design choices translate into link reduction, buffer sizing, and encoder throughput requirements for modern GPU display engines.

Problem statement and analysis: the display pipeline must deliver pixels at scanout rate while minimizing VRAM and link bandwidth. Uncompressed pixel bandwidth for a frame is proportional to horizontal and vertical resolution, refresh rate, and bits per pixel. For an uncompressed stream the bit-rate is
\begin{equation}[H]\label{eq:uncompressed_bw}
B_{\mathrm{raw}} = H \cdot V \cdot f_{\mathrm{r}} \cdot b_{\mathrm{pp}},
\end{equation}
where $H,V$ are pixel counts, $f_{\mathrm{r}}$ is refresh frequency, and $b_{\mathrm{pp}}$ is bits per pixel. Practical designs add a small protocol and blanking overhead; account for that as a multiplicative factor $(1+\alpha)$.

With compression ratio $R$ (for DSC-targeted content), effective bandwidth becomes
\begin{equation}[H]\label{eq:compressed_bw}
B_{\mathrm{link}} = \frac{B_{\mathrm{raw}}(1+\alpha)}{R} + B_{\mathrm{meta}},
\end{equation}
where $B_{\mathrm{meta}}$ is metadata rate from slice headers and ACK/control bits. $B_{\mathrm{meta}}$ is small but non-negligible at very high frame rates or many slices.

Implementation calculations guide hardware provisioning. Encoder throughput must sustain worst-case incoming pixel rate and average compression time per slice. If a compressor spends $t_{\mathrm{enc}}$ seconds per slice of $N_{\mathrm{px}}$ pixels, required encoder pixel throughput is $T_{\mathrm{enc}} = N_{\mathrm{px}}/t_{\mathrm{enc}}$. For parallel encoder lanes, number of lanes is $\lceil(B_{\mathrm{raw}}/(R \cdot T_{\mathrm{lane}}))\rceil$.

Buffer sizing is driven by variable-rate compression. If link capacity is $B_{\mathrm{link}}$ and the compressor can temporarily fall below that rate, buffer depth $D$ must satisfy
\begin{equation}[H]\label{eq:buffer_depth}
D \ge \max_{T}\left( (B_{\mathrm{raw}} - B_{\mathrm{link}})\cdot T \right),
\end{equation}
where $T$ is the maximum transient interval the encoder may underperform. For scanout-critical systems, $T$ is constrained by vertical blank duration and CRTC latency budget.

Concrete example and quick tool: the listing computes uncompressed and compressed link rates and a nominal buffer depth for a sample 4K HDR scenario.

\begin{lstlisting}[language=Python,caption={Compute bandwidth and buffer needs for a display stream.},label={lst:bw_calc}]
# Simple bandwidth calculator for display compression (example)
H, V = 3840, 2160                # resolution
fr = 60                          # refresh rate (Hz)
bpp = 30                         # bits per pixel (10bpc RGB)
alpha = 0.05                     # protocol overhead
R = 3.0                          # compression ratio (e.g., DSC target)
B_meta = 0.1e9                   # metadata budget (bps)
B_raw = H * V * fr * bpp         # Eq. used in prose
B_link = B_raw * (1+alpha) / R + B_meta
# Buffer depth for 10 ms transient where encoder falls to 50% throughput
transient_ms = 10
D_bits = max(0, (B_raw - B_link*0.5)) * (transient_ms/1000.0)
# Print Gbps and MB for buffer
print(f"Raw: {B_raw/1e9:.2f} Gbps, Link needed: {B_link/1e9:.2f} Gbps")
print(f"Buffer depth ~ {D_bits/8/1e6:.2f} MB for {transient_ms} ms transient")