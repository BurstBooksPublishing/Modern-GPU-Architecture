module decode_dispatcher #(
  parameter NENG = 4,
  parameter QDEPTH = 8,
  parameter JOBID_W = 8
)(
  input  wire         clk,
  input  wire         rstn,
  // submit job: {jobid, start_offset, ctucount, flags}
  input  wire         job_in_valid,
  input  wire [JOBID_W-1:0] job_in_id,
  output reg          job_in_ready,
  // per-engine interface
  output reg  [NENG-1:0] eng_valid,
  output reg  [JOBID_W-1:0] eng_jobid [NENG-1:0],
  input  wire [NENG-1:0] eng_ready,
  // completion feedback
  input  wire [NENG-1:0] eng_done
);
  // simple circular queue for pending jobs
  reg [JOBID_W-1:0] q_id [QDEPTH-1:0];
  reg [3:0] q_head, q_tail;
  reg [QDEPTH-1:0] q_ovfl; // occupancy bitmap
  integer i;
  // enqueue
  always @(posedge clk) begin
    if (!rstn) begin
      q_head <= 0; q_tail <= 0; job_in_ready <= 1;
    end else begin
      if (job_in_valid && job_in_ready) begin
        q_id[q_tail] <= job_in_id;
        q_tail <= q_tail + 1;
      end
      // available if not full
      job_in_ready <= ((q_tail - q_head) != QDEPTH);
    end
  end
  // simple round-robin allocate
  reg [1:0] rr_ptr;
  always @(posedge clk) begin
    if (!rstn) begin
      rr_ptr <= 0;
      for (i=0;i
\section{Section 4: Video Encode Acceleration}
\subsection{Item 1:  Motion estimation logic}
Building on the prior discussion of decode-side motion compensation and bitstream parsing, this subsection focuses on the encoder-side problem of finding inter-frame motion vectors efficiently in hardware. The goal is to minimize search cost while meeting throughput targets that match a GPU's display and video pipeline.

Problem statement and analysis. Motion estimation (ME) reduces temporal redundancy by finding a displacement that best matches a current block to a reference frame. A common cost is the sum of absolute differences (SAD), possibly extended with a rate term for rate–distortion optimization (RDO). The SAD for an $N\times N$ block is
\begin{equation}[H]\label{eq:sad}
\mathrm{SAD}(u,v)=\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}|I_{\text{cur}}(i,j)-I_{\text{ref}}(i+u,j+v)|.
\end{equation}
For full-search with radius $R$, candidate count equals $(2R+1)^2$, so computational work grows quadratically with $R$. High-resolution video combined with real-time constraints requires architectural measures to cut work.

Architectural techniques. Practical hardware combines:
\begin{itemize}
\item Multi-level hierarchical search (coarse-to-fine) to prune candidates early.
\item Parallel SAD engines evaluating many candidates concurrently, mapped to SIMD or dedicated ME lanes outside SMs.
\item Early termination and adaptive thresholds to skip full-block accumulation when partial sums exceed current minimum.
\item Sub-pixel refinement after integer-pixel search, using interpolated reference samples and smaller localized searches.
\item RDO cost combining SAD and lambda-scaled bitcost: $C(u,v)=\mathrm{SAD}(u,v)+\lambda\cdot R_{\text{bits}}(u,v)$.
\end{itemize}

Implementation example. A synthesizable streaming SAD engine for a $16\times 16$ block computes one candidate's SAD with one pixel pair consumed per cycle. Multiple identical engines operate in parallel to examine different $u,v$ offsets. Control logic sequences start pulses and collects outputs; an arbiter selects the minimum cost and associated vector.

\begin{lstlisting}[language=Verilog,caption={Synthesizable 16x16 streaming SAD engine},label={lst:sad16}]
module sad16x16 (
  input  wire        clk,
  input  wire        rst_n,
  input  wire        start,            // pulse to begin 256-pixel stream
  input  wire [7:0]  cur_pixel,        // current block pixel stream
  input  wire [7:0]  ref_pixel,        // reference candidate pixel stream
  input  wire        in_valid,         // valid when pixels are present
  output reg         done,             // asserted one cycle when SAD is ready
  output reg  [16:0] sad_out           // accumulated SAD (max 65280 < 2^16)
);
  reg [8:0] cnt;                         // count up to 256
  reg [16:0] acc;

  wire [8:0] next_cnt = cnt + 9'd1;
  wire [8:0] max_cnt = 9'd255;

  wire [8:0] diff = (cur_pixel > ref_pixel) ? (cur_pixel - ref_pixel) :
                                             (ref_pixel - cur_pixel);

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cnt <= 9'd0; acc <= 17'd0; sad_out <= 17'd0; done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        cnt <= 9'd0; acc <= 17'd0; done <= 1'b0;
      end else if (in_valid) begin
        acc <= acc + diff;           // accumulate absolute difference
        cnt <= next_cnt;
        if (cnt == max_cnt) begin
          sad_out <= acc + diff;     // final accumulate and latch
          done <= 1'b1;
          cnt <= 9'd0;
          acc <= 17'd0;
        end
      end
    end
  end
endmodule