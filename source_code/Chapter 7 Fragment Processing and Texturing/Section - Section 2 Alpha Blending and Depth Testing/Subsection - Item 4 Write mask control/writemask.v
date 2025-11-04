module write_mask_control #(
  parameter NUM_RTS = 4,
  parameter NUM_SAMPLES = 4,
  parameter CHANNELS = 4
)(
  input  wire                        clk,
  input  wire                        rst_n,
  input  wire [NUM_SAMPLES-1:0]      coverage_mask,  // per-sample coverage
  input  wire [NUM_SAMPLES-1:0]      depth_pass,     // per-sample Z test result
  input  wire [NUM_SAMPLES-1:0]      stencil_pass,   // per-sample stencil result
  input  wire                        alpha_pass,     // fragment-level alpha test
  input  wire [NUM_RTS-1:0]          rt_enable,      // per-RT enable
  input  wire [NUM_RTS*CHANNELS-1:0] channel_mask,   // flattened per-RT per-channel mask
  output reg  [NUM_RTS*NUM_SAMPLES*CHANNELS-1:0] write_enable // flattened output
);

  integer t,s,c;
  always @(*) begin
    write_enable = { (NUM_RTS*NUM_SAMPLES*CHANNELS){1'b0} };
    for (t=0; t
\section{Section 3: Texture Compression}
\subsection{Item 1:  BC and ASTC formats}
Following the texture addressing and filtering discussion, compressed texture formats supply the TMU with compact blocks that the filtering stage reconstructs into filtered samples. This subsection examines BC (Block Compression) and ASTC (Adaptive Scalable Texture Compression) from the TMU and texture-cache design perspective, quantifies decompressor throughput needs, and presents a synthesizable BC1 block decoder module suitable for an SM-local TMU pipeline.

BC formats (BC1–BC7) use fixed small blocks—BC1 (DXT1) encodes a 4×4 texel block in 64 bits (two 16-bit RGB565 endpoints plus 32 bits of 2-bit indices). ASTC uses a flexible block footprint (4×4 up to 12×12) and variable bitrates with per-block partitioning and weight grids, enabling better quality at similar bitrates for HDR and normal maps. Key operational differences for hardware:
\begin{itemize}
\item BC: simple, low-area decoders; constant block layout allows single-cycle combinational palette generation followed by index lookup; ideal for high-throughput TMUs.
\item ASTC: complex partition and weight decoding, often pipelined over multiple cycles with small LUTs and DSP-accelerated interpolation to meet per-request latency budgets.
\end{itemize}

Analyze compression impact with a simple ratio expression. Let $S_u$ be uncompressed bytes per texel (e.g., 4 for RGBA8), $B_b$ compressed block size in bytes, and $A_b$ block area in texels; compression ratio $R_c$ is
\begin{equation}[H]\label{eq:compression_ratio}
R_c \;=\; \frac{S_u \cdot A_b}{B_b}.
\end{equation}
For BC1, $S_u=4$, $A_b=16$, $B_b=8$, so $R_c=8$ (8:1). For ASTC 8×8 at 128 bits per block, $R_c = \frac{4\cdot64}{16} = 16$ (16:1) if using 128-bit blocks for 8×8.

Implementation constraints for TMU datapath:
\begin{enumerate}
\item Decoder throughput must match TMU lane demand: if a TMU issues $X$ blocks per cycle, the decompress stage must sustain that throughput to avoid stalling filtering and texture-cache refill.
\item Cache alignment: BC blocks align naturally to 8-byte cache lines; ASTC variable footprints require metadata to locate blocks within lines, increasing tag complexity.
\item Quality vs area: ASTC achieves higher visual fidelity at increased decode complexity and larger area for lookup tables and interpolation logic.
\end{enumerate}

The listing below is a synthesizable Verilog module that decodes BC1 (DXT1) blocks into sequential RGB888 pixels. It is intended as an SM/TMU-local decompressor stage feeding a bilinear filter. Signal names in text use \lstinline|...| notation.

\begin{lstlisting}[language=Verilog,caption={Synthesizable BC1 (DXT1) block decoder; outputs 16 RGB888 pixels sequentially.},label={lst:bc1_decoder}]
module bc1_decoder (
  input  wire        clk,
  input  wire        rstn,
  input  wire        valid_in,           // block valid
  input  wire [63:0] block_in,           // BC1 64-bit block
  output reg         ready,              // ready for next block
  output reg         pixel_valid,        // pixel data valid
  output reg  [3:0]  pixel_index,        // 0..15
  output reg  [23:0] rgb_out,            // RGB888
  output reg         alpha_out           // 1 if transparent (when index==3 and c0<=c1)
);
  reg [15:0] c0,c1;
  reg [23:0] col0,col1,col2,col3;
  reg [31:0] indices;
  reg [4:0]  ptr;
  reg       mode4; // 1 if c0>c1 (four-color), 0 if three-color + transparent

  // expand RGB565 to RGB888
  function [23:0] expand565(input [15:0] c);
    reg [4:0] r; reg [5:0] g; reg [4:0] b;
  begin
    r = c[15:11]; g = c[10:5]; b = c[4:0];
    expand565 = { {r,3'b000} | {r[4:2]}, {g,2'b00} | {g[5:4]}, {b,3'b000} | {b[4:2]} };
  end endfunction

  always @(posedge clk) begin
    if (!rstn) begin
      ready <= 1'b1; pixel_valid <= 1'b0; ptr <= 0; pixel_index <= 0; alpha_out <= 0;
    end else begin
      if (valid_in && ready) begin
        // latch block
        c0 <= block_in[63:48];
        c1 <= block_in[47:32];
        indices <= block_in[31:0];
        mode4 <= (block_in[63:48] > block_in[47:32]);
        // compute palette
        col0 <= expand565(block_in[63:48]);
        col1 <= expand565(block_in[47:32]);
        if (block_in[63:48] > block_in[47:32]) begin
          col2 <= { ( (2*col0[23:16] + col1[23:16]) / 3 ),
                    ( (2*col0[15:8]  + col1[15:8])  / 3 ),
                    ( (2*col0[7:0]   + col1[7:0])   / 3 ) };
          col3 <= { ( (col0[23:16] + 2*col1[23:16]) / 3 ),
                    ( (col0[15:8]  + 2*col1[15:8])  / 3 ),
                    ( (col0[7:0]   + 2*col1[7:0])   / 3 ) };
        end else begin
          col2 <= { ( (col0[23:16] + col1[23:16]) >> 1 ),
                    ( (col0[15:8]  + col1[15:8])  >> 1 ),
                    ( (col0[7:0]   + col1[7:0])   >> 1 ) };
          col3 <= 24'h000000; // transparent
        end
        ready <= 1'b0;
        ptr <= 0;
        pixel_valid <= 1'b1;
        pixel_index <= 0;
        alpha_out <= 1'b0;
      end else if (!ready) begin
        // output pixel based on 2-bit index
        reg [1:0] idx;
        idx = indices[ptr*2 +: 2];
        case (idx)
          2'b00: rgb_out <= col0;
          2'b01: rgb_out <= col1;
          2'b10: rgb_out <= col2;
          2'b11: begin rgb_out <= col3; alpha_out <= ~mode4; end
        endcase
        pixel_index <= ptr;
        ptr <= ptr + 1;
        if (ptr == 15) begin
          ready <= 1'b1; pixel_valid <= 1'b0;
        end
      end
    end
  end
endmodule