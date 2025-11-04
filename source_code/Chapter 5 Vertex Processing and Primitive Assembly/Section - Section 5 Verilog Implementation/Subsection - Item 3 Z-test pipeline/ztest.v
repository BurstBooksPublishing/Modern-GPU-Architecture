module z_test_pipeline #(
  parameter DEPTH_WIDTH = 24,
  parameter SAMPLE_COUNT = 4,
  parameter CMP_BITS = 3
)(
  input  wire                      clk,
  input  wire                      rst_n,
  // Input pixel stream
  input  wire                      in_valid,
  output reg                       in_ready,
  input  wire [DEPTH_WIDTH-1:0]    in_depth   [0:SAMPLE_COUNT-1], // per-sample src depth
  input  wire [SAMPLE_COUNT-1:0]   in_mask,    // per-sample enable
  input  wire [DEPTH_WIDTH-1:0]    in_memdepth[0:SAMPLE_COUNT-1], // per-sample dst depth read
  input  wire [CMP_BITS-1:0]       cmp_func,   // select compare
  input  wire                      depth_write_en,
  // Output pixel stream
  output reg                       out_valid,
  input  wire                      out_ready,
  output reg  [SAMPLE_COUNT-1:0]   out_pass_mask,                 // per-sample pass
  output reg  [DEPTH_WIDTH-1:0]    out_write_depth [0:SAMPLE_COUNT-1] // depth to write
);

  // Internal pipeline registers
  reg [SAMPLE_COUNT-1:0] pass_stage;
  reg [SAMPLE_COUNT-1:0] mask_stage;
  integer i;
  // Accept when next stage free
  always @(posedge clk) begin
    if (!rst_n) begin
      in_ready <= 1'b1;
      out_valid <= 1'b0;
      pass_stage <= {SAMPLE_COUNT{1'b0}};
      mask_stage <= {SAMPLE_COUNT{1'b0}};
    end else begin
      // Stage1: when input valid and we can accept it
      if (in_valid && in_ready) begin
        for (i=0;i in_memdepth[i]);   // GREATER
            3'b011: pass_stage[i] <= (in_depth[i] >= in_memdepth[i]);  // GEQUAL
            3'b100: pass_stage[i] <= (in_depth[i] == in_memdepth[i]);  // EQUAL
            default: pass_stage[i] <= (in_depth[i] < in_memdepth[i]);  // fallback
          endcase
          mask_stage[i] <= in_mask[i];
          out_write_depth[i] <= in_depth[i]; // candidate write
        end
        // Propagate valid to output stage (single-cycle latency here)
        out_valid <= 1'b1;
        out_pass_mask <= {SAMPLE_COUNT{1'b0}}; // will be set next cycle
        in_ready <= 1'b0; // stall until out_ready consumed
      end else if (out_valid && out_ready) begin
        // Stage2: aggregate and finalize write mask
        for (i=0;i
\subsection{Item 4:  Tile buffer controller}
Building on the edge function generator's per-tile coverage masks and the Z-test pipeline's depth pass/kill signals, the tile buffer controller (TBC) must accumulate color and depth for all covered samples in a tile while providing backpressure to the raster pipeline and coalesced writes to the ROP/cache. The TBC's goal is to maximize on-chip reuse (reducing VRAM traffic) while preserving correct ordering and blending semantics for ROPs and fragment shaders.

Problem statement and analysis:
\begin{itemize}
\item Inputs from the rasterizer: tile coordinates, per-pixel coverage mask, per-sample color, and a per-sample $z_{\text{pass}}$ mask produced by the Z-test pipeline.
\item Outputs: read ports for the ROP/resolver to consume completed tiles, and a write-combine stream to external memory when evicting.
\item Constraints: limited SRAM budget per shader cluster (SM/CU), multi-sample support (MSAA), and low-latency valid-ready handshakes to avoid stalling SIMT pipelines.
\end{itemize}

Sizing and cost model: for a tile of width $W$, height $H$, color bits $C$ and samples $S$, the on-chip storage per tile is
\begin{equation}[H]\label{eq:tile_bytes}
B_{\text{tile}} = W \cdot H \cdot \frac{C}{8} \cdot S.
\end{equation}
Example: for $W=16$, $H=16$, $C=32$ (RGBA8), $S=4$ (4x MSAA), $B_{\text{tile}} = 16\cdot16\cdot4\cdot4 = 4096$ bytes per tile.

Implementation approach:
\begin{enumerate}
\item Maintain dual RAM arrays: color RAM and depth RAM indexed by pixel*sample index.
\item Accept write bursts gated by coverage and $z_{\text{pass}}$ to update only winning samples.
\item Track a per-tile valid bitmask and a dirty flag for eviction.
\item Provide a small FSM to serialize eviction via a memory interface (simplified burst writer) and a read port that presents tile data to the ROP/resolver with a valid-ready handshake.
\end{enumerate}

The following Verilog is a synthesizable, parameterized controller that implements the core accumulation and eviction control. It omits a full AXI implementation but provides a clean memory-agnostic eviction handshake you can map to a DMA/AXI wrapper.

\begin{lstlisting}[language=Verilog,caption=Tile buffer controller (synthesizable),label={lst:tilebuf}]
module tile_buffer_ctrl #(
  parameter TILE_W = 16,
  parameter TILE_H = 16,
  parameter SAMPLES = 4,
  parameter COLOR_W = 32, // bits per pixel (RGBA8)
  parameter DEPTH_W = 24
)(
  input  wire clk,
  input  wire rst,

  // Write request from raster (valid-ready)
  input  wire wr_valid,
  output reg  wr_ready,
  input  wire [$clog2(TILE_W)-1:0] wr_x,
  input  wire [$clog2(TILE_H)-1:0] wr_y,
  input  wire [SAMPLES-1:0] wr_coverage,    // per-sample coverage within pixel
  input  wire [SAMPLES-1:0] wr_zpass,       // per-sample z-test result
  input  wire [COLOR_W-1:0] wr_color,       // packed color for pixel (same per sample)
  input  wire [DEPTH_W-1:0] wr_depth,       // depth value for pixel/sample

  // Read request for resolver/ROP (simple handshake)
  input  wire rd_req,
  output reg  rd_valid,
  output reg  [COLOR_W-1:0] rd_color,
  output reg  [DEPTH_W-1:0] rd_depth,
  output reg  [31:0] rd_addr,               // pixel*sample address within tile

  // Eviction interface (to DMA/VRAM writer)
  output reg  evict_req,
  input  wire evict_ack
);

  localparam PIXELS = TILE_W * TILE_H;
  localparam CELLS  = PIXELS * SAMPLES;
  // simple single-port memories (synthesizable)
  reg [COLOR_W-1:0] color_mem [0:CELLS-1];
  reg [DEPTH_W-1:0] depth_mem [0:CELLS-1];
  reg dirty; // tile dirty flag
  integer i;

  // compute linear pixel index
  function integer pix_idx(input integer x, input integer y, input integer s);
    pix_idx = ((y * TILE_W) + x) * SAMPLES + s;
  endfunction

  // Write accumulate: update only if coverage & zpass asserted
  always @(posedge clk) begin
    if (rst) begin
      wr_ready <= 1'b0;
      dirty <= 1'b0;
    end else begin
      wr_ready <= 1'b1; // always ready in this simple controller
      if (wr_valid && wr_ready) begin
        for (i = 0; i < SAMPLES; i = i + 1) begin
          if (wr_coverage[i] && wr_zpass[i]) begin
            color_mem[pix_idx(wr_x, wr_y, i)] <= wr_color;
            depth_mem[pix_idx(wr_x, wr_y, i)] <= wr_depth;
            dirty <= 1'b1;
          end
        end
      end
    end
  end

  // Read/evict FSM: present cell data sequentially when requested
  reg [31:0] rd_ptr;
  always @(posedge clk) begin
    if (rst) begin
      rd_valid <= 1'b0;
      rd_ptr <= 0;
      evict_req <= 1'b0;
    end else begin
      if (rd_req && !rd_valid && dirty) begin
        rd_ptr <= 0;
        rd_valid <= 1'b1;
        rd_color <= color_mem[0];
        rd_depth <= depth_mem[0];
        rd_addr  <= 0;
      end else if (rd_valid && rd_req) begin
        // consumer accepted current; advance
        rd_ptr <= rd_ptr + 1;
        if (rd_ptr + 1 < CELLS) begin
          rd_color <= color_mem[rd_ptr+1];
          rd_depth <= depth_mem[rd_ptr+1];
          rd_addr  <= rd_ptr+1;
        end else begin
          rd_valid <= 1'b0;
          // request eviction (write-back) once read completes
          evict_req <= dirty;
        end
      end
      if (evict_req && evict_ack) begin
        evict_req <= 1'b0;
        dirty <= 1'b0;
      end
    end
  end

endmodule