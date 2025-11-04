module meshlet_processor #(
  parameter IDX_ADDR_WIDTH = 10, // index memory pointer width
  parameter VTX_COUNT = 64,      // local vertex cache entries
  parameter Q = 16               // Q16.16 fixed point
)(
  input  wire clk,
  input  wire rst_n,
  // descriptor interface (producer supplies one descriptor per meshlet)
  input  wire desc_valid,
  output reg  desc_ready,
  input  wire [IDX_ADDR_WIDTH-1:0] desc_idx_base,
  input  wire [15:0] desc_idx_count,
  input  wire signed [31:0] desc_cx, // Q16.16 center x
  input  wire signed [31:0] desc_cy, // Q16.16 center y
  input  wire signed [31:0] desc_cz, // Q16.16 center z
  input  wire signed [31:0] desc_r,  // Q16.16 radius
  // index fetch request/response (to index memory)
  output reg  idx_req_valid,
  input  wire idx_req_ready,
  output reg  [IDX_ADDR_WIDTH-1:0] idx_req_addr,
  input  wire [31:0] idx_resp_data, // packed index (two 16-bit indices)
  input  wire idx_resp_valid,
  // meshlet output (valid/ready)
  output reg  out_valid,
  input  wire out_ready,
  output reg  out_reject,           // high if culled
  output reg  [15:0] out_index_count
);

  // local index buffer (simple SRAM implemented as regs)
  reg [31:0] idx_buf [0:(1<
\subsection{Item 3:  VRS controller}
The tessellator reduced vertex rate and the meshlet processor's primitive bins both produce the tile-aligned geometry that a VRS controller consumes, so this controller must translate shading-rate images and coarse patterns into per-tile shading commands for the raster pipeline. Building on per-tile outputs, the VRS controller enforces SIMT-friendly, tile-granular shading rates while preserving compatibility with early-Z, TMU sampling, and ROP writeback.

Problem statement and goals: map a shading-rate image (coarse map) to tiles, arbitrate conflicting hints from application and hardware heuristics, and present a valid-ready interface to the rasterizer and tile scheduler. Constraints include minimal added latency, single-cycle decision path where possible, and compact encoding to avoid extra bandwidth to the raster front-end.

Analysis: represent available shading rates as an $N$-bit code: for example $r \in \{1,2,2\times1,1\times2\}$ encoded in two bits. For a tile of size $W \times H$ and shading rate $r$ representing an $r\times r$ block, the number of shaded samples per tile approximates
\begin{equation}[H]\label{eq:shaded_samples}
N_{\text{shaded}} \approx \left\lceil\frac{W\cdot H}{r^{2}}\right\rceil,
\end{equation}
which informs TMU and ROP bandwidth reduction estimates and helps the scheduler choose fusion of small tiles. The controller must also support dynamic overrides for foveated rendering.

Implementation summary: a small FSM implements: IDLE → READ_MAP → APPLY → WAIT_ACK. A synchronous BRAM stores the shading-rate image at tile granularity. The controller reads the entry for the incoming tile address, applies policy (application override > heuristic > default), outputs a shading-rate code, and increments a per-frame tile pointer. Handshaking uses \lstinline|tile_in_valid| / \lstinline|tile_in_ready| and \lstinline|tile_out_valid| / \lstinline|tile_out_ready| to integrate with existing tile pipelines.

The following synthesizable Verilog implements the core VRS controller and a synchronous BRAM interface. It is intentionally simple to illustrate the production-ready FSM and handshaking semantics.

\begin{lstlisting}[language=Verilog,caption={VRS controller with synchronous shading-rate BRAM},label={lst:vrs_ctrl}]
module vrs_controller #(
  parameter TILE_ADDR_BITS = 13, // supports 8192 tiles
  parameter SRATE_WIDTH = 2      // 2-bit shading-rate code
)(
  input  wire                  clk,
  input  wire                  rst_n,
  // incoming tile request (address)
  input  wire [TILE_ADDR_BITS-1:0] tile_in_addr,
  input  wire                  tile_in_valid,
  output reg                   tile_in_ready,
  // shading-rate BRAM interface (synchronous read)
  output reg  [TILE_ADDR_BITS-1:0] bram_addr,
  output reg                   bram_rd_en,
  input  wire [SRATE_WIDTH-1:0] bram_dout,
  // outgoing to rasterizer
  output reg                   tile_out_valid,
  output reg [SRATE_WIDTH-1:0] tile_out_srate,
  input  wire                  tile_out_ready,
  // override control (app can force a rate)
  input  wire                  override_en,
  input  wire [SRATE_WIDTH-1:0] override_srate
);

localparam IDLE     = 2'd0;
localparam READ_MAP = 2'd1;
localparam APPLY    = 2'd2;
localparam WAIT_ACK = 2'd3;

reg [1:0] state, next_state;
reg [SRATE_WIDTH-1:0] fetched_srate;

// FSM
always @(posedge clk or negedge rst_n) begin
  if (!rst_n) state <= IDLE;
  else state <= next_state;
end

always @(*) begin
  // default outputs
  tile_in_ready = 1'b0;
  bram_rd_en    = 1'b0;
  bram_addr     = tile_in_addr;
  tile_out_valid = 1'b0;
  tile_out_srate = override_srate;
  next_state = state;
  case (state)
    IDLE: begin
      tile_in_ready = 1'b1; // accept tile requests
      if (tile_in_valid) begin
        next_state = READ_MAP;
      end
    end
    READ_MAP: begin
      bram_rd_en = 1'b1; // issue synchronous read; data available next cycle
      next_state = APPLY;
    end
    APPLY: begin
      // choose override if enabled
      if (override_en) tile_out_srate = override_srate;
      else tile_out_srate = bram_dout;
      tile_out_valid = 1'b1;
      if (tile_out_ready) next_state = IDLE;
      else next_state = WAIT_ACK;
    end
    WAIT_ACK: begin
      tile_out_valid = 1'b1;
      if (tile_out_ready) next_state = IDLE;
    end
  endcase
end

endmodule