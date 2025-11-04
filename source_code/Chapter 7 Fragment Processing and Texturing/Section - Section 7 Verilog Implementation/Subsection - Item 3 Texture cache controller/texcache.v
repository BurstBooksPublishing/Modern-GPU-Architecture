module texture_cache_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 128,        // bus width for line words (bits)
    parameter LINE_BYTES = 64,        // bytes per cache line
    parameter CACHE_BYTES = 16*1024,  // total cache size
    parameter ASSOC = 4,
    parameter MSHR_ENTRIES = 8
)(
    input  wire                     clk,
    input  wire                     rst,

    // Request from TMU: single-word address + id
    input  wire                     req_valid,
    input  wire [ADDR_WIDTH-1:0]    req_addr,
    input  wire [7:0]               req_id,      // return id
    output reg                      req_ready,

    // Response to TMU
    output reg                      resp_valid,
    output reg  [7:0]               resp_id,
    output reg  [DATA_WIDTH-1:0]    resp_data,   // entire line (filter reads slices)

    // Memory interface to L2/DRAM
    output reg                      mem_req_valid,
    output reg  [ADDR_WIDTH-1:0]    mem_req_addr,
    input  wire                     mem_req_ready,
    input  wire                     mem_resp_valid,
    input  wire [DATA_WIDTH-1:0]    mem_resp_data
);
    localparam NUM_SETS = CACHE_BYTES / (ASSOC * LINE_BYTES);
    localparam SET_IDX_BITS = $clog2(NUM_SETS);
    localparam OFFSET_BITS = $clog2(LINE_BYTES);
    localparam TAG_BITS = ADDR_WIDTH - SET_IDX_BITS - OFFSET_BITS;

    // Tag and data storage
    reg [TAG_BITS-1:0] tag_array [0:NUM_SETS-1][0:ASSOC-1];
    reg                valid_array [0:NUM_SETS-1][0:ASSOC-1];
    reg [DATA_WIDTH-1:0] data_array [0:NUM_SETS-1][0:ASSOC-1];

    // Simple round-robin LRU pointer per set
    reg [$clog2(ASSOC)-1:0] lru_ptr [0:NUM_SETS-1];

    // MSHR structure
    reg mshr_valid [0:MSHR_ENTRIES-1];
    reg [ADDR_WIDTH-1:0] mshr_addr [0:MSHR_ENTRIES-1];
    reg [7:0] mshr_owner_id [0:MSHR_ENTRIES-1];
    reg [MSHR_ENTRIES-1:0] mshr_merge_mask; // bit i set if merged into outstanding entry i

    integer i,j;
    // Reset
    always @(posedge clk) begin
        if (rst) begin
            req_ready <= 1'b1;
            resp_valid <= 1'b0;
            mem_req_valid <= 1'b0;
            for (i=0;i
\subsection{Item 4:  Fragment shader datapath}
The previous modules — the texture cache controller and the bilinear filter — supply sampled texels and filtered samples respectively; the fragment datapath consumes those outputs, performs shading math (lighting, blending, alpha tests), and drives ROP or framebuffer interfaces. In this subsection we analyze the datapath’s buffering and latency-hiding requirements, present a synthesizable RTL implementation, and summarize architectural trade-offs for SM-local shading pipelines.

Problem statement and analysis. A fragment unit must sustain a target pixel throughput $R_{\mathrm{pixels}}$ (pixels per clock) while tolerating TMU (texture mapping unit) latency $L_{\mathrm{tex}}$ (cycles) and varying shader instruction depth. To avoid stalling SM lanes, the shader needs sufficient outstanding request capacity $N_{\mathrm{pending}}$ so that
\begin{equation}[H]\label{eq:pending}
N_{\mathrm{pending}} \;\ge\; L_{\mathrm{tex}}\cdot R_{\mathrm{pixels}} .
\end{equation}
Operational relevance: if $N_{\mathrm{pending}}$ is too small the SIMT wavefronts will idle waiting for texture responses; if too large, local register and SRAM pressure increases, reducing occupancy.

Key datapath components and flow:
\begin{itemize}
\item request arbiter: converts incoming fragment attributes to TMU read requests while assigning a compact tag;
\item pending-entry store: small circular buffer (depth at least $N_{\mathrm{pending}}$) holding per-fragment attributes, interpolation data, and blending masks until texture response arrives;
\item shading ALU chain: a shallow pipeline implementing multiply-add trees for diffuse/specular, normal mapping combination, and optional sRGB conversion;
\item depth/alpha test unit and ROP interface: final comparisons and write-masking logic.
\end{itemize}

Throughput sizing example. For an SM target of $R_{\mathrm{pixels}}=0.5$ (one pixel every two cycles) with $L_{\mathrm{tex}}=64$ cycles, Eq. (1) yields $N_{\mathrm{pending}}\ge 32$ entries. If SIMD lanes increase $R_{\mathrm{pixels}}$ per SM, $N_{\mathrm{pending}}$ scales linearly.

Implementation. The following Verilog implements a compact, synthesizable fragment shader datapath that:
\begin{itemize}
\item issues TMU requests with a tag,
\item stores attributes in a small pending buffer,
\item merges TMU responses with stored attributes,
\item performs a fixed-point multiply-add shading operation,
\item outputs final color with valid-ready handshake.
\end{itemize}

\begin{lstlisting}[language=Verilog,caption={Fragment shader datapath (synthesizable).},label={lst:frag_datapath}]
module fragment_shader_datapath #(
  parameter TAG_BITS = 5,                     // depth = 32
  parameter ADDR_W   = 32,
  parameter ATTR_W   = 32,
  parameter COLOR_W  = 24
)(
  input  wire                 clk,
  input  wire                 rst_n,
  // incoming fragment
  input  wire                 in_valid,
  input  wire [ADDR_W-1:0]    in_tex_addr,
  input  wire [ATTR_W-1:0]    in_attr,       // interpolants
  output wire                 in_ready,
  // TMU request/response (simple valid/ready + tag)
  output reg                  tmu_req_valid,
  output reg  [ADDR_W-1:0]    tmu_req_addr,
  output reg  [TAG_BITS-1:0]  tmu_req_tag,
  input  wire                 tmu_req_ready,
  input  wire                 tmu_rsp_valid,
  input  wire [TAG_BITS-1:0]  tmu_rsp_tag,
  input  wire [COLOR_W-1:0]   tmu_rsp_color,
  // shaded output
  output reg                  out_valid,
  output reg  [COLOR_W-1:0]   out_color,
  input  wire                 out_ready
);

localparam DEPTH = 1 << TAG_BITS;

// simple circular buffers for attributes and state
reg [TAG_BITS-1:0] head_ptr, tail_ptr;
reg [TAG_BITS-1:0] next_tag;
reg [ATTR_W-1:0]   attr_mem [0:DEPTH-1];
reg [ADDR_W-1:0]   addr_mem [0:DEPTH-1];
reg                valid_mem [0:DEPTH-1];

// input handshake
assign in_ready = tmu_req_ready; // backpressure tied to TMU accept

// issue request and allocate entry
always @(posedge clk) begin
  if (!rst_n) begin
    head_ptr <= 0; tail_ptr <= 0; next_tag <= 0;
    tmu_req_valid <= 0; tmu_req_addr <= 0; tmu_req_tag <= 0;
  end else begin
    tmu_req_valid <= 0;
    if (in_valid && in_ready) begin
      // capture attributes into circular buffer at next_tag
      attr_mem[next_tag] <= in_attr;
      addr_mem[next_tag] <= in_tex_addr;
      valid_mem[next_tag] <= 1'b1;
      // issue TMU request
      tmu_req_valid <= 1'b1;
      tmu_req_addr  <= in_tex_addr;
      tmu_req_tag   <= next_tag;
      if (tmu_req_ready) begin
        next_tag <= next_tag + 1'b1;
      end
    end
    // retire entries when responses arrive handled separately
  end
end

// response handling and shading (fixed-point multiply-add)
reg [ATTR_W-1:0] lat_attr;
reg [COLOR_W-1:0] lat_tex;
wire [COLOR_W-1:0] shaded;
integer i;
// read stored attributes when response arrives
always @(posedge clk) begin
  if (!rst_n) begin
    out_valid <= 0; out_color <= 0;
  end else begin
    if (tmu_rsp_valid) begin
      lat_attr <= attr_mem[tmu_rsp_tag];
      lat_tex  <= tmu_rsp_color;
      valid_mem[tmu_rsp_tag] <= 1'b0;
      // simple shader: out = tex * (attr >> 8) + bias (fixed-point)
      // scale attr to 8-bit factor
      out_color <= (lat_tex * (lat_attr[15:8])) >> 8; // synthesize-friendly ops
      out_valid <= 1'b1;
    end else if (out_valid && out_ready) begin
      out_valid <= 1'b0;
    end
  end
end

endmodule