module inst_buffer #(
  parameter DATA_W = 32,          // 32-bit instruction word
  parameter BUNDLE = 4,           // words per decode bundle
  parameter DEPTH = 64            // total words (multiple of BUNDLE)
) (
  input  wire                 clk,
  input  wire                 rst_n,
  // write interface (from I-cache/fill engine)
  input  wire                 wr_valid,
  input  wire [DATA_W-1:0]    wr_data,
  output wire                 wr_ready,
  // read interface (to decoder) - atomic bundle read
  input  wire                 rd_req,    // request one bundle
  output reg  [BUNDLE*DATA_W-1:0] rd_bundle,
  output wire                 rd_valid,
  input  wire                 rd_ready
);

localparam ADDR_W = $clog2(DEPTH);
reg [ADDR_W-1:0] wr_ptr, rd_ptr;
reg [ADDR_W:0]   count; // up to DEPTH

wire full  = (count == DEPTH);
wire empty = (count == 0);

assign wr_ready = ~full;
assign rd_valid = (count >= BUNDLE) && ~empty;

// write one word per cycle (fill engine may burst)
always @(posedge clk) begin
  if (!rst_n) begin
    wr_ptr <= 0; rd_ptr <= 0; count <= 0;
  end else begin
    if (wr_valid && wr_ready) begin
      // memory modeled as reg array for synthesis
      // inferred RAM below
      mem[wr_ptr] <= wr_data;
      wr_ptr <= (wr_ptr + 1) % DEPTH;
      if (~(rd_req && rd_ready && (count==BUNDLE))) count <= count + 1;
    end
    if (rd_req && rd_ready && rd_valid) begin
      // assemble bundle from memory
      integer i;
      for (i=0;i
\subsection{Item 4:  Prefetch and alignment}
The previous discussion showed how instruction buffering smooths the fetch stream and how branch prediction supplies speculative PCs; prefetch and alignment are the hardware bridge that converts those streams into cache-friendly, line-aligned requests so the instruction buffer and fetch pipeline stay fed without wasting BRAM or L2 bandwidth.

Prefetch and alignment problem statement: shader cores fetch wide instruction packets for warps but instruction caches are organized in fixed-size lines (typically 32–128 bytes). A single fetch packet starting at byte address $A$ with width $W$ may cross one or more cache lines, creating extra misses or stall cycles if the fetch logic issues only a single-line request. The unit's goal is to align primary fetches to line boundaries and issue additional prefetches when the packet spans multiple lines, without polluting instruction cache with low-confidence streams from mispredicted branches.

Analysis: let $L$ be cache line size and let offset $o = A \bmod L$. The number of lines spanned by the packet is
\begin{equation}[H]\label{eq:lines_spanned}
n_{\text{lines}} \;=\; \left\lceil \frac{o + W}{L} \right\rceil .
\end{equation}
If $n_{\text{lines}}>1$, the prefetch unit must request the subsequent $n_{\text{lines}}-1$ lines promptly so the instruction buffer can assemble a contiguous fetch packet without stalling the warp scheduler. For SIMT workloads, frequent unaligned fetches (for example, due to variable-length instruction encodings or hot loops entering at unaligned offsets) amplify contention; for ML inference kernels with tight basic blocks, alignment reduces latency variance across warps.

Implementation: the simplest synthesizable hardware policy is on-demand sequential prefetching with single outstanding request and a conservative confidence filter that suppresses prefetches for speculative PCs with low prediction confidence. The following Verilog module is a production-ready, parameterized prefetch-and-align unit. It aligns the primary address, detects crossing, and issues a one-line-ahead prefetch with a valid/ready handshake.

\begin{lstlisting}[language=Verilog,caption={Simple prefetch-and-align unit (synthesizable).},label={lst:prefetch_align}]
module prefetch_align_unit #(
  parameter ADDR_WIDTH = 48,
  parameter LINE_BYTES = 64,      // must be power-of-two
  parameter FETCH_BYTES = 128
)(
  input  wire                     clk,
  input  wire                     rst_n,
  // fetch input from PC generator / branch predictor
  input  wire                     fetch_valid,
  input  wire [ADDR_WIDTH-1:0]    fetch_addr,
  // handshake to instruction cache / prefetch engine
  output reg                      primary_req_valid,
  output reg [ADDR_WIDTH-1:0]     primary_req_addr,
  output reg                      prefetch_req_valid,
  output reg [ADDR_WIDTH-1:0]     prefetch_req_addr,
  input  wire                     req_ack          // ack from cache/prefetch engine
);
  localparam LINE_BITS = $clog2(LINE_BYTES);
  // compute aligned base
  wire [ADDR_WIDTH-1:0] aligned_base = {fetch_addr[ADDR_WIDTH-1:LINE_BITS], {LINE_BITS{1'b0}}};
  wire [LINE_BITS-1:0] offset = fetch_addr[LINE_BITS-1:0];
  wire [15:0] sum_off = offset + FETCH_BYTES;
  wire crosses = (sum_off > LINE_BYTES);

  // simple single-entry outstanding flag
  reg outstanding;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      primary_req_valid  <= 1'b0;
      primary_req_addr   <= {ADDR_WIDTH{1'b0}};
      prefetch_req_valid <= 1'b0;
      prefetch_req_addr  <= {ADDR_WIDTH{1'b0}};
      outstanding        <= 1'b0;
    end else begin
      // issue primary aligned request when fetch_valid
      if (fetch_valid && !outstanding) begin
        primary_req_valid <= 1'b1;
        primary_req_addr  <= aligned_base;
        if (crosses) begin
          prefetch_req_valid <= 1'b1;
          prefetch_req_addr  <= aligned_base + LINE_BYTES;
        end else begin
          prefetch_req_valid <= 1'b0;
        end
        outstanding <= 1'b1;
      end
      // clear on ack (assumes ack covers both requests)
      if (req_ack) begin
        primary_req_valid  <= 1'b0;
        prefetch_req_valid <= 1'b0;
        outstanding        <= 1'b0;
      end
    end
  end
endmodule