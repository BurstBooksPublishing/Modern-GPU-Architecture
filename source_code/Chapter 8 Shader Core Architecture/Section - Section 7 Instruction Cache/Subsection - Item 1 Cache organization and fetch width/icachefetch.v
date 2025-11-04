module icache_fetch #(
  parameter ADDR_WIDTH = 32,
  parameter BANK_WIDTH_BYTES = 16,
  parameter NUM_BANKS = 4,
  parameter BANK_DEPTH = 1024
) (
  input  wire                      clk,
  input  wire                      rst_n,
  input  wire [ADDR_WIDTH-1:0]     addr,    // byte address
  input  wire                      req,
  output reg  [NUM_BANKS*BANK_WIDTH_BYTES*8-1:0] bundle, // concatenated bundle
  output reg                       valid
);
  localparam DATA_WIDTH = BANK_WIDTH_BYTES*8;
  // Per-bank memories (synthesizable); bank indexed by generate.
  genvar b;
  generate
    for (b=0; b
\subsection{Item 2:  Branch prediction}
Following cache organization and fetch-width considerations, branch prediction supplies the instruction fetch engine with a higher-quality stream so that fetched lines match the active warp's execution path and reduce wasted instruction-cache bandwidth. This subsection analyzes why lightweight predictors matter in SIMT contexts, derives simple sizing equations for a BTB indexed by fetch-granularity, and presents a synthesizable Verilog implementation of a per-warp 2-bit bimodal predictor with tag and target storage.

Problem: SM fetch stalls occur when the next-fetch PC is unknown or when taken branches redirect fetch to a distant target; long-latency cache fills or alignment penalties waste IMC and increase warp occupancy pressure. Analysis shows that inexpensive hardware can recover a large fraction of these misses for typical shader and compute workloads that exhibit temporal branch locality.

Key analysis points:
\begin{itemize}
\item Use a BTB (branch target buffer) indexed by low-order PC bits shared across warps or maintained per-warp. Let the BTB have $N$ entries; the index width is $b=\log_2 N$. For a PC width $W$ bits and fetch-line offset $o$ bits, a conservative tag width is
\begin{equation}[H]\label{eq:tag_width}
t = W - b - o,
\end{equation}
which minimizes aliasing while keeping storage small. Example: $W=32$, $N=256$ ($b=8$), $o=4$ (16-byte fetch) gives $t=20$ bits.
\item A 2-bit saturating counter predictor (strongly taken, weakly taken, weakly not, strongly not) offers a good accuracy/area tradeoff for per-PC bimodal behavior and is simple to update on resolution.
\item Predictor read latency can be one cycle; a one-cycle speculative fetch plus instruction-buffering smooths pipeline flow. Predictor misprediction penalty is the fetch+bubble cost; accuracy improvement must outweigh added read latency.
\end{itemize}

Implementation: a compact, synthesizable Verilog module below implements a BTB with tag, 2-bit counter, and target registers. It supports synchronous read (predict available next cycle) and update ports for resolved branches.

\begin{lstlisting}[language=Verilog,caption={Synthesizable per-warp BTB + 2-bit bimodal predictor},label={lst:btb_predictor}]
module btb_predictor #(
  parameter PC_W = 32,
  parameter N_ENT = 256,             // number of entries
  parameter OFFSET = 4               // fetch line offset bits
)(
  input  wire                    clk,
  input  wire                    rst,
  // read request
  input  wire                    read_req,
  input  wire [PC_W-1:0]         read_pc,
  output reg                     predict_valid,
  output reg                     predict_taken,
  output reg [PC_W-1:0]          predict_target,
  // update (resolved branch)
  input  wire                    update_req,
  input  wire [PC_W-1:0]         update_pc,
  input  wire                    update_taken,
  input  wire [PC_W-1:0]         update_target
);
  localparam IDX_W = $clog2(N_ENT);
  localparam TAG_W = PC_W - IDX_W - OFFSET;

  // storage arrays
  reg [TAG_W-1:0] tag_ram [0:N_ENT-1];
  reg [1:0]      ctr_ram [0:N_ENT-1];    // 2-bit saturating counters
  reg [PC_W-1:0] target_ram [0:N_ENT-1];
  reg            valid_ram [0:N_ENT-1];

  wire [IDX_W-1:0] idx = read_pc[OFFSET +: IDX_W];
  wire [TAG_W-1:0] tag = read_pc[OFFSET+IDX_W +: TAG_W];

  // synchronous read: outputs valid next cycle
  reg [IDX_W-1:0] read_idx_r;
  always @(posedge clk) begin
    if (rst) begin
      predict_valid <= 1'b0;
      predict_taken <= 1'b0;
      predict_target <= {PC_W{1'b0}};
      read_idx_r <= {IDX_W{1'b0}};
    end else begin
      if (read_req) begin
        read_idx_r <= idx;
        // compare tag and valid
        if (valid_ram[idx] && tag_ram[idx] == tag) begin
          predict_valid <= 1'b1;
          predict_taken <= ctr_ram[idx][1];          // MSB indicates taken bias
          predict_target <= target_ram[idx];
        end else begin
          predict_valid <= 1'b0;
          predict_taken <= 1'b0;
          predict_target <= {PC_W{1'b0}};
        end
      end else begin
        predict_valid <= 1'b0;
      end
    end
  end

  // update on branch resolution
  always @(posedge clk) begin
    if (rst) begin
      integer i;
      for (i=0;i<N_ENT;i=i+1) begin
        valid_ram[i] <= 1'b0;
        ctr_ram[i] <= 2'b01;  // weakly not-taken
      end
    end else if (update_req) begin
      integer i;
      reg [IDX_W-1:0] upd_idx;
      upd_idx = update_pc[OFFSET +: IDX_W];
      valid_ram[upd_idx] <= 1'b1;
      target_ram[upd_idx] <= update_target;
      // 2-bit saturating counter update
      if (update_taken) begin
        if (ctr_ram[upd_idx] != 2'b11)
          ctr_ram[upd_idx] <= ctr_ram[upd_idx] + 1'b1;
      end else begin
        if (ctr_ram[upd_idx] != 2'b00)
          ctr_ram[upd_idx] <= ctr_ram[upd_idx] - 1'b1;
      end
    end
  end
endmodule