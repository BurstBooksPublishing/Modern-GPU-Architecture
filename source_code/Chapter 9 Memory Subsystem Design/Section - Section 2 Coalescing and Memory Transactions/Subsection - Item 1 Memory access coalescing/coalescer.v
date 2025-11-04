module warp_coalescer #(
  parameter LANES = 32,
  parameter ADDR_WIDTH = 64,
  parameter LINE_BYTES = 128,
  parameter MAX_LINES = 8
)(
  input  wire                     clk,
  input  wire                     rst,
  input  wire                     start_valid,
  input  wire [LANES-1:0]         lane_valid,           // per-lane active mask
  input  wire [LANES*ADDR_WIDTH-1:0] lane_addr_flat,      // flattened addresses
  output reg                      done_valid,
  output reg  [MAX_LINES-1:0]     line_valid,           // which slots populated
  output reg  [MAX_LINES*(ADDR_WIDTH-$clog2(LINE_BYTES))-1:0] line_addr_flat, // base addresses (line index)
  output reg  [3:0]               num_lines,
  output reg                      spill                 // true if overflowed
);

localparam OFFSET = $clog2(LINE_BYTES);
integer i, j;
reg [ADDR_WIDTH-1:0] lane_addr [0:LANES-1];
reg [ADDR_WIDTH-OFFSET-1:0] idx_list [0:MAX_LINES-1];
reg [ADDR_WIDTH-OFFSET-1:0] cur_idx;
reg [ADDR_WIDTH-OFFSET-1:0] tmp_idx;
reg [MAX_LINES-1:0]        tmp_valid;

always @(posedge clk) begin
  if (rst) begin
    done_valid <= 0;
    line_valid <= 0;
    num_lines  <= 0;
    spill      <= 0;
  end else begin
    done_valid <= 0;
    if (start_valid) begin
      // unpack addresses
      for (i=0;i> OFFSET;
        // check if tmp_idx already in idx_list
        cur_idx = { (ADDR_WIDTH-OFFSET){1'b0} };
        for (j=0;j MAX_LINES) spill = 1'b1;
        end
      end
      // pack outputs
      for (j=0<j
\subsection{Item 2:  Transaction formation}
Following the coalescing analysis that groups per-lane accesses into cacheline-aligned sets, transaction formation converts those groups into concrete memory-interface requests that the memory controller and DRAM expect. This subsection examines the mapping from lane addresses to line-aligned transaction descriptors, the hardware algorithm to enumerate unique cache lines per warp, and a synthesizable Verilog implementation that emits compact burst requests.

Problem: given a warp of $W$ lanes with virtual addresses $A_i$ and a cache line size of $L$ bytes (power-of-two), produce the minimal set of aligned transactions (address + lane mask) so that each lane's requested byte range is covered exactly once. Analysis first defines the canonical line index
\begin{equation}[H]\label{eq:line_index}
\ell_i \;=\; \left\lfloor \dfrac{A_i}{L} \right\rfloor \;=\; A_i \gg s,
\end{equation}
with $s=\log_2 L$. A transaction is described by the pair (base\_addr, mask), where base\_addr $= \ell \cdot L$ and mask identifies which lanes map to $\ell$. The number of transactions $T$ satisfies
\begin{equation}[H]\label{eq:Tupper}
1 \le T \le \min(W,\,U),
\end{equation}
where $U$ is the number of distinct $\ell_i$ in the warp. For uniform random addresses across a region of size $R$ bytes partitioned into $L$-byte lines, the expected distinct-line count approximates $U \approx W \left(1 - \left(1-\frac{1}{R/L}\right)^{W-1}\right)$, illustrating why locality reduces $T$.

Implementation: the hardware needs to
\begin{itemize}
\item compute $\ell_i$ for every lane (cheap barrel shift),
\item detect unique line indices and build an array of unique entries,
\item generate per-transaction lane masks and aligned addresses,
\item pipeline issuance with valid/ready handshake to the memory request channel.
\end{itemize}

The following synthesizable Verilog module implements a simple combinational unique-detector and a small FSM that issues one transaction per unique line sequentially. It assumes flattened input buses for lane addresses and returns a $W$-bit lane mask for each transaction.

\begin{lstlisting}[language=Verilog,caption={Warp transaction formation unit (synthesizable).},label={lst:txn_form}]
module txn_form #(
  parameter W = 32,                 // warp width
  parameter ADDR_W = 64,
  parameter LINE_SHIFT = 6          // 64B lines
) (
  input  wire                    clk,
  input  wire                    rst_n,
  input  wire                    start,                              // start processing warp
  input  wire [W*ADDR_W-1:0]     lane_addrs_flat,                     // flattened lane addresses
  input  wire [W-1:0]            lane_valid,                          // per-lane active mask
  // memory request interface
  output reg                     mem_req_valid,
  input  wire                    mem_req_ready,
  output reg [ADDR_W-1:0]        mem_req_addr,
  output reg [W-1:0]             mem_req_lane_mask
);

  // unpack addresses and compute line indices
  reg [ADDR_W-1:0] lane_addr [0:W-1];
  reg [ADDR_W-LINE_SHIFT-1:0] line_idx [0:W-1];
  integer i,j;
  always @* begin
    for (i=0;i> LINE_SHIFT;
    end
  end

  // mark unique line indices and build compact list
  reg unique_flag [0:W-1];
  reg [ADDR_W-LINE_SHIFT-1:0] uniq_list [0:W-1];
  reg [W-1:0]                 uniq_mask_list [0:W-1];
  reg [5:0]                   uniq_count;
  always @* begin
    // init
    for (i=0;i0) next_state = ISSUE;
      ISSUE: if (issue_ptr+1 >= uniq_count && mem_req_valid && mem_req_ready) next_state = IDLE;
      default: next_state = IDLE;
    endcase
  end

endmodule