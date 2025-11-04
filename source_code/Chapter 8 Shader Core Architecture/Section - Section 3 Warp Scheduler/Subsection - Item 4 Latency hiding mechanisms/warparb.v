module warp_arbiter #(
  parameter NUM_WARPS = 32
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire [NUM_WARPS-1:0] ready_vec,   // warp requests
  input  wire [NUM_WARPS-1:0] score_ok,    // 1 = no dependency stalls
  output reg  [NUM_WARPS-1:0] grant,       // one-hot grant
  output reg  [$clog2(NUM_WARPS)-1:0] nxt_ptr
);
  // rotate request by pointer, pick first set bit
  wire [NUM_WARPS-1:0] cand = ready_vec & score_ok;
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      grant   <= {NUM_WARPS{1'b0}};
      nxt_ptr <= 0;
    end else begin
      grant <= {NUM_WARPS{1'b0}};
      // simple rotate search
      for (i=0;i
\section{Section 4: Register File Design}
\subsection{Item 1:  Banked register organization}
Building on the warp-scheduler demands for issuing many per-thread operands and the ISA's per-thread register semantics, the banked register organization addresses how to supply many simultaneous reads and writes without a prohibitive number of physical ports. It leverages spatial partitioning (banks), address-to-bank mapping, and compiler-aware allocation to balance area, timing, and parallel access needs.

Problem statement and analysis. A single monolithic register file with one physical port per simultaneous access scales poorly in area and wire congestion for an SM that issues across multiple ALUs, TMUs, or tensor cores. Banked organization partitions the logical register space into $B$ independent banks so that each bank can be a smaller multi-ported RAM or a single-ported RAM combined with banking time-multiplexing. The key metrics are:
\begin{itemize}
\item warp size $W$ (lanes issuing in lockstep),
\item number of simultaneous read accesses $R$ and write accesses $W_{\mathrm{r}}$ per cycle,
\item bank count $B$ and the mapping function $\mathrm{bank}(\mathrm{addr})$.
\end{itemize}
Under an assumption of uniform random bank mapping the probability that $W$ concurrent accesses all hit distinct banks (no bank conflicts) is
\begin{equation}[H]\label{eq:p_no_conflict}
P_{\text{no\_conflict}}=\frac{B(B-1)\cdots(B-W+1)}{B^{W}}=\frac{B!}{(B-W)!\,B^{W}}.
\end{equation}
In practice shader access patterns are not uniform: stride, swizzle, and contiguous vector accesses strongly affect conflicts. For contiguous lane $i$ accessing address $\mathrm{base}+i$, using $B$ equal to warp size often yields near-zero conflict for typical SIMD register layouts; however, strided accesses where stride $S$ is a multiple of $B$ produce pathological conflicts.

Implementation example. The following synthesizable Verilog module implements a parameterizable banked register file with synchronous reads, single-cycle write, bank selection via low-order bits (power-of-two $B$), and simple conflict detection flags. It is a conservative design intended for an SM datapath where external scheduling prevents multi-write-per-bank per cycle; conflict signals allow software or the scheduler to detect and avoid hazards.

\begin{lstlisting}[language=Verilog,caption={Synthesis-friendly banked register file with conflict detection},label={lst:banked_rf}]
module banked_regfile #(
  parameter NUM_BANKS = 8,
  parameter BANK_DEPTH = 64,
  parameter REG_WIDTH = 32,
  parameter READ_PORTS = 2,
  parameter WRITE_PORTS = 1
)(
  input  wire clk,
  input  wire [READ_PORTS-1:0][$clog2(NUM_BANKS*BANK_DEPTH)-1:0] raddr, // logical addr per read
  output reg  [READ_PORTS-1:0][REG_WIDTH-1:0] rdata,                      // synchronous read data
  input  wire [WRITE_PORTS-1:0]                       wen,
  input  wire [WRITE_PORTS-1:0][$clog2(NUM_BANKS*BANK_DEPTH)-1:0] waddr,
  input  wire [WRITE_PORTS-1:0][REG_WIDTH-1:0]        wdata,
  output wire [NUM_BANKS-1:0]                         bank_read_conflict, // >1 read to same bank
  output wire [NUM_BANKS-1:0]                         bank_write_conflict // >1 write to same bank
);

  localparam ADDR_W = $clog2(NUM_BANKS*BANK_DEPTH);
  // per-bank memories
  reg [REG_WIDTH-1:0] bank_mem [0:NUM_BANKS-1][0:BANK_DEPTH-1];

  // extract bank and local address (power-of-two NUM_BANKS)
  function [ $clog2(NUM_BANKS)-1:0 ] get_bank(input [ADDR_W-1:0] a);
    get_bank = a[ $clog2(NUM_BANKS)-1 : 0 ]; // low bits => bank
  endfunction
  function [ $clog2(BANK_DEPTH)-1:0 ] get_index(input [ADDR_W-1:0] a);
    get_index = a[ADDR_W-1 : $clog2(NUM_BANKS)];
  endfunction

  // conflict detection
  integer b,i,j;
  reg [NUM_BANKS-1:0] rcount, wcount;
  always @(*) begin
    rcount = {NUM_BANKS{1'b0}};
    wcount = {NUM_BANKS{1'b0}};
    for (i=0;i 1;
  assign bank_write_conflict = wcount > 1;

  // synchronous reads
  reg [REG_WIDTH-1:0] read_pipeline [0:READ_PORTS-1];
  always @(posedge clk) begin
    for (i=0;i
\subsection{Item 2:  Read/write port design}
The banked organization described previously reduces port contention by distributing lanes across independent memory arrays; the port design must now quantify residual contention, arbitration, and forwarding so the shader core meets SIMD/SIMT throughput targets. Here we analyze read/write port combinations, derive conflict probability, and present a synthesizable banked register-file RTL pattern that prioritizes correctness and predictable timing for warp-wide graphics and ML kernels.

Problem: a warp issues $R$ simultaneous reads and $W$ simultaneous writes per cycle to logical registers; physical banks $B$ each support $p$ single-cycle accesses (typically $p=1$ for single-ported SRAM, $p=2$ for true dual-port SRAM). To avoid stalls, choose $B$ large enough or provide port-multiplexing. A conservative sizing rule is
\begin{equation}[H]\label{eq:bank_count}
B \;\ge\; \left\lceil \frac{R+W}{p} \right\rceil,
\end{equation}
which ensures aggregate physical bandwidth matches logical peak. For uniformly distributed accesses, the probability of at least one bank conflict among $N=R+W$ accesses is
\begin{equation}[H]\label{eq:conflict_prob}
P_{\mathrm{conflict}} \;=\; 1 - \frac{B(B-1)\cdots(B-N+1)}{B^{N}}.
\end{equation}
This combinatorial model accurately predicts conflicts for randomized shader/compute workloads; real code exhibits locality, increasing conflict rates if mapping aligns badly with bank index bits.

Analysis and implementation choices:
\begin{itemize}
\item Replicated read ports via latch/replication reduce read latency but increase area and power. Many GPUs prefer narrow multi-banked SRAMs plus operand-collector crossbar to combine multiple single-ported banks into a logical multi-port file.
\item Write handling must implement forwarding to readers in the same cycle to preserve write-after-read and read-after-write semantics without pipeline bubbles.
\item Arbitration on multiple writes to same bank must be deterministic; typical policy is fixed priority or last-writer-wins within the same cycle, with software/compiler constraints to avoid hotspots.
\end{itemize}

Below is a synthesizable Verilog example implementing a parameterizable banked register file that maps multiple logical read/write ports to banks by low-address bits, performs cycle-local write-forwarding, and serializes conflicting writes with deterministic priority. The design uses flattened port buses so it is synthesizable with standard tools.

\begin{lstlisting}[language=Verilog,caption={Banked register file with multi-port logical interface; per-cycle forwarding and deterministic write priority.},label={lst:banked_rf}]
module banked_rf #(
  parameter DATA_WIDTH = 32,
  parameter TOTAL_REGS = 1024,
  parameter NUM_BANKS  = 16,
  parameter NUM_RPORTS = 4,
  parameter NUM_WPORTS = 2,
  parameter ADDR_WIDTH = $clog2(TOTAL_REGS)
)(
  input  wire                       clk,
  input  wire                       rst_n,
  input  wire [NUM_RPORTS*ADDR_WIDTH-1:0] raddr_flat, // read addresses
  output reg  [NUM_RPORTS*DATA_WIDTH-1:0] rdata_flat, // registered read data
  input  wire [NUM_WPORTS*ADDR_WIDTH-1:0] waddr_flat, // write addresses
  input  wire [NUM_WPORTS*DATA_WIDTH-1:0] wdata_flat, // write data
  input  wire [NUM_WPORTS-1:0]             wvalid     // write valids
);

localparam BANK_DEPTH = TOTAL_REGS / NUM_BANKS;
integer i, p;
reg [DATA_WIDTH-1:0] mem [0:NUM_BANKS-1][0:BANK_DEPTH-1]; // banked memories

// helper to extract slices
function [ADDR_WIDTH-1:0] get_addr;
  input integer idx;
  input [NUM_RPORTS*ADDR_WIDTH-1:0] vec;
  get_addr = vec[ idx*ADDR_WIDTH +: ADDR_WIDTH ];
endfunction

function [DATA_WIDTH-1:0] get_wdata;
  input integer idx;
  input [NUM_WPORTS*DATA_WIDTH-1:0] vec;
  get_wdata = vec[ idx*DATA_WIDTH +: DATA_WIDTH ];
endfunction

// synchronous reads with write-forwarding and deterministic write priority
always @(posedge clk) begin
  if (!rst_n) begin
    rdata_flat <= 0;
  end else begin
    // apply writes to banks in fixed priority order (w0 then w1 ..)
    for (i=0;i=0;i=i-1) begin
        if (wvalid[i]) begin
          integer wa = get_addr(i, waddr_flat);
          if (wa == raddr) begin fwd = get_wdata(i, wdata_flat); fwd_valid = 1; end
        end
      end
      if (fwd_valid) rdata_flat[p*DATA_WIDTH +: DATA_WIDTH] <= fwd;
      else rdata_flat[p*DATA_WIDTH +: DATA_WIDTH] <= mem[rb][rla];
    end
  end
end

endmodule