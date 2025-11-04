module rename_table #(
  parameter ARCH_REGS = 32,
  parameter PHYS_REGS = 1024,
  parameter NWARPS = 32,
  parameter WID_WIDTH = 5, // log2(NWARPS)
  parameter LREG_WIDTH = 5 // log2(ARCH_REGS)
)(
  input  wire clk, rst,
  // allocate port
  input  wire alloc,                              // single alloc request
  input  wire [WID_WIDTH-1:0] alloc_wid,
  input  wire [LREG_WIDTH-1:0] alloc_lreg,
  output reg  [$clog2(PHYS_REGS)-1:0] alloc_phys, // new phys assigned
  output reg  [$clog2(PHYS_REGS)-1:0] freed_phys, // old phys returned for retire
  output reg  alloc_ok,
  // free port (phys freed on retire)
  input  wire free_valid,
  input  wire [$clog2(PHYS_REGS)-1:0] free_phys_in
);

  // per-warp rename table: NWARPS x ARCH_REGS of phys indices
  reg [$clog2(PHYS_REGS)-1:0] rename_tbl [0:NWARPS-1][0:ARCH_REGS-1];
  reg [PHYS_REGS-1:0] free_bitmap; // 1 = free, 0 = allocated

  integer i,j;
  // init
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      free_bitmap <= {PHYS_REGS{1'b1}};
      for (i=0;i
\subsection{Item 4:  Spilling and allocation}
Register renaming and port organization determine how many physical locations each warp can use and how fast those locations are read or written; spilling bridges those constraints when the register file capacity or port bandwidth cannot satisfy compiler demands. Spilling and allocation must therefore be analyzed both as a compiler-driven mapping problem and as an on-chip runtime resource manager that trades SM occupancy against memory and pipeline latency.

Problem: when per-thread register demand $R_{\mathrm{req}}$ exceeds allocatable registers per warp $R_{\mathrm{alloc}}$, some live values must be evicted to slower storage (spill buffer in LDS/L1 or device memory). The cost of spilling is a combination of memory-store latency and lost parallelism due to reduced warp residency. A simple occupancy model is
\begin{equation}[H]\label{eq:occupancy}
\mathrm{occupancy} = \left\lfloor\frac{R_{\mathrm{RF}}}{R_{\mathrm{warp}}}\right\rfloor \Big/ W_{\max},
\end{equation}
where $R_{\mathrm{RF}}$ is total physical registers per SM, $R_{\mathrm{warp}}$ is per-warp register usage after allocation, and $W_{\max}$ is maximum warps per SM. Spilling increases $R_{\mathrm{warp}}$ effective cost and lowers occupancy.

Analysis: quantify spill cost for a kernel with average $N_s$ spilled registers per thread and memory-store latency $L_s$ cycles. The steady-state added latency per warp roughly equals
\begin{equation}[H]\label{eq:spill_cost}
C_{\mathrm{spill}} \approx N_s\cdot L_s + N_s\cdot L_r,
\end{equation}
with $L_r$ the refill latency. For ML kernels using tensor cores, $C_{\mathrm{spill}}$ can exceed compute time per tile, collapsing throughput.

Implementation: on-chip allocator must balance three responsibilities:
\begin{itemize}
\item fast single-cycle allocation for speculative register assignment by the compiler back-end or warp scheduler,
\item spill-slot bookkeeping that supports concurrent frees and reclaims across many warps,
\item locality-aware placement to favor LDS over off-chip when possible to minimize $L_s$.
\end{itemize}
A practical hardware design uses a circular free-list for spill slots with a bump allocator for steady allocations and a FIFO-style reclamation on commit. The following synthesizable Verilog implements a parameterizable spill-slot allocator that hands out contiguous slot indices and tracks occupancy; it supports allocate and free handshake signals.

\begin{lstlisting}[language=Verilog,caption={Simple spill-slot allocator (synthesizable).},label={lst:spill_alloc}]
module spill_allocator #(parameter SLOTS=256, ADDR_W=$clog2(SLOTS))(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 alloc_req,            // request N slots
  input  wire [7:0]           alloc_n,              // number requested
  output reg                  alloc_grant,
  output reg [ADDR_W-1:0]     alloc_addr,           // base slot index
  input  wire                 free_req,             // free N slots (commit)
  input  wire [7:0]           free_n,
  output wire                 full
);
  reg [ADDR_W-1:0] head, tail;
  reg [ADDR_W:0]     used;                          // count used slots
  assign full = (used + alloc_n > SLOTS);

  always @(posedge clk) begin
    if (rst) begin
      head <= 0; tail <= 0; used <= 0;
      alloc_grant <= 0; alloc_addr <= 0;
    end else begin
      alloc_grant <= 0;
      if (alloc_req && !full) begin
        alloc_addr <= head;
        head <= head + alloc_n;
        used <= used + alloc_n;
        alloc_grant <= 1;
      end
      if (free_req) begin
        tail <= tail + free_n;
        used <= (used >= free_n) ? used - free_n : 0;
      end
    end
  end
endmodule