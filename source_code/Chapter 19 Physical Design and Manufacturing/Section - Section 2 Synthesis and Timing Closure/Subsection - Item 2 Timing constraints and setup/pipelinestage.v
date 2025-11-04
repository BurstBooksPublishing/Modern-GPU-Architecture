module pipeline_stage #(
  parameter WIDTH = 32
)(
  input  wire              clk,      // core clock
  input  wire              rst_n,
  input  wire [WIDTH-1:0]  din,      // source data
  output reg  [WIDTH-1:0]  dout      // destination register
);
  reg [WIDTH-1:0] stage_reg;
  // long combinational logic example (synthesizable: replicated XOR tree)
  wire [WIDTH-1:0] comb;
  genvar i;
  generate
    for (i=0; i
\subsection{Item 3:  Clock domain verification}
The previous subsections established timing constraints and the RTL synthesis flow required for timing closure; clock domain verification extends those topics by ensuring crossing paths and constraints align across synthesized netlists and placed-and-routed designs. Correct CDC verification prevents elusive functional failures that survive synthesis and timing signoff, especially between SMs, memory controllers, and I/O domains such as PCIe or HBM.

Problem: GPUs have many independent clock domains (SM/compute, memory controller, display, video, interconnect), so signals cross asynchronous boundaries frequently. A missed synchronization or incorrect false-path constraint can corrupt texture descriptors in a TMU, confuse ROP write sequencing, or deadlock command queues. Metastability is the fundamental failure mode; design verification must quantify and mitigate its risk.

Analysis: Practical CDC verification combines static lint, formal checks, simulation with randomized domain phase relationships, and post-layout static timing checks with correctly specified \lstinline|create_clock| and \lstinline|set_false_path| SDC entries. For synchronizer design, Mean Time Between Failures (MTBF) approximates the expected time between unresolved metastability events. With two independent frequencies $f_1$ and $f_2$, and synchronizer resolution time $T_s$, an often-used model is
\begin{equation}[H]\label{eq:mtbf}
\mathrm{MTBF}\approx\frac{\exp(T_s/\tau)}{f_1 f_2 \tau},
\end{equation}
where $\tau$ is the metastability time constant of the flop. Equation (1) shows exponential improvement with added resolution slack, highlighting why extra synchronizer stages or deeper FIFO elastic depth can be cost-effective.

Implementation: for data-path crossings carry either:
\begin{itemize}
\item single-bit signals through two-flop synchronizers, or
\item multi-bit or burst transfers through asynchronous FIFOs using Gray-coded pointers and synchronized read/write pointer copies.
\end{itemize}

The listing below is a synthesizable, parameterized asynchronous FIFO used in many GPU datapaths (command queues, DMA descriptors). It uses Gray pointers and two-flop synchronizers for pointer transfer, producing safe empty/full flags and supporting block transfers without assuming phase alignment.

\begin{lstlisting}[language=Verilog,caption={Asynchronous FIFO with Gray-coded pointers and 2-flop synchronizers},label={lst:async_fifo}]
module async_fifo #(
  parameter DATA_WIDTH = 64,
  parameter ADDR_WIDTH = 6  // depth = 2^ADDR_WIDTH
) (
  input  wire                  wr_clk,
  input  wire                  rd_clk,
  input  wire                  rst_n,
  input  wire                  wr_en,
  input  wire [DATA_WIDTH-1:0] wr_data,
  input  wire                  rd_en,
  output reg  [DATA_WIDTH-1:0] rd_data,
  output wire                  full,
  output wire                  empty
);
  localparam DEPTH = (1<> 1);
      end
    end
  end

  // read domain logic
  always @(posedge rd_clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_ptr_bin <= 0;
      rd_ptr_gray <= 0;
      wr_ptr_gray_rdclk <= 0;
      wr_ptr_gray_rdclk_r <= 0;
      rd_data <= 0;
    end else begin
      // synchronize write pointer into read clock domain (2-flop)
      wr_ptr_gray_rdclk_r <= wr_ptr_gray_rdclk;
      wr_ptr_gray_rdclk   <= wr_ptr_gray;
      if (rd_en && !empty) begin
        rd_data <= mem[rd_ptr_bin[ADDR_WIDTH-1:0]]; // read data
        rd_ptr_bin <= rd_ptr_bin + 1;
        rd_ptr_gray <= (rd_ptr_bin + 1) ^ ((rd_ptr_bin + 1) >> 1);
      end
    end
  end

  // convert synchronized gray to binary for comparisons
  function [ADDR_WIDTH:0] gray_to_bin;
    input [ADDR_WIDTH:0] g;
    integer i;
    begin
      gray_to_bin[ADDR_WIDTH] = g[ADDR_WIDTH];
      for (i = ADDR_WIDTH-1; i >= 0; i = i-1)
        gray_to_bin[i] = gray_to_bin[i+1] ^ g[i];
    end
  endfunction

  wire [ADDR_WIDTH:0] rd_ptr_bin_sync = gray_to_bin(rd_ptr_gray_wrclk_r);
  wire [ADDR_WIDTH:0] wr_ptr_bin_sync = gray_to_bin(wr_ptr_gray_rdclk_r);

  assign full  = ((wr_ptr_bin[ADDR_WIDTH:0] - rd_ptr_bin_sync) == DEPTH);
  assign empty = (wr_ptr_bin_sync == rd_ptr_bin);
endmodule