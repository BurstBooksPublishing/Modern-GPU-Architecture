module divergence_stack #(
  parameter LANES = 32,
  parameter PC_WIDTH = 16,
  parameter DEPTH = 8
)(
  input  wire                   clk,
  input  wire                   rst,
  input  wire                   push,            // push new entry
  input  wire                   pop,             // pop top entry
  input  wire [LANES-1:0]       push_mask,       // active lanes saved
  input  wire [PC_WIDTH-1:0]    push_reconv_pc,  // reconvergence PC
  output wire [LANES-1:0]       top_mask,
  output wire [PC_WIDTH-1:0]    top_reconv_pc,
  output wire                   empty,
  output wire                   full,
  output wire [3:0]             depth_count      // debug/telemetry
);

  reg [LANES-1:0]    stack_mask [0:DEPTH-1];
  reg [PC_WIDTH-1:0] stack_pc   [0:DEPTH-1];
  reg                stack_v    [0:DEPTH-1];
  reg [$clog2(DEPTH+1)-1:0] sp; // points to next free entry
  integer i;

  // synchronous push/pop pointer update
  always @(posedge clk) begin
    if (rst) begin
      sp <= 0;
      for (i=0;i
\subsection{Item 5:  Shader core testbench}
Building on the divergence stack's reconvergence semantics and the ALU/FPU datapath timing, the testbench must validate end-to-end behavior: correct active-mask propagation, latency hiding by the warp scheduler, and numerical correctness from the FP pipeline under concurrent warps.

Problem statement: verify a shader core under mixed workloads that exercise SIMT divergence, ALU and FPU pipelines, and warp scheduling. Analysis identifies three verification targets:
\begin{itemize}
\item functional correctness of divergence reconvergence and active-mask updates when branches diverge;
\item timing correctness: pipeline latency, scoreboard blocking, and bypass behavior across ALU/FPU units;
\item throughput and occupancy under stress patterns (many warps, high FPU utilization).
\end{itemize}

Operational strategy:
\begin{enumerate}
\item Create representative microbenchmarks: arithmetic-heavy warps, memory-heavy warps (modeled as stalls), and branch-divergent warps.
\item Drive the DUT with deterministic sequences and check outputs against a golden model calculated on-the-fly.
\item Measure cycles per retired warp and compute effective occupancy versus theoretical occupancy.
\end{enumerate}

Key performance math: if the scheduler services $W$ active warps and the average instruction pipeline latency is $L$ cycles, with per-warp instruction count $I$, the steady-state warp throughput $T$ (warps/clock) approximates
\begin{equation}[H]\label{eq:throughput}
T \approx \frac{W}{L + \frac{I}{\alpha}}
\end{equation}
where $\alpha$ is instruction issue width (instructions per cycle per SM). This relation guides stimulus intensity to saturate ALU/FPU without starving the scheduler.

Implementation: the testbench below instantiates a minimal synthesizable shader core stub and a harness that injects three warp contexts. The harness asserts reset, pulses instructions (including a branch opcode code 8'hBA to force divergence), and monitors a simple retired counter. Comments are brief inline.

\begin{lstlisting}[language=Verilog,caption={Synthesizable testbench harness for shader core microtests},label={lst:shader_tb}]
module tb_shader_core;
  reg clk = 0;
  reg rst = 1;
  // simple instruction bus signals
  reg        in_valid;
  reg [4:0]  in_warp_id;
  reg [31:0] in_instr;
  wire       retired; // simple retire handshake

  // Clock generator (50 MHz for simulation realism)
  always #10 clk = ~clk;

  // DUT: small synthesizable shader core stub (behavioral but RTL-style).
  shader_core_stub dut (
    .clk(clk), .rst(rst),
    .in_valid(in_valid), .in_warp_id(in_warp_id),
    .in_instr(in_instr), .retired(retired)
  );

  initial begin
    // release reset and feed three warps with mixed patterns
    #50 rst = 0;
    // Warp 0: arithmetic loop (keeps ALU busy)
    send_instr(0, 32'h00000001); // add
    send_instr(0, 32'h00000001);
    // Warp 1: divergent branch pattern
    send_instr(1, 32'h000000BA); // branch opcode forces divergence
    send_instr(1, 32'h00000001);
    // Warp 2: FPU-heavy
    send_instr(2, 32'h10000002); // fmul
    send_instr(2, 32'h10000002);
    #1000 $finish;
  end

  // simple task to send an instruction (synthesizable-style)
  task send_instr(input [4:0] wid, input [31:0] instr);
    begin
      @(posedge clk);
      in_warp_id <= wid;
      in_instr   <= instr;
      in_valid   <= 1;
      @(posedge clk);
      in_valid   <= 0;
    end
  endtask
endmodule

// Minimal shader core stub: synthesizable RTL modeling queue, basic execute, retire.
module shader_core_stub(
  input clk, input rst,
  input in_valid, input [4:0] in_warp_id, input [31:0] in_instr,
  output reg retired
);
  reg [7:0] pc [0:31];
  reg [3:0] state;
  always @(posedge clk) begin
    if (rst) begin
      state <= 0; retired <= 0;
    end else begin
      retired <= 0;
      if (in_valid) begin
        // emulate divergence opcode 0xBA by flipping a mask bit
        if (in_instr[7:0] == 8'hBA) pc[in_warp_id] <= pc[in_warp_id] + 2;
        else pc[in_warp_id] <= pc[in_warp_id] + 1;
        retired <= 1; // immediate retire for this stub
      end
    end
  end
endmodule