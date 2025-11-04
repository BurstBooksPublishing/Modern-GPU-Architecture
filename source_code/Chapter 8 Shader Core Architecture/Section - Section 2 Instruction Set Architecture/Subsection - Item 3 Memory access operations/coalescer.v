module warp_coalescer #(
  parameter W = 32,               // warp width
  parameter ADDR_WIDTH = 64,
  parameter LINE_BITS = 6         // 64B lines
)(
  input  wire                    clk,
  input  wire                    rstn,
  input  wire                    start,                       // start capture
  input  wire [W*ADDR_WIDTH-1:0] addrs_flat,                  // flattened per-lane addresses
  output reg  [$clog2(W+1)-1:0]  unique_count,
  output reg  [W*(ADDR_WIDTH-LINE_BITS)-1:0] unique_lines_flat,
  output reg                     ready
);
  // internal decoded line indices
  reg [ADDR_WIDTH-LINE_BITS-1:0] line_idx [0:W-1];
  integer i,j;
  // combinational unpack
  always @(*) begin
    for (i=0;i> LINE_BITS;
    end
  end

  // deduplicate into unique list (simple O(W^2) CAM)
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      unique_count <= 0;
      unique_lines_flat <= 0;
      ready <= 1'b0;
    end else if (start) begin
      unique_count <= 0;
      unique_lines_flat <= 0;
      // build unique list
      for (i=0;i
\subsection{Item 4:  Control flow encoding}
Building on the previous discussion of memory-access encodings and extended arithmetic operands, control-flow encoding must tightly couple branch target representation, predicate information, and reconvergence metadata so the shader core can update SIMT active masks and PC state with minimal cycle cost.

Control-flow encoding problem statement: map all semantics required for warp-level branching into a fixed-width instruction word so hardware can cheaply extract next-PC, per-lane taken/not-taken masks, and a reconvergence token. Analysis shows three competing needs:
\begin{enumerate}
\item compact signed displacement for short relative branches (minimize fetch bandwidth cost),
\item explicit reconvergence identifier (join-id) to avoid expensive per-warp full-stack scans,
\item predicate selection or embedding to produce per-lane masks without extra micro-ops.
\end{enumerate}

Let instruction width be $W$ bits and fields widths be opcode $o$, join-id $j$, immediate $i$, predicate descriptor $p$, and reserved $r$. The width constraint is
\begin{equation}[H]\label{eq:fieldsum}
o + j + i + p + r = W.
\end{equation}
The signed branch displacement range is determined by $i$ bits, giving a reachable offset magnitude up to
\begin{equation}[H]\label{eq:disprange}
\text{range} = \pm 2^{\,i-1}.
\end{equation}
Reconvergence support requires $j \ge \lceil \log_2(S)\rceil$ bits to index $S$ reconvergence entries or join points in the divergence stack.

Implementation approach: encode a branch instruction with
\begin{itemize}
\item an opcode identifying conditional branch,
\item a join-id that labels the post-dominator reconvergence point,
\item an immediate displacement for fast PC update,
\item a predicate specifier that selects a per-lane predicate vector provided by the warp context.
\end{itemize}

The following synthesizable Verilog module implements the decode and mask computation used by a shader core front-end. It extracts fields, computes sign-extended target PC, produces taken and not-taken active masks, and asserts a push signal when divergence occurs (both masks non-zero). Inputs include the current \lstinline|active_mask| and per-lane \lstinline|predicates| vectors; outputs give the next PC and masks for downstream scheduler and reconvergence stack logic.

\begin{lstlisting}[language=Verilog,caption={Control-flow decode: branch/predicate mask extraction and target computation.},label={lst:cf_decoder}]
module cf_decoder
 #(parameter W=32, OPCW=6, JIDW=4, IMMW=12, N=32)
 (
  input  wire [W-1:0] inst_word,            // instruction word
  input  wire [31:0] pc,
  input  wire [N-1:0] active_mask,          // warp active lanes
  input  wire [N-1:0] predicates,           // per-lane predicate bits
  output reg  [31:0] next_pc,
  output reg  [N-1:0] taken_mask,
  output reg  [N-1:0] not_taken_mask,
  output reg         push_reconv,
  output reg  [JIDW-1:0] join_id
 );
 // field positions (MSB..LSB)
 localparam OPC_H = W-1;
 localparam OPC_L = W-OPCW;
 localparam JID_H = OPC_L-1;
 localparam JID_L = JID_H-JIDW+1;
 localparam IMM_H = JID_L-1;
 localparam IMM_L = IMM_H-IMMW+1;

 wire [OPCW-1:0] opcode = inst_word[OPC_H:OPC_L];
 wire signed [IMMW-1:0] imm_field = inst_word[IMM_H:IMM_L];

 // sign-extend immediate to 32 bits
 wire signed [31:0] imm_se = {{(32-IMMW){imm_field[IMMW-1]}}, imm_field};

 always @(*) begin
   join_id = inst_word[JID_H:JID_L];
   // opcode value for conditional branch assumed 6'b000100 (example)
   if (opcode == 6'b000100) begin
     taken_mask     = active_mask & predicates;         // lanes taking branch
     not_taken_mask = active_mask & ~predicates;        // lanes falling through
     push_reconv    = (taken_mask != 0) && (not_taken_mask != 0);
     // if any lane takes, choose target; hardware may use per-warp policy
     if (taken_mask != 0)
       next_pc = pc + imm_se;
     else
       next_pc = pc + 32'd4;
   end else begin
     // non-branch default
     taken_mask     = active_mask;
     not_taken_mask = {N{1'b0}};
     push_reconv    = 1'b0;
     next_pc = pc + 32'd4;
   end
 end
endmodule