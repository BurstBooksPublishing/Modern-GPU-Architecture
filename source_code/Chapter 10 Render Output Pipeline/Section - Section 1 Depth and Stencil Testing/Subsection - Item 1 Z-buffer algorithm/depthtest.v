module depth_test #(
  parameter ADDR_WIDTH = 16,       // address bits -> pixels per tile
  parameter DEPTH_BITS = 24,       // depth precision
  parameter DEPTH_SIZE = 1<
\subsection{Item 2:  Depth and stencil compare functions}
The Z-buffer discussion established why per-fragment depth values are necessary to determine visible surfaces; this subsection builds on that by specifying the boolean logic used to compare incoming and stored values and how stencil interacts with depth to influence ROP behavior. We will analyze the compare functions, show their algebraic form, and provide a synthesizable Verilog implementation suitable for integration into a ROP/early-Z unit.

Operational relevance: depth and stencil compares determine whether a fragment is discarded before expensive shader or blend work, enabling Early-Z and Hierarchical-Z benefits. They also control per-pixel effects (stencil masking for shadow volumes or decals). Correct encoding and fast evaluation in hardware affect throughput, memory traffic, and the ability to perform early rejection.

Depth compare functions. GPUs commonly support eight depth compare functions: Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always. For unsigned fixed-point depth values (common 24-bit depth buffer), a depth test is:

\begin{equation}[H]\label{eq:depth_cmp}
\text{depth\_pass} \;=\; \text{compare}\big(d_{\text{frag}},\, d_{\text{buffer}}\big)
\end{equation}

where compare implements one of the operators: $<$, $\le$, $==$ , etc. For depth-only workloads, a pass may also enable a depth write: if depth passes and depth-write is enabled, the ROP issues a write to the depth cache and depth memory.

Stencil compare functions and mask operation. Stencil tests are typically performed as a masked compare of the reference value against the stored stencil byte. Formally:

\begin{equation}[H]\label{eq:stencil_cmp}
\text{stencil\_pass} \;=\; \text{compare}\big((S_{\text{ref}}\ \&\ M),\,(S_{\text{buf}}\ \&\ M)\big)
\end{equation}

where $S_{\text{ref}}$ is the application-provided reference, $S_{\text{buf}}$ is the stored stencil, and $M$ is the stencil mask. Common stencil ops executed depending on pass/fail include KEEP, REPLACE, INCR\_SAT, DECR\_SAT, INVERT, INCR\_WRAP, DECR\_WRAP.

Hardware analysis and implications:
\begin{itemize}
\item Evaluate depth and stencil in a pipelined stage that precedes color writes to enable early rejection. When both tests are enabled, the overall pass is logical AND: pass = depth\_pass AND stencil\_pass.
\item For MSAA, tests operate per-sample; combining sample mask reduction with hierarchical Z requires careful metadata to avoid false positives.
\item Depth compression schemes (HiZ, delta encoding) require conservative pass/early-reject decisions to maintain inviolate compressed ranges.
\end{itemize}

Implementation (synthesizable Verilog). The module below takes encoded compare functions and outputs pass signals and write enables. It assumes 24-bit unsigned depth and 8-bit stencil storage.

\begin{lstlisting}[language=Verilog,caption={Depth and stencil compare unit (synthesizable)},label={lst:depth_stencil}]
module depth_stencil_cmp (
    input  wire         clk,            // clock
    input  wire         rst_n,          // active-low reset
    input  wire         en_depth,       // depth test enable
    input  wire         en_depth_write, // depth write enable
    input  wire         en_stencil,     // stencil test enable
    input  wire [2:0]   depth_func,     // encode 0:NEVER,1:LESS,2:EQUAL,3:LEQUAL,4:GREATER,5:NOTEQUAL,6:GEQUAL,7:ALWAYS
    input  wire [2:0]   stencil_func,   // same encoding for stencil compare
    input  wire [23:0]  depth_in,       // incoming fragment depth (unsigned)
    input  wire [23:0]  depth_buf,      // stored depth
    input  wire [7:0]   stencil_ref,    // reference stencil
    input  wire [7:0]   stencil_buf,    // stored stencil
    input  wire [7:0]   stencil_mask,   // mask
    output reg          depth_pass,     // result of depth test
    output reg          stencil_pass,   // result of stencil test
    output wire         final_pass,     // final combined pass
    output wire         depth_write_en, // assert when depth should be written
    output wire         stencil_write_en// assert when stencil update should occur
);
    // Compare function helper (combinational)
    function automatic logic cmp;
        input [2:0] func;
        input [31:0] a;
        input [31:0] b;
        begin
            case (func)
                3'd0: cmp = 1'b0;                   // NEVER
                3'd1: cmp = (a < b);                // LESS
                3'd2: cmp = (a == b);               // EQUAL
                3'd3: cmp = (a <= b);               // LEQUAL
                3'd4: cmp = (a > b);                // GREATER
                3'd5: cmp = (a != b);               // NOTEQUAL
                3'd6: cmp = (a >= b);               // GEQUAL
                3'd7: cmp = 1'b1;                   // ALWAYS
                default: cmp = 1'b0;
            endcase
        end
    endfunction

    wire [7:0] masked_ref  = stencil_ref  & stencil_mask;
    wire [7:0] masked_buf  = stencil_buf  & stencil_mask;

    always @(*) begin
        if (!en_depth) depth_pass = 1'b1; // disabled => pass
        else depth_pass = cmp(depth_func, {8'b0, depth_in}, {8'b0, depth_buf});
        if (!en_stencil) stencil_pass = 1'b1;
        else stencil_pass = cmp(stencil_func, {24'b0, masked_ref}, {24'b0, masked_buf});
    end

    assign final_pass = depth_pass & stencil_pass;
    assign depth_write_en = final_pass & en_depth_write;
    assign stencil_write_en = final_pass & en_stencil; // flexible: may depend on stencil op logic

endmodule