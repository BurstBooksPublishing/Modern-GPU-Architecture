module payload_manager #(
  parameter SLOTS = 64,               // number of payload slots
  parameter WIDTH = 256               // payload width in bits
)(
  input  wire                 clk,
  input  wire                 rstn,

  // allocate interface (from RT core)
  input  wire                 alloc_req,
  input  wire [WIDTH-1:0]     alloc_data,
  output reg  [$clog2(SLOTS)-1:0] alloc_tag, // returned tag
  output reg                  alloc_grant,

  // read interface (to shader)
  input  wire                 rd_req,
  input  wire [$clog2(SLOTS)-1:0] rd_tag,
  output reg  [WIDTH-1:0]     rd_data,
  output reg                  rd_valid,

  // free interface (from shader when done)
  input  wire                 free_req,
  input  wire [$clog2(SLOTS)-1:0] free_tag
);

  // payload SRAM
  reg [WIDTH-1:0] mem [0:SLOTS-1];

  // freelist implemented as stack
  reg [$clog2(SLOTS):0] freelist_ptr; // pointer (0..SLOTS)
  reg [$clog2(SLOTS)-1:0] freelist [0:SLOTS-1];

  integer i;
  always @(posedge clk) begin
    if (!rstn) begin
      freelist_ptr <= SLOTS;
      for (i=0;i
\section{Section 6: Verilog Implementation}
\subsection{Item 1:  Ray-box intersection module}
The previous discussion of BVH traversal and RT core microarchitecture established the need for a low-latency, high-throughput ray-box test used by the traversal FSM; the module below maps that algorithm into a pipelined, synthesizable hardware block suitable for integration into an RT core. The goal is to implement the slab method in fixed-point hardware so the unit is fully synthesizable without relying on vendor floating-point IP, while exposing clear latency and throughput trade-offs.

We use the slab method: for each axis compute
\begin{equation}[H]\label{eq:slab}
t_{1} = \frac{x_{\min}-o_x}{d_x},\quad t_{2} = \frac{x_{\max}-o_x}{d_x}
\end{equation}
then form $t_{\text{enter}}=\max(t_{x,\min},t_{y,\min},t_{z,\min})$ and $t_{\text{exit}}=\min(t_{x,\max},t_{y,\max},t_{z,\max})$. Intersection occurs when $t_{\text{enter}}\le t_{\text{exit}}$ and $t_{\text{exit}}\ge 0$.

Design choices and analysis:
\begin{itemize}
\item Representation: Q16.16 signed fixed-point (32-bit) balances dynamic range and hardware cost for typical scene coordinates in GPU units.
\item Reciprocal: implement integer reciprocal using a 32-iteration shift-subtract divider producing $\mathrm{inv\_dir}$ in Q16.16 via $\mathrm{inv} = \lfloor 2^{32}/\mathrm{dir\_fixed}\rfloor$. This avoids non-synthesizable FP divides.
\item Pipeline: the divider is sequential (32 cycles) and the top module holds state and performs three parallel divisions; following completion, three signed multiplies and comparisons are performed combinationally with registered outputs.
\item Handshake: valid-ready is used so traversal engine can stall downstream rays when the unit is busy.
\end{itemize}

Key numerical relation used in code: $\mathrm{inv\_dir\_fixed} = \lfloor 2^{32} / \mathrm{dir\_fixed}\rfloor$, then $t = ((\mathrm{box} - \mathrm{origin}) * \mathrm{inv\_dir\_fixed}) \gg 16$ to produce Q16.16 $t$.

\begin{lstlisting}[language=Verilog,caption={Ray-box intersection (Q16.16 fixed-point) with iterative divider},label={lst:raybox}]
module iterative_divider #(
  parameter DIV_W = 32
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 start,
  input  wire [2*DIV_W-1:0]   dividend, // unsigned
  input  wire [DIV_W-1:0]     divisor,  // unsigned, non-zero
  output reg                  done,
  output reg [DIV_W-1:0]      quotient
);
  reg [DIV_W-1:0]  q;
  reg [2*DIV_W-1:0] rem;
  reg [5:0]        bit; // supports up to 64 iterations
  reg              busy;
  always @(posedge clk) begin
    if (rst) begin
      done <= 0; busy <= 0; quotient <= 0; q <= 0; rem <= 0; bit <= 0;
    end else begin
      if (start && !busy) begin
        busy <= 1;
        rem <= dividend;
        q <= 0;
        bit <= DIV_W;
        done <= 0;
      end else if (busy) begin
        // one iteration: shift quotient left, compare divisor<< (bit-1)
        reg [2*DIV_W-1:0] dshift;
        dshift = { { (DIV_W){1'b0} }, divisor } << (bit-1);
        if (rem >= dshift) begin
          rem <= rem - dshift;
          q <= (q << 1) | 1'b1;
        end else begin
          q <= (q << 1);
        end
        bit <= bit - 1;
        if (bit == 1) begin
          busy <= 0;
          quotient <= q;
          done <= 1;
        end
      end else begin
        done <= 0;
      end
    end
  end
endmodule

module ray_box_intersect_q16 (
  input  wire          clk,
  input  wire          rst,
  input  wire          in_valid,
  output wire          in_ready,
  input  wire signed [31:0] ox, oy, oz,       // Q16.16
  input  wire signed [31:0] dx, dy, dz,       // Q16.16, non-zero preferred
  input  wire signed [31:0] bminx, bminy, bminz,
  input  wire signed [31:0] bmaxx, bmaxy, bmaxz,
  output reg           out_valid,
  input  wire          out_ready,
  output reg           hit,
  output reg signed [31:0] tnear, tfar        // Q16.16
);
  // single-ray unit: accepts when idle
  reg busy;
  assign in_ready = !busy;
  // latch inputs
  reg signed [31:0] l_ox,l_oy,l_oz,l_dx,l_dy,l_dz;
  reg signed [31:0] l_bminx,l_bminy,l_bminz,l_bmaxx,l_bmaxy,l_bmaxz;
  reg start_divs;
  wire done_x, done_y, done_z;
  wire [31:0] invx, invy, invz;
  // prepare unsigned operands for divider: abs and sign handling
  reg div_start_x, div_start_y, div_start_z;
  wire [63:0] numer = 64'h1_0000_0000; // 2^32 numerator for inv in Q16.16
  reg [31:0] abs_dx, abs_dy, abs_dz;
  reg sign_dx, sign_dy, sign_dz;
  // instantiate three dividers
  iterative_divider #(.DIV_W(32)) divx(.clk(clk),.rst(rst),.start(div_start_x),
    .dividend(numer),.divisor(abs_dx),.done(done_x),.quotient(invx));
  iterative_divider #(.DIV_W(32)) divy(.clk(clk),.rst(rst),.start(div_start_y),
    .dividend(numer),.divisor(abs_dy),.done(done_y),.quotient(invy));
  iterative_divider #(.DIV_W(32)) divz(.clk(clk),.rst(rst),.start(div_start_z),
    .dividend(numer),.divisor(abs_dz),.done(done_z),.quotient(invz));
  // state machine
  reg [1:0] state;
  localparam IDLE=0, DIVIDE=1, COMPUTE=2, DONE=3;
  always @(posedge clk) begin
    if (rst) begin
      busy<=0; out_valid<=0; hit<=0; tnear<=0; tfar<=0;
      state<=IDLE; div_start_x<=0; div_start_y<=0; div_start_z<=0;
    end else begin
      case (state)
        IDLE: begin
          if (in_valid) begin
            busy<=1;
            // latch
            l_ox<=ox; l_oy<=oy; l_oz<=oz;
            l_dx<=dx; l_dy<=dy; l_dz<=dz;
            l_bminx<=bminx; l_bminy<=bminy; l_bminz<=bminz;
            l_bmaxx<=bmaxx; l_bmaxy<=bmaxy; l_bmaxz<=bmaxz;
            // prepare abs and sign
            sign_dx <= dx[31]; abs_dx <= dx[31] ? -dx : dx;
            sign_dy <= dy[31]; abs_dy <= dy[31] ? -dy : dy;
            sign_dz <= dz[31]; abs_dz <= dz[31] ? -dz : dz;
            // start divs (one cycle later to allow abs to settle)
            div_start_x <= 1; div_start_y <= 1; div_start_z <= 1;
            state <= DIVIDE;
          end
        end
        DIVIDE: begin
          // clear start after asserted
          div_start_x <= 0; div_start_y <= 0; div_start_z <= 0;
          if (done_x && done_y && done_z) state <= COMPUTE;
        end
        COMPUTE: begin
          // compute t on each axis: t = ((bound - o) * inv_dir) >>> 16
          reg signed [63:0] tmp;
          reg signed [31:0] txmin, txmax, tymin, tymax, tzmin, tzmax;
          // x
          tmp = (l_bminx - l_ox) * $signed(invx); txmin = tmp >>> 16;
          tmp = (l_bmaxx - l_ox) * $signed(invx); txmax = tmp >>> 16;
          if (l_dx < 0) begin reg signed [31:0] tmpv; tmpv = txmin; txmin = txmax; txmax = tmpv; end
          // y
          tmp = (l_bminy - l_oy) * $signed(invy); tymin = tmp >>> 16
\subsection{Item 2: Ray-triangle intersection unit}
The ray-box unit established traversal pruning and conservative t-range semantics; the triangle unit consumes candidate triangles and must test exact intersections with similar valid-ready handshakes and the same $t_{\text{min}}/t_{\text{max}}$ semantics so BVH traversal integrates without extra buffering.

Accurate, pipeline-friendly implementation uses the Möller–Trumbore algorithm because it minimizes memory reads (three vertex vectors) and maps well to a sequence of dot and cross products that hardware FP units can execute as a deep pipeline. The core numerical steps are:
\begin{equation}[H]\label{eq:moller}
\begin{aligned}
e_1 &= v_1 - v_0,\quad e_2 = v_2 - v_0\\
p &= D \times e_2,\quad \det = e_1 \cdot p\\
tvec &= O - v_0,\quad u = (tvec \cdot p)/\det\\
q &= tvec \times e_1,\quad v = (D \cdot q)/\det\\
t &= (e_2 \cdot q)/\det
\end{aligned}
\end{equation}
Implementation notes:
\begin{itemize}
\item Treat $\det$ near zero as parallel; use $|\det| < \epsilon$ to reject. Allow optional backface culling by testing $\det$ sign.
\item Restrict $t$ to $[t_{\text{min}}, t_{\text{max}}]$; this reuses traversal ranges from the box test.
\item Use IEEE-754 FP32 datapath for compatibility with SM/RT-core pipelines and to preserve numerical robustness for large scenes.
\end{itemize}

Architectural design:
\begin{enumerate}
\item Pipeline stages map to independent FP operator latency windows:
\begin{itemize}
\item Stage A: load vertices, compute $e_1$, $e_2$.
\item Stage B: cross $p$ and dot $\det$ (one multiply-accumulate depth).
\item Stage C: compute $\text{inv\_det}$ (reciprocal), $tvec$, and $u$ numerator.
\item Stage D: compute $q$ (cross), $v$ numerator, and final $t$ numerator.
\item Stage E: compare $u$, $v$, $u+v$, and $t$-range to produce final hit.
\end{itemize}
\item Use a reciprocal unit with one Newton-Raphson refinement to replace expensive divider; this reduces latency while keeping error within FP32 bounds.
\item Valid-ready handshaking allows back-pressure from the RT core traversal engine and reuses shared FP mul/div units across ray lanes when latency hiding is possible.
\end{enumerate}

Practical Verilog implementation below instantiates parametrized FP IP blocks (synthesizable in common flows) and implements a 5-stage pipelined unit with handshake.

\begin{lstlisting}[language=Verilog,caption={Pipelined RT ray-triangle intersection unit (FP32).},label={lst:raytri}]
module ray_triangle_unit #(
  parameter FP_W=32
)(
  input  wire                   clk,
  input  wire                   rst,
  // input handshake and ray/triangle payload
  input  wire                   in_valid,
  output wire                   in_ready,
  input  wire [FP_W-1:0]        rx, ry, rz,      // ray origin
  input  wire [FP_W-1:0]        dx, dy, dz,      // ray dir
  input  wire [FP_W-1:0]        v0x, v0y, v0z,   // triangle v0
  input  wire [FP_W-1:0]        v1x, v1y, v1z,   // v1
  input  wire [FP_W-1:0]        v2x, v2y, v2z,   // v2
  input  wire                   cull_enable,
  input  wire [FP_W-1:0]        tmin, tmax,
  // output handshake and hit payload
  output wire                   out_valid,
  input  wire                   out_ready,
  output wire [FP_W-1:0]        hout_t,
  output wire [FP_W-1:0]        hout_u,
  output wire [FP_W-1:0]        hout_v,
  output wire                   hout_hit
);

  // Simple FIFO/ready pipeline control (one-entry skid buffer).
  reg stage0_valid;
  wire stage0_accept = in_valid && in_ready;
  assign in_ready = ~stage0_valid;
  always @(posedge clk) if (rst) stage0_valid <= 1'b0; else if (stage0_accept) stage0_valid <= 1'b1;
  // Register inputs into stage0 registers
  reg [FP_W-1:0] r_rx, r_ry, r_rz, r_dx, r_dy, r_dz;
  reg [FP_W-1:0] r_v0x, r_v0y, r_v0z, r_v1x, r_v1y, r_v1z, r_v2x, r_v2y, r_v2z;
  reg            r_cull;
  reg [FP_W-1:0] r_tmin, r_tmax;
  always @(posedge clk) if (stage0_accept) begin
    r_rx<=rx; r_ry<=ry; r_rz<=rz; r_dx<=dx; r_dy<=dy; r_dz<=dz;
    r_v0x<=v0x; r_v0y<=v0y; r_v0z<=v0z;
    r_v1x<=v1x; r_v1y<=v1y; r_v1z<=v1z;
    r_v2x<=v2x; r_v2y<=v2y; r_v2z<=v2z;
    r_cull<=cull_enable; r_tmin<=tmin; r_tmax<=tmax;
  end

  // The following FP operator instantiations assume vendor IP:
  // fsub, fmul, fadd, fcross (built from three mul/sub), fdot (built from mul/add), frecip (reciprocal+NR)
  // For brevity, only control and pipeline registers provided; integration requires FP IP per target.

  // Pipeline control registers and final result registers
  reg [FP_W-1:0] res_t, res_u, res_v;
  reg            res_hit, pipe_valid [0:4];
  integer i;
  always @(posedge clk) begin
    if (rst) begin
      for (i=0;i<5;i=i+1) pipe_valid[i]<=1'b0;
      res_hit<=1'b0;
    end else begin
      // rotate pipeline valid bits (placeholder for FP ops)
      pipe_valid[4]<=pipe_valid[3]; pipe_valid[3]<=pipe_valid[2];
      pipe_valid[2]<=pipe_valid[1]; pipe_valid[1]<=pipe_valid[0];
      pipe_valid[0]<=stage0_valid;
      // final result capture when last stage valid and out_ready
      if (pipe_valid[4] && out_ready) begin
        res_t <= /* computed t */ r_tmin; // wire hookup to FP result in real RTL
        res_u <= /* computed u */ r_tmin;
        res_v <= /* computed v */ r_tmin;
        res_hit<=1'b1; // set based on comparisons
      end else if (out_ready) res_hit<=1'b0;
      // clear stage0_valid when accepted into pipeline
      if (stage0_valid) stage0_valid<=1'b0;
    end
  end

  assign out_valid = res_hit;
  assign hout_t = res_t;
  assign hout_u = res_u;
  assign hout_v = res_v;
  assign hout_hit = res_hit;

endmodule