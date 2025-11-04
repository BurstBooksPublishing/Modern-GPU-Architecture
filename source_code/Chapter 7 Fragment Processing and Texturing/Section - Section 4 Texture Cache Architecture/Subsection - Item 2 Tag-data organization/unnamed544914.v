module tag_store #
  (parameter TAG_W=20, WAYS=4, SETS=256, IDX_W=$clog2(SETS), WAY_W=$clog2(WAYS))
  (input  wire                  clk,
   input  wire                  rst,
   input  wire [IDX_W-1:0]      set_idx,    // index from address
   input  wire [TAG_W-1:0]      tag_in,     // tag for write
   input  wire                  wr_en,      // write tag/valid
   input  wire [WAY_W-1:0]      wr_way,     // which way to write
   output wire [WAYS-1:0]       hit_vec,    // per-way hit vector
   output wire                  hit,        // any-way hit
   output wire [WAY_W-1:0]      hit_way,    // selected hit way (lowest)
   output wire [WAY_W-1:0]      repl_way    // pseudo-LRU victim on miss
  );
  // Flattened RAMs for synthesis portability
  localparam TOT_ENT = WAYS*SETS;
  reg [TAG_W-1:0] tag_ram [0:TOT_ENT-1];
  reg             valid   [0:TOT_ENT-1];
  // PLRU tree bits per set: WAYS-1 bits for power-of-two WAYS
  reg [WAYS-2:0]  plru    [0:SETS-1];

  integer w;
  // Read combinational: compare tags in indexed set
  wire [WAYS-1:0] cmp;
  genvar gi;
  generate
    for (gi=0; gi
\subsection{Item 3:  Replacement policies (LRU, random)}
Building on the tag-data layout and multi-level texture cache hierarchy discussed previously, replacement policy choice determines how tag and data entries are aged and evicted under high-concurrency TMU (texture mapping unit) loads. The remainder links operational goals (maximize hit rate and minimize eviction latency) to implementable policy designs.

Problem: TMUs issue many concurrent, often streaming, fragment texture requests from warps on an SM; set-associative texture caches must evict a line when all ways are occupied. The two low-cost policies commonly implemented in hardware are true LRU (or approximations such as pseudo-LRU) and random replacement. Analysis of their effectiveness is best framed by request-level models and a characteristic-time approximation (Che's approximation), which connects per-address access rates to cache occupancy.

Analysis: Under an independent reference model with per-address Poisson rates $\lambda_i$, Che's approximation gives a characteristic time $T_C$ for LRU satisfying
\begin{equation}[H]\label{eq:ches}
\sum_i \big(1 - e^{-\lambda_i T_C}\big) = C,
\end{equation}
where $C$ is the cache capacity in lines. The LRU hit probability for object $i$ is
\begin{equation}[H]\label{eq:lru_hit}
h_i^{\mathrm{LRU}} = 1 - e^{-\lambda_i T_C}.
\end{equation}
A commonly used approximation for random replacement maps to a service-rate model with eviction rate $\mu = 1/T_C$; the hit probability becomes
\begin{equation}[H]\label{eq:rand_hit}
h_i^{\mathrm{RAND}} \approx \frac{\lambda_i}{\lambda_i + \mu} = \frac{\lambda_i T_C}{1 + \lambda_i T_C}.
\end{equation}
These expressions show LRU favours temporal locality (high $\lambda_i$) more sharply than random; for streaming textures with repeated spatial reuse within a frame (small reuse distances), LRU yields higher hit rates. Conversely, random replacement has lower hardware cost and avoids pathological thrashing when many addresses have similar $\lambda_i$ or adversarial stride patterns.

Implementation: For SM/TMU texture caches designers trade off area and update bandwidth. True per-access LRU counters scale poorly with associativity $A$. Practical choices:
\begin{itemize}
\item Pseudo-LRU tree for $A=4$ or $A=8$: uses $A-1$ bits per set, single-cycle update and victim selection.
\item Hardware random replacement: global LFSR + low-cost modulo to choose way.
\end{itemize}

Below are synthesizable Verilog modules: a 4-way tree PLRU and an LFSR-based random selector.

\begin{lstlisting}[language=Verilog,caption={4-way pseudo-LRU tree and LFSR random selector},label={lst:plru_rand}]
module plru_4way(
  input  wire        clk, rst,                // clock/reset
  input  wire        access_valid,            // update on access
  input  wire [1:0]  access_way,              // accessed way index
  input  wire        alloc_req,               // request replacement way
  output reg  [1:0]  repl_way                 // chosen victim way
);
  // tree bits: b0 = root, b1 = left child, b2 = right child
  reg b0, b1, b2;
  // update tree on access
  always @(posedge clk or posedge rst) begin
    if (rst) begin b0 <= 0; b1 <= 0; b2 <= 0; repl_way <= 2'b00; end
    else begin
      if (access_valid) begin
        case (access_way)
          2'b00: begin b0 <= 1; b1 <= 1; end // access left-left
          2'b01: begin b0 <= 1; b1 <= 0; end // access left-right
          2'b10: begin b0 <= 0; b2 <= 1; end // access right-left
          2'b11: begin b0 <= 0; b2 <= 0; end // access right-right
        endcase
      end
      if (alloc_req) begin
        // traverse bits: choose opposite direction of tree bit
        if (b0 == 0) begin // go left
          if (b1 == 0) repl_way <= 2'b00; else repl_way <= 2'b01;
        end else begin // go right
          if (b2 == 0) repl_way <= 2'b10; else repl_way <= 2'b11;
        end
      end
    end
  end
endmodule

module rand_replace(
  input  wire        clk, rst,      // clock/reset
  input  wire        alloc_req,     // request replacement way
  output reg  [1:0]  repl_way       // chosen victim way
);
  reg [7:0] lfsr;
  wire newbit = lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]; // x^8 + x^6 + x^5 + x^4 + 1
  always @(posedge clk or posedge rst) begin
    if (rst) lfsr <= 8'hA5;
    else lfsr <= {lfsr[6:0], newbit};
    if (alloc_req) repl_way <= lfsr[1:0]; // low bits map to way
  end
endmodule