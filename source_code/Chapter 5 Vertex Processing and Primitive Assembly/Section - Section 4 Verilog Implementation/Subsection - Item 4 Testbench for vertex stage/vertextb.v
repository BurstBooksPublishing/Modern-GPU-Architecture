`timescale 1ns/1ps
module vertex_stage(
  input  wire         clk,
  input  wire         rst_n,
  input  wire         in_valid,
  output wire         in_ready,
  input  wire [95:0]  in_pos,    // three Q16.16 signed coords x,y,z (3*32)
  input  wire [127:0] mtx00_33,  // 4x4 matrix (4*32)
  output reg          out_valid,
  input  wire         out_ready,
  output reg  [31:0]  out_x,     // Q16.16
  output reg  [31:0]  out_y,
  output reg  [31:0]  out_z,
  output reg          clipped
);
  // simple ready logic: accept when not holding a valid output
  assign in_ready = ~out_valid | out_ready;

  // pipeline registers (latency 2): stage0 capture, stage1 multiply+homogeneous check
  reg [31:0] sx, sy, sz;
  reg [127:0] m00_33;
  reg        s_valid0;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_valid0 <= 0;
      sx <= 0; sy <= 0; sz <= 0; m00_33 <= 0;
    end else begin
      if (in_valid & in_ready) begin
        sx <= in_pos[95:64]; sy <= in_pos[63:32]; sz <= in_pos[31:0];
        m00_33 <= mtx00_33;
        s_valid0 <= 1;
      end else if (out_ready & out_valid) begin
        s_valid0 <= 0;
      end
    end
  end

  // helper: fixed-point multiply Q16.16
  function [31:0] fmul;
    input [31:0] a; input [31:0] b;
    reg signed [63:0] prod;
    begin
      prod = $signed(a) * $signed(b);
      fmul = prod >>> 16; // Q16.16 result
    end
  endfunction

  reg s_valid1;
  reg [31:0] rx, ry, rz, rw;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_valid1 <= 0; out_valid <= 0; clipped <= 0;
    end else begin
      if (s_valid0) begin
        // matrix elements: row-major m00,m01,...m33 each 32-bit
        reg [31:0] m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33;
        {m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33} = m00_33;
        // multiply: compute homogeneous coordinates
        rx = fmul(m00, sx) + fmul(m01, sy) + fmul(m02, sz) + m03;
        ry = fmul(m10, sx) + fmul(m11, sy) + fmul(m12, sz) + m13;
        rz = fmul(m20, sx) + fmul(m21, sy) + fmul(m22, sz) + m23;
        rw = fmul(m30, sx) + fmul(m31, sy) + fmul(m32, sz) + m33;
        s_valid1 <= 1;
      end else if (out_ready & out_valid) begin
        s_valid1 <= 0;
      end

      if (s_valid1) begin
        // simple frustum: require rw > 0 and |x/w|<=1 etc. use fixed-point compare
        if (rw <= 0) begin
          clipped <= 1; out_valid <= 1; out_x <= 0; out_y <= 0; out_z <= 0;
        end else begin
          // perspective divide: x'=x/w (Q16.16 division via 32-bit signed)
          out_x <= ($signed(rx) <<< 16) / $signed(rw); // approximate
          out_y <= ($signed(ry) <<< 16) / $signed(rw);
          out_z <= ($signed(rz) <<< 16) / $signed(rw);
          clipped <= 0; out_valid <= 1;
        end
      end else if (out_ready & out_valid) begin
        out_valid <= 0; clipped <= 0;
      end
    end
  end
endmodule

// Testbench
module tb_vertex_stage;
  reg clk=0, rst_n=0;
  reg in_valid=0; wire in_ready;
  reg [95:0] in_pos;
  reg [127:0] mtx;
  wire out_valid; reg out_ready=1;
  wire [31:0] out_x,out_y,out_z; wire clipped;

  vertex_stage dut(.clk(clk), .rst_n(rst_n), .in_valid(in_valid), .in_ready(in_ready),
                   .in_pos(in_pos), .mtx00_33(mtx), .out_valid(out_valid),
                   .out_ready(out_ready), .out_x(out_x), .out_y(out_y), .out_z(out_z),
                   .clipped(clipped));

  always #5 clk = ~clk; // 100MHz for simulation timescale

  integer i;
  initial begin
    #1 rst_n = 0; #20 rst_n = 1;
    // load identity matrix Q16.16
    mtx = {32'h00010000,32'h0,32'h0,32'h0,
           32'h0,32'h00010000,32'h0,32'h0,
           32'h0,32'h0,32'h00010000,32'h0,
           32'h0,32'h0,32'h00010000,32'h00010000}; // last row sets w=1
    // directed tests: three vertices
    reg [31:0] vx[0:2], vy[0:2], vz[0:2];
    vx[0]=32'h00020000; vy[0]=32'h00010000; vz[0]=32'h00000000; // (2,1,0)
    vx[1]=32'h00000000; vy[1]=32'h00000000; vz[1]=32'h00000000; // origin
    vx[2]=32'hFFFF0000; vy[2]=32'h00030000; vz[2]=32'h00000000; // (-1,3,0)
    for (i=0;i<3
\chapter{Chapter 6: Rasterization}
\section{Section 1: Tile-Based Scan Conversion}
\subsection{Item 1: Screen space subdivision}
Building on triangle setup and edge-equation derivations from the vertex-processing stage, we now show how to partition screen space into tiles that exploit spatial locality and enable early coarse culling before per-pixel shading. The following develops the math for tile-level rejection, derives a compact hardware interface for a subdivision unit, and gives a synthesizable Verilog implementation suitable for integration with a binning/raster pipeline.

Tile subdivision problem: given triangle screen coordinates in fixed-point, produce a conservative tile bounding box and edge coefficients so downstream hardware can iterate tiles and quickly decide coverage. Use fixed-point with $S$ integer and $F$ fractional bits to preserve subpixel precision; tile size is power-of-two to allow shifts.

Edge function and conservative tile test. For triangle vertices $(x_1,y_1),(x_2,y_2),(x_3,y_3)$ the oriented edge function for edge $i$ is
\begin{equation}[H]\label{eq:edge}
E_i(x,y) = A_i x + B_i y + C_i,
\end{equation}
where $A_i = y_j - y_k$, $B_i = x_k - x_j$, and $C_i = x_j y_k - y_j x_k$ for cyclic indices $(i,j,k)$. A tile with corner set $T=\{(x_{t0},y_{t0}),\dots\}$ is trivially outside the triangle if any edge has $E_i(x_{tc},y_{tc})<0$ for all corners $c$. Conservative acceptance uses corner evaluations only; this avoids per-pixel tests at tile granularity.

Bounding-box to tile indices. Compute integer tile indices via
\begin{equation}[H]\label{eq:tileidx}
\mathrm{tileX}_{\min} = \left\lfloor\frac{\min(x_1,x_2,x_3)}{T_w}\right\rfloor,\quad
\mathrm{tileX}_{\max} = \left\lfloor\frac{\max(x_1,x_2,x_3)}{T_w}\right\rfloor,
\end{equation}
with analogous expressions for Y. If tiles are power-of-two $T_w=2^k$, division becomes a logical right-shift by $k$ bits in fixed-point representation.

Implementation choices and interface. The unit emits:
\begin{itemize}
\item tile bbox: min/max tile indices clamped to framebuffer tile grid,
\item edge coefficients $A_i,B_i,C_i$ in the same fixed-point format,
\item tile corner offsets for fast corner evaluation (precomputed multiples of $T_w$).
\end{itemize}
This keeps the raster SMs/tiler simple: they iterate tiles in the range, fetch edges, and perform four corner evaluations per tile. The corner arithmetic is low-cost integer multiplies and adds, amenable to SIMD/SIMT lanes or dedicated tile engines.

\begin{lstlisting}[language=Verilog,caption={Tile subdivision unit: computes tile bbox and edge coefficients (synthesizable)},label={lst:tile_subdiv}]
module tile_subdivider #(
  parameter IW=24, // total bits for fixed-point coordinates
  parameter FW=8,  // fractional bits
  parameter TILE_LOG2=4 // tile width = 2^TILE_LOG2
)(
  input  wire                 clk,
  input  wire                 rst,
  input  wire                 valid_in,
  input  wire signed [IW-1:0] v0x, v0y, // vertex 0 (fixed-point)
  input  wire signed [IW-1:0] v1x, v1y, // vertex 1
  input  wire signed [IW-1:0] v2x, v2y, // vertex 2
  input  wire [15:0]          fb_tiles_x, fb_tiles_y, // tile counts
  output reg                  valid_out,
  output reg  [15:0]          tile_x_min, tile_x_max,
  output reg  [15:0]          tile_y_min, tile_y_max,
  output reg signed [IW+1:0]  A0,B0,C0, A1,B1,C1, A2,B2,C2
);

  // Local min/max in fixed-point
  wire signed [IW-1:0] xmin = (v0x < v1x) ? ((v0x < v2x) ? v0x : v2x)
                                         : ((v1x < v2x) ? v1x : v2x);
  wire signed [IW-1:0] xmax = (v0x > v1x) ? ((v0x > v2x) ? v0x : v2x)
                                         : ((v1x > v2x) ? v1x : v2x);
  wire signed [IW-1:0] ymin = (v0y < v1y) ? ((v0y < v2y) ? v0y : v2y)
                                         : ((v1y < v2y) ? v1y : v2y);
  wire signed [IW-1:0] ymax = (v0y > v1y) ? ((v0y > v2y) ? v0y : v2y)
                                         : ((v1y > v2y) ? v1y : v2y);

  // Tile index extraction by shifting out fractional and tile bits
  wire [31:0] tx_min_w = $unsigned(xmin) >> (FW + TILE_LOG2);
  wire [31:0] tx_max_w = $unsigned(xmax) >> (FW + TILE_LOG2);
  wire [31:0] ty_min_w = $unsigned(ymin) >> (FW + TILE_LOG2);
  wire [31:0] ty_max_w = $unsigned(ymax) >> (FW + TILE_LOG2);

  // Edge coefficients (wider to avoid overflow)
  wire signed [IW+1:0] a0 = v1y - v2y;
  wire signed [IW+1:0] b0 = v2x - v1x;
  wire signed [IW+1:0] c0 = (v1x * v2y) - (v1y * v2x);
  wire signed [IW+1:0] a1 = v2y - v0y;
  wire signed [IW+1:0] b1 = v0x - v2x;
  wire signed [IW+1:0] c1 = (v2x * v0y) - (v2y * v0x);
  wire signed [IW+1:0] a2 = v0y - v1y;
  wire signed [IW+1:0] b2 = v1x - v0x;
  wire signed [IW+1:0] c2 = (v0x * v1y) - (v0y * v1x);

  always @(posedge clk) begin
    if (rst) begin
      valid_out <= 0;
      tile_x_min <= 0; tile_x_max <= 0;
      tile_y_min <= 0; tile_y_max <= 0;
      {A0,B0,C0,A1,B1,C1,A2,B2,C2} <= 0;
    end else if (valid_in) begin
      // clamp to framebuffer tile counts
      tile_x_min <= (tx_min_w[31:0] >= fb_tiles_x) ? fb_tiles_x-1 : tx_min_w[15:0];
      tile_x_max <= (tx_max_w[31:0] >= fb_tiles_x) ? fb_tiles_x-1 : tx_max_w[15:0];
      tile_y_min <= (ty_min_w[31:0] >= fb_tiles_y) ? fb_tiles_y-1 : ty_min_w[15:0];
      tile_y_max <= (ty_max_w[31:0] >= fb_tiles_y) ? fb_tiles_y-1 : ty_max_w[15:0];
      // export edges for downstream tile tests
      A0 <= a0; B0 <= b0; C0 <= c0;
      A1 <= a1; B1 <= b1; C1 <= c1;
      A2 <= a2; B2 <= b2; C2 <= c2;
      valid_out <= 1;
    end else begin
      valid_out <= 0;
    end
  end
endmodule