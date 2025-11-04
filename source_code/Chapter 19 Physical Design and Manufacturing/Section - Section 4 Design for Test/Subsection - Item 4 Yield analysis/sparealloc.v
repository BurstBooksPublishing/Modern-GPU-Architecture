module spare_row_manager #(
  parameter NUM_COLS = 256,
  parameter NUM_SPARES = 8
)(
  input  wire                     clk,
  input  wire                     rst_n,
  input  wire                     start,             // trigger mapping from fault_mask
  input  wire [NUM_COLS-1:0]      fault_mask,        // 1 = faulty column
  output reg  [31:0]              spare_used_count,  // number of spares consumed
  output reg                      done,
  output reg  [$clog2(NUM_COLS)-1:0] remap_table [NUM_SPARES-1:0] // maps spare i -> replaced column idx
);
  // simple FSM: scan fault_mask and allocate sequential spares
  integer i;
  reg [$clog2(NUM_COLS)-1:0] next_idx;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      spare_used_count <= 0;
      done <= 1'b0;
      for (i=0;i
\section{Section 5: Packaging Technologies}
\subsection{Item 1:  Flip-chip and BGA packaging}
Following place-and-route and IR-drop analysis, packaging choices close the physical-design loop by determining the PDN, signal integrity, and thermal path available to the many SMs and tensor cores on a large GPU die. The flip-chip + BGA combination solves the electrical and thermal needs of high-density SIMT processors by minimizing loop inductance, enabling large I/O counts, and providing a direct heat conduction path to the board-level heat spreader.

Problem: modern GPU dies deliver hundreds of watts in a few square centimeters, so packaging must satisfy three coupled constraints: low PDN impedance across DC–GHz, minimal signal and power inductance for fast TMU/ROP interfaces, and low thermal resistance from junction to ambient. Analysis:

\begin{itemize}
\item Electrical: flip-chip uses an array of solder bumps (microbumps/C4) to connect die pads directly to an organic substrate or interposer. Short vertical paths reduce current-loop area and series inductance compared to perimeter wire-bonds, improving transient voltage droop and enabling tighter decoupling budgets. The available I/O count $N$ scales inversely with bump pitch $p$:
\begin{equation}[H]\label{eq:io_count}
N \approx \frac{A_{\text{die}}}{p^{2}},
\end{equation}
so reducing pitch increases routing complexity but enables distributed PDN and many high-speed SERDES lanes or memory channels.
\item Thermal: the dominant metric is junction-to-case thermal resistance \lstinline|RθJC|. For steady-state power $P$, junction rise is
\begin{equation}[H]\label{eq:deltaT}
\Delta T = P \cdot R_{\theta JC},
\end{equation}
making low thermal-resistance solder bumps and thermal interfaces critical for SM frequency scaling.
\end{itemize}

Implementation notes (assembly and materials):

\begin{enumerate}
\item Die attach and reflow using controlled atmosphere; bumps formed as solder paste or preformed balls.
\item Substrate options:
   \begin{itemize}
   \item Organic BGA substrates: lower cost, high layer count for routing and embedded capacitance.
   \item Silicon interposer (2.5D): used for HBM stacks, supports TSVs and microbumps.
   \end{itemize}
\item Underfill epoxy reduces CTE-driven shear on bumps, improving thermal cycling reliability.
\item TI materials: thermal interface material (TIM) + heatspreader + heatsink attach for spreading heat from bump region.
\end{enumerate}

Practical calculation and quick checks: a small script computes junction temperature rise and approximate I/O count for a target pitch.

\begin{lstlisting}[language=Python,caption=Package quick-check: junction temp and IO count,label={lst:pkg_check}]
# simple calculator for junction rise and IO count
A_die_mm2 = 600.0        # die area in mm^2 (e.g., large GPU)
pitch_mm = 0.5           # bump pitch in mm
P_W = 300.0              # package power in watts
R_theta_JC = 0.08        # junction-to-case R (degC/W) -- engineering estimate

# I/O count estimate (ignore keepout and power pad allocation)
N_io = int((A_die_mm2) / (pitch_mm**2))
deltaT = P_W * R_theta_JC

print(f"Estimated I/O count: {N_io} balls")   # many balls enable distributed PDN
print(f"Junction rise (degC): {deltaT:.1f}") # drives TIM and heatsink design