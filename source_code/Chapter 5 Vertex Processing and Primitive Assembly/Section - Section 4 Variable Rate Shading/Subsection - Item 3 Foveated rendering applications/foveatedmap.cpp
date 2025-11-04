#include 
#include 
#include 
// Produce shading rates per tile: 1..maxRate (1 = full)
// gazeX/Y in pixels, width/height screen, tileSize in pixels.
std::vector buildFoveaRateMap(int width,int height,int tileSize,
                                    float gazeX,float gazeY,
                                    float A,float E0,float gamma,int Smax){
  int tilesX = (width + tileSize-1)/tileSize;
  int tilesY = (height + tileSize-1)/tileSize;
  std::vector rates(tilesX*tilesY);
  for(int ty=0; ty
\subsection{Item 4:  Bandwidth and power benefits}
Building on the perceptual steering used in foveated rendering and the tile-aligned shading-rate maps described previously, we now quantify how VRS (variable rate shading) reduces memory bandwidth and dynamic power across the raster pipeline. The goal is to translate a shading-rate map into concrete bandwidth and power savings for SMs, TMUs, and ROP/L2/VRAM subsystems.

Problem statement and model assumptions:
\begin{itemize}
\item Rasterization produces $N$ screen pixels arranged in $T$ tiles; each tile $t$ has a shading rate $r_t$ expressed as the fraction of baseline shading work (e.g., $r=1$ for per-pixel, $r=0.25$ for one quarter rate).
\item Each shaded sample triggers an average of $f_{\text{tex}}$ texture fetches hitting TMUs and $C_{\text{alus}}$ shader ALU cycles.
\item Output writes per shaded sample consume $b_{\text{out}}$ bytes to the ROP/cache and may generate additional compressed metadata.
\end{itemize}

Analysis (bandwidth first). Effective shaded-sample count $S$ under a shading-rate map:
\begin{equation}[H]\label{eq:effective_samples}
S \;=\; \sum_{t=1}^{T} r_t \cdot P_t,
\end{equation}
where $P_t$ is pixels per tile and baseline $S_{\text{base}}=N$. If average per-sample memory traffic is $M_s$ (texture fetch bytes + write-back bytes after filtering/compression), then total bandwidth demand is $BW = S\cdot M_s$. Relative bandwidth reduction is:
\begin{equation}[H]\label{eq:bandwidth_reduction}
\text{Reduction} \;=\; 1 - \frac{S}{N}.
\end{equation}
This simple proportional model is conservative: real systems see super-linear gains because reduced shading often improves cache hit rates (lower TMU pressure) and reduces ROP write-back frequency, so $M_s$ can also decrease with VRS.

Power model (operational relevance). Dynamic power decomposes to memory-bound and compute-bound components:
\begin{equation}[H]\label{eq:power_model}
P_{\text{dyn}} \approx \alpha \cdot BW + \beta \cdot U_{\text{ALU}},
\end{equation}
where $\alpha$ is energy per byte transferred (L2+DRAM path) and $U_{\text{ALU}}$ is ALU utilization proportional to $S$. VRS reduces both $BW$ and $U_{\text{ALU}}$, so power savings track shading-rate-weighted reductions in sample counts.

Implementation: quick estimator to convert a shading-rate map into savings. The snippet below computes effective samples and estimated bandwidth/power reductions for a tile grid.

\begin{lstlisting}[language=Python,caption={VRS bandwidth/power estimator},label={lst:vrs_est}]
import numpy as np
# tile_grid: 2D array of shading rates (fractions of baseline)
# pixels_per_tile: scalar, e.g. 16*16
def estimate_savings(tile_grid, pixels_per_tile, M_s, alpha, beta, ALU_per_sample):
    N_tiles = tile_grid.size
    P = pixels_per_tile
    N = N_tiles * P                       # baseline pixels
    S = (tile_grid * P).sum()             # eq. (1)
    BW = S * M_s                          # bytes transferred
    Pdyn = alpha * BW + beta * (S*ALU_per_sample)  # eq. (3)
    return {'baseline_pixels': N, 'effective_shaded': S,
            'bw_bytes': BW, 'p_dyn': Pdyn,
            'bw_reduction': 1 - S/N}
# Example use: tile_grid with center high rate, periphery low rate