module cxl_dir #(
  parameter DEPTH = 1024,
  parameter TAG_WIDTH = 48,    // physical tag bits
  parameter OWNER_WIDTH = 8    // ID of owning agent (CPU/GPU)
)(
  input  wire                    clk,
  input  wire                    rst,
  input  wire                    req_valid,
  input  wire [TAG_WIDTH-1:0]    req_tag,
  input  wire                    req_write,    // 1 = write, 0 = read
  output reg                     resp_valid,
  output reg  [OWNER_WIDTH-1:0]  resp_owner,
  output reg  [1:0]              resp_state     // 00=I,01=S,10=M
);
  // storage arrays
  reg [TAG_WIDTH-1:0]   tag_mem [0:DEPTH-1];
  reg [OWNER_WIDTH-1:0] owner_mem [0:DEPTH-1];
  reg                   valid_mem [0:DEPTH-1];
  reg [1:0]             state_mem [0:DEPTH-1];

  integer i;
  // combinational hit vector
  reg [DEPTH-1:0] hit_vec;
  always @(*) begin
    for (i=0;i
\subsection{Item 4:  Topologies for multi-GPU systems}
These points build directly on the preceding discussion of coherent interconnects and NVLink-style fabrics: coherence reduces software overhead for shared-address models, while point-to-point link performance determines which topology yields the best effective bandwidth for graphics and ML workloads. The following examines topologies, models their collective performance cost, and presents a small tool to compare AllReduce cost under different link layouts.

Problem — when multiple GPUs must exchange large tensors or framebuffers (graphics share, multi-GPU ML), topology determines both raw bisection bandwidth and contention. Key topologies:
\begin{itemize}
\item PCIe tree (root complex + switches): cheap, ubiquitous; per-link bandwidth limited to PCIe Gen4/5 x16 ($\sim$16–32 $\mathrm{GB\,s^{-1}}$ per direction); often non-uniform hop counts to the CPU or to other GPUs.
\item Point-to-point mesh (pairwise NVLink): direct high-bandwidth links (25–50+ $\mathrm{GB\,s^{-1}}$ per link) connecting selected GPU pairs; low hop count but limited degree without a switch.
\item Switch-based fabric (NVSwitch, InfiniBand/Aries switches): provides full bisection, constant hop counts, and virtualization of links; highest cost and power.
\item Ring/daisy-chain: minimal links, simple wiring, but poor bisection and high latency for large $p$.
\item Fat-tree: used in large-scale clusters (IB fat-tree), scales bisection bandwidth with cost.
\end{itemize}

Analysis — use the latency-bandwidth alpha–beta model to compare collective costs. For $p$ GPUs performing an AllReduce of size $S$ (bytes) with per-link bandwidth $B$ (bytes/s) and per-hop latency $\alpha$, the ring AllReduce requires $2(p-1)$ steps where each step transfers $S/p$ bytes, so:
\begin{equation}[H]\label{eq:allreduce_ring}
T_{\text{ring}} \;=\; 2(p-1)\alpha \;+\; 2\frac{p-1}{p}\frac{S}{B}.
\end{equation}
In contrast, a switch-based all-to-all can complete in $O(\log p)$ rounds via tree-based algorithms with
\begin{equation}[H]\label{eq:allreduce_tree}
T_{\text{tree}} \;=\; c\log p\cdot\alpha \;+\; \frac{2S}{B},
\end{equation}
where $c$ is a small constant depending on algorithmic phases. The math shows topologies with lower diameter (switch, full mesh) reduce $\alpha$-dominated cost; high per-link $B$ reduces the bandwidth term.

Implementation — practical evaluation requires modeling per-link degrees and oversubscription. The script below computes $T_{\text{ring}}$ and $T_{\text{tree}}$ for configurable $p$, $S$, $B$, $\alpha$. Use it to quantify trade-offs for TMU-heavy texture streaming vs tensor reduction in ML.

\begin{lstlisting}[language=Python,caption={AllReduce cost calculator for ring vs tree topologies},label={lst:allreduce_calc}]
# simple alpha-beta AllReduce model; sizes in bytes, B in bytes/s, alpha in seconds
def allreduce_ring_time(p, S, B, alpha):
    return 2*(p-1)*alpha + 2*(p-1)/p * (S / B)

def allreduce_tree_time(p, S, B, alpha, c=2):
    import math
    return c*math.log2(p)*alpha + 2*(S / B)

# example: 8 GPUs, 1GB tensor, NVLink B=40GB/s, alpha=1e-6s
if __name__ == "__main__":
    p, S = 8, 1<<30
    B_nvlink = 40*(1<<30)  # bytes/s
    alpha = 1e-6
    print("Ring:", allreduce_ring_time(p,S,B_nvlink,alpha))
    print("Tree:", allreduce_tree_time(p,S,B_nvlink,alpha))