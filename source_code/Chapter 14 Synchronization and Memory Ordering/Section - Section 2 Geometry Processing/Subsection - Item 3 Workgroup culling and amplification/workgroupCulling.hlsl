TaskPayload TaskMain(uint3 dispatchId : SV_DispatchThreadID)
{
    // shared aggregator in threadgroup memory (declared elsewhere)
    // Each thread loads one meshlet's bounding sphere (center, radius)
    Meshlet mesh = LoadMeshlet(dispatchId.x);                  // read-only fetch
    float3 viewPos = mul(viewMatrix, float4(mesh.center,1)).xyz;
    float z = viewPos.z;
    float rScreen = mesh.radius * focalLength / max(0.0001, z); // eq. (1)
    bool visible = FrustumTest(viewPos, mesh.radius);
    if (visible && (3.14159 * rScreen * rScreen) > areaToEmit) {
        // compute amplification factor adaptively
        uint ampl = min(Amax, (uint)clamp(rScreen / amplScale, 1, (float)Amax));
        // compact into threadgroup list using atomic add to groupCount
        uint idx = 0;
        if (ThreadIsLeader()) {
            // leader writes groupCount to global atomic to reserve space
            idx = InterlockedAdd(globalIndirectCounter, groupCountLocal);
        }
        GroupBarrier();
        // write ampl entries (one per amplified instance) into indirect buffer
        for (uint a=0; a
\subsection{Item 4:  Hardware resource mapping}
Building on task/mesh pipeline sequencing and the culling/amplification strategies, hardware resource mapping ties those dynamic workloads to fixed SM/CU resources so that amplified meshlets run without starving registers, shared memory, or thread slots.

The problem: mesh shaders emit meshlets with variable vertex and primitive counts, and task shaders can amplify work unpredictably. The allocator must satisfy three hard constraints per SM: register file capacity, shared/LDS capacity, and maximum concurrent thread slots (warps/wavefronts). Practical mapping therefore becomes a constrained packing problem with additional locality objectives (co-locating connected meshlets to reduce texture/TMU and L1 pressure) and time multiplexing to hide latency.

Analysis: the number of resident workgroups (meshlet thread-groups) per SM is the bottlenecked minimum of resource ratios:
\begin{equation}[H]\label{eq:occupancy}
N_{\text{resident}}=\min\left(\left\lfloor\frac{R_{\text{SM}}}{R_{\text{wg}}}\right\rfloor,\ \left\lfloor\frac{S_{\text{SM}}}{S_{\text{wg}}}\right\rfloor,\ \left\lfloor\frac{T_{\text{SM}}}{T_{\text{wg}}}\right\rfloor\right)
\end{equation}
where $R$=registers, $S$=shared memory bytes, $T$=thread slots. Designers must compute $R_{\text{wg}}$ and $S_{\text{wg}}$ at compile-time when possible; for mesh shaders they are often runtime-estimated, so conservative static reservation or a two-phase allocation (speculative launch then trim) is used.

Implementation recommendations:
\begin{itemize}
\item Use a small hardware allocator per dispatch unit that tracks per-SM free resources and applies a best-fit-first policy to maximize locality (prefer SMs with texture cache residency for that meshlet).
\item Support dynamic reclamation: when meshlet finishes, free registers and LDS atomically to enable immediate dispatch from task shader amplification.
\item Offer a lightweight per-meshlet descriptor in a shared queue with fields: vertex_count, primitive_count, expected_registers, expected_smem, locality_hint (texture cache slice). The scheduler computes a fast-fit via parallel comparators.
\end{itemize}

A compact synthesizable allocator FSM in Verilog that implements first-fit allocation across NUM_SMS SMs is below; it accepts request vectors and returns a grant index when resources suffice.

\begin{lstlisting}[language=Verilog,caption={Simple SM resource allocator (synthesizable).},label={lst:sm_alloc}]
module sm_alloc #(
  parameter NUM_SMS = 8,
  parameter REG_WIDTH = 16,   // bits for reg counts
  parameter SMEM_WIDTH = 20,  // bytes count width
  parameter IDX_WIDTH = 3
)(
  input  wire                    clk,
  input  wire                    rstn,
  input  wire                    req_valid,
  input  wire [REG_WIDTH-1:0]    req_regs,
  input  wire [SMEM_WIDTH-1:0]   req_smem,
  input  wire [15:0]             req_threads, // threads needed
  output reg                     grant_valid,
  output reg  [IDX_WIDTH-1:0]    grant_idx
);
  // Per-SM free resources
  reg [REG_WIDTH-1:0]  free_regs [0:NUM_SMS-1];
  reg [SMEM_WIDTH-1:0] free_smem [0:NUM_SMS-1];
  reg [15:0]           free_threads [0:NUM_SMS-1];

  integer i;
  // init example capacities (could be loaded from config)
  initial begin
    for (i=0;i= req_regs && free_smem[i] >= req_smem && free_threads[i] >= req_threads) begin
            // allocate resources
            free_regs[i]   <= free_regs[i] - req_regs;
            free_smem[i]   <= free_smem[i] - req_smem;
            free_threads[i]<= free_threads[i] - req_threads;
            grant_valid <= 1'b1;
            grant_idx <= i[IDX_WIDTH-1:0];
          end
        end
      end
    end else begin
      grant_valid <= 1'b0;
    end
  end

  // Deallocation interface would be separate in full design (not shown).
endmodule