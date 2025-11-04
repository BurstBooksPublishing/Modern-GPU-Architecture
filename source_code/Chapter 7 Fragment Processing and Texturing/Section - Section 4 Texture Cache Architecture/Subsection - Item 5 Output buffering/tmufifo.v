module tmu_output_fifo #(
  parameter WIDTH = 128,              // payload + metadata bits
  parameter ADDR_WIDTH = 5            // depth = 2^ADDR_WIDTH
) (
  input  wire                   clk,
  input  wire                   rst, // synchronous reset
  // write side (TMU)
  input  wire                   wr_valid,
  input  wire [WIDTH-1:0]       wr_data,
  output wire                   wr_ready,
  // read side (consumer)
  input  wire                   rd_ready,
  output wire                   rd_valid,
  output reg  [WIDTH-1:0]       rd_data
);
  localparam DEPTH = (1<
\section{Section 6: Lighting Calculations}
\subsection{Item 1:  Ambient, diffuse, and specular components}
Following interpolation and texture filtering stages, the fragment pipeline must combine lighting contributions per-pixel to produce physically plausible color while minimizing SM/CU compute and TMU bandwidth. The problem is balancing visual fidelity—ambient, diffuse, and specular contributions—with throughput constraints in a SIMT execution model where many fragments execute identical shader code but may diverge in control flow or texture access.

Analysis: use simple analytic models that map well to GPU datapaths. The canonical Phong/Blinn decomposition separates:
\begin{itemize}
\item ambient term: low-frequency scene illumination approximated as a constant or ambient occlusion texture sample,
\item diffuse term: Lambertian response proportional to the cosine of the incident angle,
\item specular term: mirror-like lobe controlled by a shininess exponent.
\end{itemize}

Combine these into a per-fragment radiance $L$:
\begin{equation}[H]\label{eq:phong}
L = k_a I_a + \sum_{i=1}^{N} \left[ k_{d} I_{i} \max(0, \mathbf{N}\!\cdot\!\mathbf{L}_i) + k_{s} I_{i} \max(0, \mathbf{N}\!\cdot\!\mathbf{H}_i)^{\alpha} \right],
\end{equation}
where $\mathbf{H}_i = \frac{\mathbf{L}_i+\mathbf{V}}{\|\mathbf{L}_i+\mathbf{V}\|}$ for Blinn-Phong, $\mathbf{N}$ is the normal, $\mathbf{L}_i$ and $I_i$ are light direction and intensity, and $\alpha$ is the specular exponent. Use Blinn-Phong because computing $\mathbf{H}$ replaces a costly reflection vector and maps better to FMA-friendly operations on FP units.

Implementation considerations on modern GPUs:
\begin{itemize}
\item Texture and normal maps are fetched by TMUs; compressing normal maps to RGB8 or BC5 reduces bandwidth at the cost of some precision.
\item Compute work per fragment maps to ALUs in SMs; specular exponentiation is expensive if done with $\mathrm{pow}()$. Replace $\mathrm{pow}(x, \alpha)$ with $\exp_2(\alpha\log_2(x))$ only when hardware has fast $\log_2/\exp_2$ or use precomputed shininess LUT in shared memory.
\item Use half precision ($\mathrm{FP16}$) for intermediate products in high-throughput paths when tonal range permits; this increases throughput on tensor-core accelerated $\mathrm{FP16}$ paths only for compute shaders, not fragment ROP paths.
\end{itemize}

A compact GLSL-style fragment implementation (Blinn-Phong with normal map and single distant light):
\begin{lstlisting}[language=GLSL,caption={Blinn-Phong per-fragment lighting (single directional light)},label={lst:blinn_phong}]
uniform sampler2D colorMap; // base color
uniform sampler2D normalMap; // normal in tangent-space
uniform vec3 lightDir; // normalized light direction in tangent-space
uniform vec3 viewDir;  // normalized view vector in tangent-space
uniform vec3 Ia;       // ambient intensity
uniform float shininess;
in vec2 fragUV;        // interpolated UV
out vec4 outColor;
void main() {
  vec3 albedo = texture(colorMap, fragUV).rgb;      // TMU fetch
  vec3 n = texture(normalMap, fragUV).xyz * 2.0 - 1.0; // unpack normal
  float NdotL = max(dot(n, lightDir), 0.0);
  vec3 H = normalize(lightDir + viewDir);
  float spec = pow(max(dot(n, H), 0.0), shininess);
  vec3 color = albedo * (Ia + NdotL) + spec; // ambient+diffuse+specular
  outColor = vec4(color, 1.0);
}