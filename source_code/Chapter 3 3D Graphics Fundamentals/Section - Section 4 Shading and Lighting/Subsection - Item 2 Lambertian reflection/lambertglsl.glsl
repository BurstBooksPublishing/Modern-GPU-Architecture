#version 450
in vec3 vViewPos;             // view-space position
in vec3 vTangent;             // interpolated tangent
in vec3 vBitangent;           // interpolated bitangent
in vec2 vUV;                  // texture coords
layout(binding=0) uniform sampler2D uNormalMap;
layout(std140, binding=1) uniform LightBlock { vec4 lights[128]; int nLights; }; // packed lights
out vec4 fragColor;

void main() {
  // fetch normal map (stored [0,1]); convert to [-1,1]
  vec3 nmap = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
  // build TBN matrix; assumes interpolated tangent/bitangent are orthonormal-ish
  mat3 TBN = mat3(normalize(vTangent), normalize(vBitangent), normalize(cross(vTangent, vBitangent)));
  vec3 N = normalize(TBN * nmap); // normalize in fragment shader
  vec3 color = vec3(0.0);
  for (int i=0; i
\subsection{Item 3:  Specular highlights}
Building on Lambertian diffuse terms and the per-vertex versus per-fragment trade-offs discussed in Gouraud and Phong shading, we now examine how specular highlights model view-dependent shiny behavior and how GPUs implement them efficiently. The focus is on numerical form, shader implementation, and microarchitectural costs under SIMT execution.

Specular reflection models produce bright, view-dependent peaks where the surface normal aligns with reflected or half-vector directions. Two common models are Phong and Blinn–Phong. Phong computes the reflection vector $R=\text{reflect}(-L,N)$ and evaluates
\begin{equation}[H]\label{eq:phong}
S_{\text{Phong}} = k_s \max(0, R\cdot V)^{\alpha},
\end{equation}
where $L$ is the light direction, $N$ the surface normal, $V$ the view vector, $k_s$ the specular coefficient, and $\alpha$ the shininess exponent. Blinn–Phong uses the half-vector $H=\frac{L+V}{\|L+V\|}$ and is numerically cheaper in many shader pipelines:
\begin{equation}[H]\label{eq:specular}
S_{\text{Blinn}} = k_s \max(0, N\cdot H)^{\alpha}.
\end{equation}

Analysis: the dominant costs for per-fragment specular are:
\begin{itemize}
\item transcendental power evaluation for exponent $\alpha$, typically implemented by a hardware pow instruction or approximated with $\exp/\log$ sequences.
\item normalization and vector ops (dot, add, normalize), which consume ALU throughput and register pressure.
\item texture lookups for normal, roughness, or gloss maps increasing TMU use and bandwidth.
\end{itemize}

Implementation: a production fragment shader commonly uses normal mapping and a roughness (gloss) texture to vary $\alpha$ spatially. The following GLSL fragment implements Blinn–Phong specular with a gloss map; comments note performance-sensitive lines.

\begin{lstlisting}[language=GLSL,caption={Fragment shader: Blinn–Phong specular with normal and gloss maps},label={lst:spec_shader}]
#version 450
in vec3 vNormal; in vec3 vView; in vec3 vLightDir; in vec2 vUV;
layout(binding=0) uniform sampler2D normalMap; // normal map (TEX)
layout(binding=1) uniform sampler2D glossMap;   // gloss/roughness map
uniform vec3 ks; uniform float shininessScale;   // specular color and scale
out vec4 fragColor;
void main() {
  // fetch normal, convert from [0,1] to [-1,1]
  vec3 n = texture(normalMap, vUV).xyz * 2.0 - 1.0;
  n = normalize(n); // per-fragment normalize cost
  vec3 v = normalize(vView);
  vec3 l = normalize(vLightDir);
  vec3 h = normalize(l + v); // half vector
  float gloss = texture(glossMap, vUV).r; // gloss in red channel
  float alpha = max(1.0, gloss * shininessScale); // map gloss to exponent
  float nDotH = max(0.0, dot(n, h));
  float spec = pow(nDotH, alpha); // expensive op: pow
  vec3 specular = ks * spec;
  fragColor = vec4(specular, 1.0);
}