#version 450
in vec3 v_world_pos;   // from vertex shader
in vec3 v_normal;      // interpolated normal
uniform sampler2D tex; // texture
uniform vec2 texSize;  // texture dimensions
out vec4 fragColor;

vec2 planarUV(vec3 pos, int plane) {
  // 0: XY, 1: YZ, 2: XZ
  if (plane==0) return pos.xy;
  if (plane==1) return pos.yz;
  return pos.xz;
}

void main() {
  vec3 n = normalize(v_normal);
  // blending weights from absolute normal
  vec3 w = abs(n);
  w /= (w.x + w.y + w.z);
  // compute per-plane UVs
  vec2 uv0 = planarUV(v_world_pos, 0);
  vec2 uv1 = planarUV(v_world_pos, 1);
  vec2 uv2 = planarUV(v_world_pos, 2);
  // perspective-correct interpolation already done by rasterizer for v_world_pos/v_normal
  // compute derivatives for each projection and choose conservative LOD
  vec2 dudx = ddx(uv0) * texSize;
  vec2 dudy = ddy(uv0) * texSize;
  float rho = max(length(dudx), length(dudy));
  float lod = max(0.0, log2(rho)); // clamp as needed
  // sample each projection with explicit LOD
  vec4 c0 = textureLod(tex, uv0, lod);
  vec4 c1 = textureLod(tex, uv1, lod);
  vec4 c2 = textureLod(tex, uv2, lod);
  fragColor = c0 * w.x + c1 * w.y + c2 * w.z; // blend
}