#version 450
layout(location=0) in vec3 a_pos;    // vertex position
layout(location=1) in vec2 a_uv;     // texture coord
layout(push_constant) uniform U { mat4 u_MVP; } u; // MVP matrix

out vec3 v_uv_over_w; // pass a_uv divided by clip.w
out float v_recip_w;  // pass 1/clip.w

void main() {
  vec4 clip = u.u_MVP * vec4(a_pos, 1.0);    // 4x4 multiply
  v_uv_over_w = vec3(a_uv, 0.0) / clip.w;   // pack uv/w (z unused)
  v_recip_w = 1.0 / clip.w;                  // reciprocal w
  gl_Position = clip;                        // rasterizer uses clip pos
}