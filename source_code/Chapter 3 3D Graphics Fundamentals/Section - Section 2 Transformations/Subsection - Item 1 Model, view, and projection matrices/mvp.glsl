#version 450
layout(std140, binding = 0) uniform Matrices {
  mat4 u_Model;   // model matrix
  mat4 u_View;    // view matrix
  mat4 u_Proj;    // projection matrix
} mats;

layout(location = 0) in vec3 in_pos;    // object-space position
layout(location = 1) in vec3 in_norm;   // object-space normal

layout(location = 0) out vec3 v_viewNorm; // pass to fragment stage

void main() {
  mat4 MVP = mats.u_Proj * mats.u_View * mats.u_Model; // compose once per vertex
  vec4 worldPos = mats.u_Model * vec4(in_pos, 1.0);    // optional world-space use
  gl_Position = MVP * vec4(in_pos, 1.0);              // clip-space position
  mat3 normalMat = transpose(inverse(mat3(mats.u_Model))); // normal transform
  v_viewNorm = mat3(mats.u_View) * (normalMat * in_norm);  // view-space normal
}