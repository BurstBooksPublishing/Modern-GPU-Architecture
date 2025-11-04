#version 450
layout(location=0) in vec3 inPos, inNormal;
layout(location=0) out vec3 vColor; // for Gouraud
layout(location=1) out vec3 vNormal; // for Phong

uniform mat4 uMVP, uModel;
uniform vec3 uLightPos, uViewPos;
uniform vec3 kd, ks, ambient;
uniform float shininess;
uniform bool uUsePhong;

vec3 lighting(vec3 N, vec3 P) { // Blinn-Phong
  vec3 L = normalize(uLightPos - P);
  vec3 V = normalize(uViewPos - P);
  vec3 H = normalize(L + V);
  float diff = max(dot(N,L), 0.0);
  float spec = (diff>0.0) ? pow(max(dot(N,H),0.0), shininess) : 0.0;
  return ambient + kd*diff + ks*spec;
}

void main() {
  vec4 worldPos = uModel * vec4(inPos,1.0);
  vNormal = mat3(uModel) * inNormal; // interpolate for Phong
  if(!uUsePhong) {
    vColor = lighting(normalize(vNormal), worldPos.xyz); // Gouraud
  }
  gl_Position = uMVP * vec4(inPos,1.0);
}

// Fragment shader would either consume vColor directly (Gouraud) or
// normalize vNormal and call lighting(...) per-fragment (Phong).