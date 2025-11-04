#version 450
layout(location = 0) in vec3 inPosition; // object-space pos
layout(location = 1) in vec3 inNormal;   // object-space normal

layout(binding = 0) uniform UBO {
    mat4 model;        // model matrix
    mat4 view;         // view matrix
    mat4 proj;         // projection matrix
    mat3 normalMat;    // (model^{-1})^T 3x3 normal matrix
} ubo;

layout(location = 0) out vec3 vNormal;   // varying for fragment shader

void main() {
    mat4 mvp = ubo.proj * ubo.view * ubo.model;      // precompose MVP per-draw
    vec4 clipPos = mvp * vec4(inPosition, 1.0);      // clip-space position
    gl_Position = clipPos;                           // hardware will perspective-divide
    vNormal = normalize(ubo.normalMat * inNormal);  // transformed normal
}