#version 450
layout(location=0) in vec3 inPos; // vertex position
void main() {
    gl_Position = /* model-view-proj */ vec4(inPos,1.0); // minimal vertex work
}
// No fragment shader needed if pipeline supports depth-only; else use empty FS.
// Driver: bind depth-only pipeline; disable color writes; draw geometry. // reduce ROP bandwidth