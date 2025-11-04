#version 450
layout(location=0) in vec3 hdrColor; // scene-linear RGB
layout(push_constant) uniform PC { float exposure; float white; } pc;
out vec4 outColor;
float luminance(vec3 c){ return dot(c, vec3(0.2126,0.7152,0.0722)); } // Rec.709 Luma
void main(){
  vec3 c = hdrColor * pc.exposure;                    // apply exposure
  float L = luminance(c);
  float Lm = (L*(1.0 + L/(pc.white*pc.white)))/(1.0 + L); // eq. (1)
  vec3 mapped = c * (Lm / max(L,1e-6));               // scale chroma
  mapped = pow(mapped, vec3(1.0/2.2));                // display gamma
  outColor = vec4(clamp(mapped,0.0,1.0),1.0);
}