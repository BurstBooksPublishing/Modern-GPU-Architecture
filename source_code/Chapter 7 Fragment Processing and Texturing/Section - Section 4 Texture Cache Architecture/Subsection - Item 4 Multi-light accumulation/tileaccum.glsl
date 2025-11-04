#version 450
layout(binding=0) uniform sampler2D normalMap; // normal map
layout(std430, binding=1) buffer LightList { vec4 lights[]; }; // packed per-light params
layout(std430, binding=2) buffer TileIndex { int tileOffset[]; }; // per-tile offsets
in vec2 uv; in vec3 viewDir; // interpolants
out vec4 fragColor;
void main() {
  vec3 N = texture(normalMap, uv).xyz * 2.0 - 1.0; // fetch once
  int tile = computeTile(gl_FragCoord.xy); // cheap integer division
  int offset = tileOffset[tile];
  int count = int(lights[offset].w); // stored count in w
  vec3 accum = vec3(0.0);
  for(int i=1;i<=count;++i){ // loop over tile lights
    vec4 p = lights[offset + i]; // x,y,z,intensity packed
    float d = length(p.xyz - gl_FragCoord.xyz); // distance
    float att = 1.0/(p.w + d*d); // simple attenuation
    vec3 L = normalize(p.xyz - gl_FragCoord.xyz);
    float NdotL = max(dot(N,L),0.0);
    // simple lambert + blinn-phong specular
    vec3 H = normalize(L + viewDir);
    float spec = pow(max(dot(N,H),0.0), 16.0);
    accum += att * (NdotL + spec) * vec3(p.w); // p.w reused as intensity
  }
  fragColor = vec4(accum + vec3(0.05),1.0); // add ambient
}