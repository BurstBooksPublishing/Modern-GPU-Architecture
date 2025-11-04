#version 450
layout(location=0) out vec4 RT0_albedo_rough;  // rgb: albedo, a: roughness
layout(location=1) out vec4 RT1_normal_metal;  // xy: octahedral normal, z: metal, w: pad
layout(location=2) out vec2 RT2_motion;        // motion vectors
// Octahedral encode (unit normal -> 2D)
vec2 octEncode(vec3 n){
  n /= (abs(n.x)+abs(n.y)+abs(n.z));
  vec2 enc = n.xy;
  if(n.z < 0.0) enc = (1.0 - abs(enc.yx)) * sign(enc.xy);
  return enc * 0.5 + 0.5; // pack into [0,1]
}
void main(){
  vec3 albedo = texture(albedoMap, uv).rgb; // sample textures (TMU)
  float rough = texture(roughnessMap, uv).r;
  float metal = texture(metallicMap, uv).r;
  vec3 normal = normalize((texture(normalMap, uv).rgb * 2.0) - 1.0);
  RT0_albedo_rough = vec4(albedo, rough);
  RT1_normal_metal = vec4(octEncode(normal), metal, 0.0);
  RT2_motion = computeMotion(); // velocity from prev-frame matrices
  gl_FragDepth = gl_FragCoord.z; // depth written separately if desired
}