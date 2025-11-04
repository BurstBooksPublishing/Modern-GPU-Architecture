#version 450
layout(local_size_x=8, local_size_y=8) in;
layout(binding=0, rgba8) uniform writeonly image2D rateImage; // per-tile rates
uniform vec2 gaze; // normalized [0,1] gaze pos
uniform sampler2D depthTex;
void main() {
  ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
  // sample depth variance in tile (brief) -> drives coarse detail metric
  float var = textureGather(depthTex, vec2(tile)/vec2(imageSize(rateImage))).r; // approx
  float dist = distance((vec2(tile)+0.5)/vec2(imageSize(rateImage)), gaze);
  int rate = (dist < 0.05 && var > 0.01) ? 1 : (dist < 0.2 ? 2 : 4); // 1=full, 2=half,4=quarter
  imageStore(rateImage, tile, vec4(rate,0,0,0));
}