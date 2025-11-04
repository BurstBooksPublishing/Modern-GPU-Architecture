#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out; // up to 2 tris per input
layout(triangle_strip, max_vertices = 6) out;
uniform float expand; // expansion factor
void main() {
  // compute centroid of input triangle
  vec3 centroid = (gl_in[0].gl_Position.xyz +
                   gl_in[1].gl_Position.xyz +
                   gl_in[2].gl_Position.xyz) / 3.0;
  for(int i=0;i<3;++i) {
    // emit original vertex
    gl_Position = gl_in[i].gl_Position;
    gl_Layer = int(gl_in[0].gl_Position.w) % 4; // example layered render
    EmitVertex();
  }
  EndPrimitive();
  // emit expanded triangle (silhouette extrude) as additional primitive
  for(int i=0;i<3;++i) {
    vec4 dir = normalize(gl_in[i].gl_Position - vec4(centroid,0.0));
    gl_Position = gl_in[i].gl_Position + vec4(dir.xyz * expand, 0.0);
    gl_Layer = (gl_Layer + 1) % 4; // route to different layer
    EmitVertex();
  }
  EndPrimitive();
}