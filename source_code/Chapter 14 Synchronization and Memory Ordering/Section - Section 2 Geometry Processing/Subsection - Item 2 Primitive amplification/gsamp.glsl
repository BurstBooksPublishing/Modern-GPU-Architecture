#version 450
layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out; // cap total emitted verts
uniform mat4 projView;
void main() {
    // Emit for each input triangle up to 3 quads (amplification factor <=3)
    for(int q=0;q<3;++q){ // simple bounded amplification
        // compute quad vertex positions (shader-local math) and emit 6 verts -> 2 tris
        for(int v=0; v<6; ++v){
            gl_Position = projView * vec4(gl_in[v%3].gl_Position.xyz + vec3(0.01*q),1.0);
            EmitVertex(); // small comment: emits transformed vertex
        }
        EndPrimitive();
    }
}