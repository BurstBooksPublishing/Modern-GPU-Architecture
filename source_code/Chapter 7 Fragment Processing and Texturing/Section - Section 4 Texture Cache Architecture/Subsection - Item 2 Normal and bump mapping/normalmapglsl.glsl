uniform sampler2D normalMap;         // normal map in linear space
in vec2 v_uv;                        // interpolated UV
in vec3 v_tangent;                   // interpolated tangent (w holds bitangent sign)
in vec3 v_normal;                    // interpolated geometric normal
out vec4 fragColor;

void main() {
    vec3 nmap = texture(normalMap, v_uv).rgb;          // sample TMU
    vec3 n_t = normalize(2.0 * nmap - 1.0);            // decode to [-1,1]
    vec3 t = normalize(v_tangent);
    vec3 b = normalize(cross(v_normal, t) * sign(v_tangent.w)); // reconstruct bitangent
    mat3 TBN = mat3(t, b, normalize(v_normal));       // build basis
    vec3 n_world = normalize(TBN * n_t);              // transform + renormalize
    // lighting uses n_world for diffuse/specular
    fragColor = vec4(n_world * 0.5 + 0.5, 1.0);       // debug output
}