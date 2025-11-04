vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0); // Schlick approx.
}
float distributionGGX(float NdotH, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float denom = (NdotH*NdotH)*(a2-1.0) + 1.0;
    return a2 / (3.14159265 * denom*denom); // normalized D
}
float geometrySmith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r*r) / 8.0; // UE4 style
    float gv = NdotV/(NdotV*(1.0-k)+k);
    float gl = NdotL/(NdotL*(1.0-k)+k);
    return gv*gl;
}
vec3 cookTorranceBRDF(vec3 N, vec3 V, vec3 L, vec3 F0, float roughness) {
    vec3 H = normalize(V+L);
    float NdotV = max(dot(N,V), 0.0);
    float NdotL = max(dot(N,L), 0.0);
    float NdotH = max(dot(N,H), 0.0);
    float VdotH = max(dot(V,H), 0.0);
    vec3 F = fresnelSchlick(VdotH, F0);
    float D = distributionGGX(NdotH, roughness);
    float G = geometrySmith(NdotV, NdotL, roughness);
    return (D * G * F) / max(4.0 * NdotV * NdotL, 1e-6); // specular RGB
}