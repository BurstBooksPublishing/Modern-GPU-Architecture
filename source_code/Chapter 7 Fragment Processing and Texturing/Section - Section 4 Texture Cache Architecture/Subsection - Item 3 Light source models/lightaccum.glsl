vec3 accumulateLights(vec3 pos, vec3 N, vec3 V) {
    vec3 result = vec3(0.0);
    // Directional: cheap, no attenuation (ALU only).
    vec3 Ld = normalize(dirLight.dir);
    float NdL = max(dot(N, Ld), 0.0);
    vec3 H = normalize(V + Ld);
    result += dirLight.intensity * (NdL + pow(max(dot(N,H),0.0), dirLight.shininess));
    // Point light: distance attenuation (ALU), no texture fetch.
    vec3 Lp = pointLight.pos - pos;
    float d = length(Lp);
    Lp /= d; // normalize (ALU cost)
    float att = 1.0 / (pointLight.kc + pointLight.kl * d + pointLight.kq * d*d);
    result += pointLight.intensity * att * max(dot(N,Lp),0.0);
    // Spot: angular falloff (ALU), extra branchless smoothstep.
    vec3 Ls = spotLight.pos - pos;
    float ds = length(Ls); Ls /= ds;
    float spotCos = dot(Ls, normalize(spotLight.dir));
    float spotFactor = smoothstep(spotLight.cosOuter, spotLight.cosInner, spotCos); // ALU
    float attS = 1.0 / (spotLight.kc + spotLight.kl * ds + spotLight.kq * ds*ds);
    result += spotLight.intensity * attS * spotFactor * max(dot(N,Ls),0.0);
    return result;
}