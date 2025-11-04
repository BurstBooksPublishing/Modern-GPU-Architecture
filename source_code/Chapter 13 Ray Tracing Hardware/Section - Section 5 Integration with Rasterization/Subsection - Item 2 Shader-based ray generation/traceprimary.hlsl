struct Payload { float3 color; uint hit; };                    // small payload
RayPayloadPayloadType TraceHitShader();                          // hit shader binding
[shader("pixel")]
float4 PSMain(float2 uv : TEXCOORD0) : SV_Target {
    // compute NDC and unproject to world space
    float2 ndc = uv * 2.0f - float2(1.0f,1.0f);                 // map to [-1,1]
    float3 origin = cameraPos;                                  // camera origin
    float4 dirH = mul(invViewProj, float4(ndc, 1.0f, 0.0f));    // eq. (1)
    float3 dir = normalize(dirH.xyz);
    Payload p; p.color = float3(0,0,0); p.hit = 0;              // init payload
    TraceRay(RTScene, /*flags*/ 0, /*instance*/ 0, /*rayId*/ 0,
             origin, 0.0f, dir, 1e32f, /*sbtIndex*/ 0, p);
    // Compose color from payload returned by hit/miss shaders
    return float4(p.color, 1.0f);
}