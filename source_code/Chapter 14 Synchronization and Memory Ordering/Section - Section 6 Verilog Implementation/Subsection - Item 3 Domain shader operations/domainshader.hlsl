struct HS_Output { float3 pos : POSITION; float3 nrm : NORMAL; float2 uv : TEXCOORD0; }; 
struct DS_Input { float3 pos : POSITION; float3 nrm : NORMAL; float2 uv : TEXCOORD0; }; 

PatchConstantBuffer { HS_Output cp[3]; } pcb; // control points passed from hull shader

[domain("tri")]                                 // triangle domain shader
void DSMain(const OutputPatch patch, float3 bary : SV_DomainLocation,
            out float4 oPos : SV_Position, out float3 oNrm : NORMAL, out float2 oUV : TEXCOORD0)
{
    // interpolate position and attributes using barycentric weights
    float3 P = bary.x*patch[0].pos + bary.y*patch[1].pos + bary.z*patch[2].pos; // position
    float3 N = normalize(bary.x*patch[0].nrm + bary.y*patch[1].nrm + bary.z*patch[2].nrm); // normal
    float2 UV = bary.x*patch[0].uv + bary.y*patch[1].uv + bary.z*patch[2].uv; // uv

    // displacement sample (TMU fetch) and apply along interpolated normal
    float d = displacementSampler.SampleLevel(displacementSamplerSamplerState, UV, 0); // explicit LOD
    P += d * N;

    // emit clip-space position (assume viewProj matrix is available)
    oPos = mul(float4(P,1.0), viewProj); 
    oNrm = N;
    oUV = UV;
}