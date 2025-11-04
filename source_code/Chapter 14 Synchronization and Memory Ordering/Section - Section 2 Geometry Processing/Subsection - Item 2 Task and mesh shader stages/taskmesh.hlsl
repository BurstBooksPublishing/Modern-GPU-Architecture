StructuredBuffer meshletHeaders : register(t0); // read-only meshlet metadata
RWStructuredBuffer meshletList   : register(u0);     // append list of visible meshlet IDs
RWByteAddressBuffer indirectArgs       : register(u1);     // [0] = meshletCount (written by task)

// Task shader: coarse cull and append visible meshlet IDs
[shader("task")]
void TaskMain(uint3 tid : SV_DispatchThreadID)
{
    // load bounding, do frustum/occlusion/LOD test
    if (CoarseCull(meshletHeaders[tid.x])) return;
    uint idx = InterlockedAdd(indirectArgs, 0, 1);        // atomically increment visible count
    meshletList.Append(tid.x);                            // append meshlet ID for mesh shader
}

// Mesh shader: per-meshlet vertex fetch, per-primitive culling, and emit
[shader("mesh")]
void MeshMain(uint3 meshID : SV_GroupID)
{
    uint mID = meshletList[meshID.x];                     // get meshlet descriptor
    Meshlet m = LoadMeshlet(mID);                         // fetch indices/vertices
    // shared memory staging of vertex attributes
    GroupMemoryBarrierWithGroupSync();
    // per-thread vertex work then primitive cull and EmitMesh
    for (each primitive) EmitVertexAndTriangle(...);      // outputs to rasterizer
}