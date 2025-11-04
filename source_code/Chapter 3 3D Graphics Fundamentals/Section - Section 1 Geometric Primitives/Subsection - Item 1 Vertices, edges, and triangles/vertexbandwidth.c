#include 
// inputs: total_triangles, unique_vertices, bytes_per_vertex, index_bytes
uint64_t estimate_indexed(uint64_t T, uint64_t Vu, uint32_t A, uint32_t I){
    return Vu*(uint64_t)A + T*(uint64_t)I; // bytes moved from memory
}
uint64_t estimate_nonindexed(uint64_t T, uint32_t A){
    return 3*T*(uint64_t)A; // bytes moved for non-indexed lists
}
// compute reuse factor R = 3T / Vu
double reuse_factor(uint64_t T, uint64_t Vu){
    return (3.0*(double)T) / (double)Vu;
}