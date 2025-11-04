typedef struct { uint64_t shader_token; uint8_t root_data[]; } SBTRecord;
uint8_t *sbt_base;               // base GPU address (byte pointer)
size_t   record_size;            // R: aligned size per record
size_t   ray_type_stride;        // S: number of records per ray-type

// Compute pointer for ray type t and index i; // returns device pointer.
static inline SBTRecord* sbt_ptr(size_t t, size_t i) {
    size_t idx = t * ray_type_stride + i;          // compute logical index
    return (SBTRecord*)(sbt_base + idx * record_size); // pointer arithmetic
}