#include 
// issue PIM reduction: addr -> HBM tile, len elements, op=SUM
int pim_reduce(void *addr, size_t len) {
    // build command packet (DMA descriptor) -- driver maps to memory controller
    struct cmd { uint64_t base; uint32_t len; uint8_t op; } c = { (uint64_t)addr, (uint32_t)len, 1 };
    // submit to memory controller queue (kernel call); returns job id
    int job = submit_memcmd(&c); // blocking submit for simplicity
    // wait for completion and fetch result (small scalar read)
    int64_t result = fetch_result(job);
    return (int)result;
}
// Usage: offload large reductions to HBM-local PIM units.