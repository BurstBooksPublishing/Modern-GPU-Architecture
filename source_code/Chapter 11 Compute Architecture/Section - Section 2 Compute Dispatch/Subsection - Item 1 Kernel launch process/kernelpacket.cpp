struct KernelPacket {             // serialized to submission queue
  uint64_t kernel_id;            // kernel code identifier
  uint32_t grid_x, grid_y, grid_z; // grid dims
  uint16_t block_x, block_y, block_z; // block dims
  uint32_t shmem_bytes;          // dynamic shared mem per block
  uint16_t regs_per_thread;      // ABI-reported reg usage
  uint8_t  priority;             // stream/priority hint
  uint64_t arg_buffer_addr;      // device pointer to args
};

void submit_kernel(KernelPacket pkt) {
  // push to ring buffer and notify command processor (doorbell)
  ring_push(pkt);                 // enqueue packet
  doorbell_ring();                // notify GPU
}