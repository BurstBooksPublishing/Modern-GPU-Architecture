struct desc { uint64_t cmd_buf_addr; uint32_t length; uint32_t flags; };

void submit_to_queue(volatile struct desc *ring, unsigned ring_size,
                     unsigned *head, struct desc d, volatile uint32_t *doorbell_mmio) {
  unsigned idx = __atomic_fetch_add(head, 1, __ATOMIC_RELAXED) % ring_size;
  ring[idx] = d;                         // publish descriptor
  __atomic_thread_fence(__ATOMIC_RELEASE); // ensure descriptor visible
  *doorbell_mmio = idx;                  // ring doorbell (MMIO write)
  /* optional: check MSI-X or completion later */
}