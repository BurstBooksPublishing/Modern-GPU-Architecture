int dispatch_loop(int G, WorkGroup *queue, SMState *sms, int S) {
  // queue: work-group descriptors; sms: per-SM resource state
  while (atomic_fetch(&remaining) > 0) {
    int wg_id = atomic_dequeue(queue);                 // pop next work-group
    for (int sm = 0; sm < S; ++sm) {
      if (try_acquire_resources(&sms[sm], wg_id)) {    // attempt resource accounting
        assign_wg_to_sm(sm, wg_id);                    // handshake to SM
        break;
      }
    }
    // fall back: re-enqueue or stall until resources free (backoff)
  }
  return 0;
}