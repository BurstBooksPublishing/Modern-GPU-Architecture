#include <stdatomic.h>
// shared: data[], flag
// Producer: publish data then set flag (release)
void producer(int *data, atomic_int *flag) {
  data[0] = 42;                      // write data
  atomic_thread_fence(memory_order_release); // ensure prior stores ordered
  atomic_store_explicit(flag, 1, memory_order_release); // publish
}
// Consumer: wait for flag (acquire) then read data
void consumer(int *data, atomic_int *flag) {
  while (atomic_load_explicit(flag, memory_order_acquire) != 1) { /* spin */ }
  atomic_thread_fence(memory_order_acquire); // ensure subsequent loads see data
  int v = data[0];                     // safe read
  (void)v;
}