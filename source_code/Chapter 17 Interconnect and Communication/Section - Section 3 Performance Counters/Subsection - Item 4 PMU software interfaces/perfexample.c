#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

// open perf-like PMU (vendor encoding done in driver) and read counter.
int open_pmu(int type, int config) {
  struct perf_event_attr attr = {0};
  attr.type = type; attr.size = sizeof(attr);
  attr.config = config; // vendor event id
  attr.disabled = 1; attr.exclude_kernel = 1;
  return syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
}
int main() {
  int fd = open_pmu(/*PMU_TYPE*/0x4, /*EVENT_ID*/0x123);
  // enable, run workload, then read 64-bit counter
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
  // run GPU kernel (submit commands)...
  uint64_t val; read(fd, &val, sizeof(val)); // raw count
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
  printf("raw count = %llu\n", (unsigned long long)val);
  close(fd);
}