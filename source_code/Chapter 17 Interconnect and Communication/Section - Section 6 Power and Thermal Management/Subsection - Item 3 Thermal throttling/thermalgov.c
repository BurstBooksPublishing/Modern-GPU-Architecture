#include 
int read_temp_millideg(void);             // read sensor (milli-deg C)
void apply_freq_cap(uint32_t mhz);       // enforce cap via registers
uint32_t current_cap = 2000;             // initial cap (MHz)
while (1) {
    int t = read_temp_millideg();        // e.g., 85000 == 85.0C
    if (t > 85000) {                     // throttle threshold (85 C)
        current_cap = current_cap > 200 ? current_cap - 200 : 200;
        apply_freq_cap(current_cap);     // reduce performance
    } else if (t < 80000) {              // hysteresis lower bound (80 C)
        current_cap = current_cap < 3000 ? current_cap + 100 : current_cap;
        apply_freq_cap(current_cap);     // relax cap
    }
    sleep_ms(50);                        // sampling cadence
}