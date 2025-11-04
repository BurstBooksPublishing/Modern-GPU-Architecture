/* sample counters and sensors */
uint32_t sm_util = read_hw_counter(SM_UTIL);      // percent
uint32_t gflops = read_hw_counter(GFLOPS);        // rolling GFLOPS
float temp = read_sensor(TEMP_SENSOR);            // degrees C

/* compute estimates */
float perf = (float)gflops;                       // measured throughput
float power = read_power_meter();                 // instantaneous W
float eta = perf / max(power, 1e-3f);             // GFLOPS/W

/* PI controller tries to increase eta by nudging freq or cap */
static float integrator = 0;
float error = target_eta - eta;
integrator += ki * error;
float control = kp * error + integrator;

/* map control output to discrete DVFS steps or cap */
int freq_step = clamp(curr_step + (int)roundf(control), MIN_STEP, MAX_STEP);
if (temp > MAX_SAFE_TEMP) freq_step = min(freq_step, THROTTLE_STEP); // thermal safety

apply_freq_step(freq_step);  // fast switch supported by firmware