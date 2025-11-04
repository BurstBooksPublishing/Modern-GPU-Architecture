# baseline parameters (example values)
baseline_trans_per_mm2 = 1e9    # transistors per mm^2 at reference node
area_per_SM_mm2 = 100.0         # mm^2 per SM (including local cache)
density_gain = 1.8              # e.g., CFET ~1.8x effective density
C_per_trans = 2e-15             # Farads per transistor (typical order)
f_core = 1.5e9                  # core frequency (Hz)
V_dd = 0.8                      # supply voltage (V)
leakage_per_trans = 1e-9        # static leakage (A)
# derived
new_trans_per_mm2 = baseline_trans_per_mm2 * density_gain
SMs_per_die = int((new_trans_per_mm2 * 1e-6) / (area_per_SM_mm2 * baseline_trans_per_mm2/1e6))
P_dyn_per_trans = C_per_trans * V_dd**2 * f_core     # dynamic per-transistor
P_static_per_trans = leakage_per_trans * V_dd
# totals
total_trans = new_trans_per_mm2 * 100.0               # example die area 100 mm^2
P_total = total_trans * (P_dyn_per_trans + P_static_per_trans)
print("Projected SMs per die:", SMs_per_die)           # rough capacity estimate
print("Projected total power (W):", P_total)