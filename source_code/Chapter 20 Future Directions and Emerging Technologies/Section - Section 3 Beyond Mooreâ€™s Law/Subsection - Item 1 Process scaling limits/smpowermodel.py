def max_sms(power_budget_w, per_sm_dyn, per_sm_leak):
    # per_sm_dyn: dynamic power at target utilization (W)
    # per_sm_leak: static leakage per SM (W)
    return int(power_budget_w // (per_sm_dyn + per_sm_leak))

# example: 350 W card, dynamic 0.6 W/SM, leakage 0.05 W/SM
print(max_sms(350, 0.6, 0.05))  # compute max SMs fitting thermal/power budget