# Parse counters dict; keys are category names, values are counts.
def compute_mix(counters): 
    total = sum(counters.values())
    return {k: v/total for k, v in counters.items()}  # normalized mix

# Example hardware target mix for balanced SM usage (engineer-tuned).
target = {'FP':0.35,'INT':0.15,'MEM':0.20,'SFU':0.05,'TMU':0.10,'MMA':0.15}

def l1_imbalance(mix, target):
    return sum(abs(mix.get(k,0)-target.get(k,0)) for k in target)

# Example usage
counters = {'FP':1_200_000,'INT':300_000,'MEM':700_000,'SFU':100_000,'TMU':200_000,'MMA':500_000}
mix = compute_mix(counters)          # normalized instruction mix
imbalance = l1_imbalance(mix, target)  # scalar metric to prioritize tuning