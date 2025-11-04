# masks: list of 32-bit ints, one per warp execution at PC (1=taken,0=not)
from collections import Counter
masks = [...]  # collected by sampler; each entry is a 32-bit warp mask
cnt = Counter(masks)
total = sum(cnt.values())

# per-PC taken fraction p and expected serial paths S_exp
taken_bits = sum(bin(m).count("1") * c for m, c in cnt.items())
p = taken_bits / (total * 32.0)            # average per-thread taken prob
W = 32
S_exp = 2 - p**W - (1-p)**W                 # eqn (1) applied
# branch entropy per observation (binary)
import math
if p in (0.0,1.0):
    H = 0.0
else:
    H = - (p*math.log2(p) + (1-p)*math.log2(1-p))
print("p=",p,"S_exp=",S_exp,"H=",H)