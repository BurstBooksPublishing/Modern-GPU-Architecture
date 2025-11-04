# simple bandwidth calculator (bytes, pixels)
W, H, F = 3840, 2160, 60         # 4K@60
b_g = 32                         # G-buffer bytes/pixel (RGBA + normal+z)
R_read = 1                       # lighting pass reads G-buffer once
b_t = 64                         # avg texture bytes/pixel (cache ineff.)
T_avg = 2                        # TMU fetches per pixel
N_tile = 64*64                   # tile pixels
# DRAM BW (bytes/s)
B_dram = F * W * H * (b_g*(1+R_read) + b_t*T_avg)
S_tile = N_tile * b_g            # tile SRAM bytes
print(B_dram/(1<<30), "GiB/s DRAM BW")
print(S_tile/(1<<20), "MiB per tile SRAM")