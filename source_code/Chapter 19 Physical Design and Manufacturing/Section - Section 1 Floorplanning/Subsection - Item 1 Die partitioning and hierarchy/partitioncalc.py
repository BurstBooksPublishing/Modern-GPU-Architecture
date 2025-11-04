# compute number of tiles given die area and per-tile areas
die_area_mm2 = 600.0      # total die area in mm^2 (example)
sm_area = 20.0            # area per SM cluster in mm^2
l2_slice = 4.0            # area per L2 slice in mm^2
io_macros = 30.0          # total area reserved for I/O/PHYs
# allocate area for tiles (SM + L2)
tile_area = sm_area + l2_slice
available = die_area_mm2 - io_macros
num_tiles = int(available // tile_area)  # integer tiles that fit
print(f"Tiles: {num_tiles}, remaining_area_mm2: {available - num_tiles*tile_area}")
# further compute grid dims (closest square)
import math
cols = math.ceil(math.sqrt(num_tiles))
rows = math.ceil(num_tiles / cols)
print(f"Grid: {rows}x{cols} tiles")