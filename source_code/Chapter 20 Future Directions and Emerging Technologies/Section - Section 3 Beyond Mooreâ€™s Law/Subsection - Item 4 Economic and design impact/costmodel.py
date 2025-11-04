import math

# wafer and defect model params
C_wafer = 10000.0   # $ per wafer
R = 150.0           # wafer radius mm
D = 0.5e-6          # defects per mm^2

def cost_per_good_die(area_mm2, C_pack):
    N_die = math.pi * R*R / area_mm2                       # approximate dies per wafer
    Y = math.exp(-area_mm2 * D)                            # Poisson yield
    return C_wafer / (N_die * Y) + C_pack                  # $ per good die

# monolithic GPU
mono_area = 800.0     # mm^2
mono_pack = 50.0
print(cost_per_good_die(mono_area, mono_pack))            # cost monolithic

# chiplet GPU: smaller compute die plus IO die on interposer
chiplet_area = 200.0  # mm^2 per compute chiplet
num_chiplets = 4
io_die_area = 100.0
interposer_pack = 200.0  # includes advanced packaging cost
# cost includes all chiplets plus IO die and higher packaging
total_chiplet_cost = num_chiplets * cost_per_good_die(chiplet_area, 20.0) \
                     + cost_per_good_die(io_die_area, 20.0) + interposer_pack
print(total_chiplet_cost)  # total assembled GPU cost estimate