# compute deltaT across TIM and total interface
def deltaT_tim(P, A, k, t, R_contact=0.0):
    R_tim = t / (k * A)            # conduction resistance
    R_total = R_contact + R_tim    # ignore spreader term here
    return P * R_total

# example: 200 W localized over 1e-4 m^2 (10x10 mm hotspot)
P = 200.0
A = 1e-4
print(deltaT_tim(P, A, k=5.0, t=50e-6))   # advanced grease, 50um
print(deltaT_tim(P, A, k=1.0, t=200e-6))  # gap filler, 200um