import numpy as np
x = 0
y = 100
z = 100
phi = np.arctan2(y , x)  # y / x
theta = np.arccos(z/((x**2+y**2+z**2)**0.5))

print(phi)
print(theta)