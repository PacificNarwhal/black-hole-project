import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

x = []
y = []
z = []

for i in range(10):
    x.append(i)
    y.append(i*i)
    z.append(i)

ax.scatter(x,y,z)
plt.show()

