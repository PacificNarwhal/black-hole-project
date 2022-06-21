import numpy as np
from PIL import Image, ImageColor

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

width = 10
height = 10

NUM_RAYS = width*height
FOV = 3


def normalize(vec):
    return vec / np.linalg.norm(vec)


def generate_rays(width, height):
    rays = np.zeros((width*height,  3))

    center = np.array([width / 2, height / 2])

#    for x in range(0, width):
#        for y in range(0, height):
#            rays[x+y*width] = normalize(np.array([2*(center[0]-x)/width, 2*(center[1]-y)/height, fov]))
    for j in range(NUM_RAYS):
        x,y = (j%width), (j//width)
        rays[j] = normalize(np.array([2*(center[0] - x) / width, 2*(center[1]-y) / height, FOV]))
    return rays


ray_directions = generate_rays(width, height, 10)
ax.scatter(ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

SKYBOX_DISTANCE = 10
SKYBOX = Image.open("nebula1.png")
SKYBOX_WIDTH, SKYBOX_HEIGHT = SKYBOX.size


def skybox_color_at(position):
    # TODO
    return SKYBOX.getpixel((0, 0))


# Euler chromer for every ray.
# The dummy function should apply the curved space
def dummy_helper(positions, ray_directions):
    return np.zeros_like(ray_directions)


# c = 1 system
MAX_STEPS = 1000
c = 2
dt = 0.5
ray_positions = np.zeros((height * width, 3))

img = Image.new('RGB', (width, height), "white")
finished_rays = np.full(width * height, False)

for i in range(MAX_STEPS):
    ray_directions += dummy_helper(ray_positions, ray_directions) * dt
    ray_positions += c * ray_directions * dt
    for j in range(height * width):
        if finished_rays[j] == True:
            continue
        ray = ray_positions[j]
        #r2 = np.dot(ray, ray)
        if ray[2] >= SKYBOX_DISTANCE:
            img.putpixel((j % width, j // width),
                         skybox_color_at(ray))
            finished_rays[j] = True
            print(ray[2])
    if np.all(finished_rays):
        break

print(np.count_nonzero(finished_rays))

img.save("test2.png")
