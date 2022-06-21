import numpy as np
from PIL import Image, ImageColor

from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

width = 10
height = 10

NUM_RAYS = width*height
FOV =30 #in degrees 


def normalize(vec):
    if(np.linalg.norm(vec)==0):
        print("DIVIDING BY ZERO!!!")
    return vec / np.linalg.norm(vec)


def generate_rays(width, height):
    rays = np.zeros((width*height,  3))

    center = np.array([width / 2, height / 2])

    for x in range(0, width):
        for y in range(0, height):
           z = np.cos(np.radians(FOV/2))/np.sin(np.radians(FOV/2))
           rays[x+y*width] = normalize(np.array([(center[0]-x), center[1]-y, z]))
           # rays[x+y*width] = normalize(np.array([2*(center[0]-x)/width, 2*(center[1]-y)/height, FOV]))
    #for j in range(NUM_RAYS):
    #    x,y = (j%width), (j//width)
    #    rays[j] = normalize(np.array([center[0]-x,#2*(center[0] - x) / width,
    #        2*(center[1]-y) / height, FOV]))
    
    return rays


ray_directions = generate_rays(width, height)
#print(ray_directions[:,0])
ax.scatter(ray_directions[:, 0], ray_directions[:, 1], ray_directions[:, 2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



