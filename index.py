import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageColor

def normalize(vec):
    return vec / np.linalg.norm(vec, axis=1)[:, None]


WIDTH, HEIGHT = 100, 100

NUM_RAYS = WIDTH * HEIGHT
FOV = 90  # in degrees
ORIGIN = np.array([0., 0., 0.]) # Camera Position
ROTATION = Rotation.from_euler("xyz", [0, 3, 0], degrees=True) # Camera rotation
BLACK_HOLE_POSITION = np.array([0, -0.5, 40])
SCHWARZSCHILD_RADIUS = 5


def generate_rays():
    j = np.arange(0, NUM_RAYS)
    x, y = (j % WIDTH), (j // WIDTH)
    scaling = 2 / np.max([WIDTH, HEIGHT])
    vx = scaling * (WIDTH / 2 - x)
    vy = scaling * (HEIGHT / 2 - y)
    vz = np.cos(np.radians(FOV / 2)) / np.sin(np.radians(FOV / 2))
    rays = np.column_stack((vx, vy, np.full((NUM_RAYS), vz)))
    return normalize(rays)

SKYBOX = Image.open("backgrounds/simplegrid.jpeg").convert("RGB")
SKYBOX_WIDTH, SKYBOX_HEIGHT = SKYBOX.size
SKYBOX_DATA = np.array(SKYBOX) 

def map_rays_to_skybox(direction, position):
    r = np.linalg.norm(direction[:], axis=1)
    #phi = np.arctan(direction[:, 0] / direction[:, 2])
    phi = np.arctan2(direction[:,0], direction[:, 2])
    theta = np.arctan(direction[:, 1] / r)
    #skybox_x = (SKYBOX_WIDTH / 2) * (1 + (phi / np.pi))
    print(np.max(np.degrees(phi)), np.min(np.degrees(phi)))
    skybox_x = (SKYBOX_WIDTH / 2) * (phi-0)*np.cos(0) 
    #skybox_y = (SKYBOX_HEIGHT / 2) * (1 + (theta / np.pi))
    skybox_y = (SKYBOX_HEIGHT / 2) * (theta-0) 
    return np.column_stack((skybox_x, skybox_y)).astype(int)

def blackhole_forcefield(positions):
    h_squared = SCHWARZSCHILD_RADIUS**2
    factor = -(1.5) * h_squared
    dr = positions - BLACK_HOLE_POSITION
    r2 = np.sum(dr**2, axis=1)[:, None]
    return factor * dr / np.power(r2, 3)


ray_velocities = generate_rays()
ray_velocities = ROTATION.apply(ray_velocities)
ray_positions = np.tile(ORIGIN, (NUM_RAYS, 1))

# c = 1 system
FULL_TIME = 150
STEPS = 50
c = 1
dt = FULL_TIME / STEPS

for i in range(STEPS):
    # Euler Chromer
    acc = blackhole_forcefield(ray_positions)
    ray_velocities += acc * dt

    # Midpoint
    #k1 = dt * blackhole_forcefield(ray_positions)
    #k2 = dt * blackhole_forcefield(ray_positions + 0.5 * k1)
    #ray_velocities += k2

    # Runge Kutta 4
    #k1 = dt * blackhole_forcefield(ray_positions)
    #k2 = dt * blackhole_forcefield(ray_positions + 0.5 * k1)
    #k3 = dt * blackhole_forcefield(ray_positions + 0.5 * k2)
    #k4 = dt * blackhole_forcefield(ray_positions + k3)
    #ray_velocities += 1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4

    ray_velocities = normalize(ray_velocities)
    ray_positions += ray_velocities * dt

    if i % 10 == 0:
        print("Iteration", i)
        dr = ray_positions-BLACK_HOLE_POSITION
        close_to_blackhole_mask = np.sum(dr**2, axis=1) < SCHWARZSCHILD_RADIUS#*100
        skybox_indices = map_rays_to_skybox(ray_velocities, ray_positions)
        ray_colors = SKYBOX_DATA[skybox_indices[:, 1], skybox_indices[:, 0]]
        ray_colors[close_to_blackhole_mask] = np.array([0, 0, 0])
        img = ray_colors.reshape((HEIGHT, WIDTH, 3))
        img = Image.fromarray(np.uint8(img))
        img.save("imgn"+str(i)+".png")
        

#def generate_image():
#    dr = ray_positions-BLACK_HOLE_POSITION
#    close_to_blackhole_mask = np.sum(dr**2, axis=1) < SCHWARZSCHILD_RADIUS*100
#    skybox_indices = map_rays_to_skybox(ray_velocities, ray_positions)
#    ray_colors = SKYBOX_DATA[skybox_indices[:, 1], skybox_indices[:, 0]]
#    ray_colors[close_to_blackhole_mask] = np.array([0, 0, 0])
#    return ray_colors.reshape((HEIGHT, WIDTH, 3))
#
#
#img = generate_image()
#img = Image.fromarray(np.uint8(img))
#img.save("imgn"+str(i)+".png")
