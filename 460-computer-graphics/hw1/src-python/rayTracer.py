import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import raster2screen
from object import Sphere, Plane
from ray import Ray

argparser = argparse.ArgumentParser()

argparser.add_argument("--config_file")

args = argparser.parse_args()

width = 1000
height = 1000

scale = 50 # Screen is from -50 to +50

screen_position = 100


if __name__ == '__main__':

    # Create spheres and the plane is default
    objects = [
        Plane(
            normal=np.array([0, -1, 0]),
            point= np.array([0, -50, 100]),
            rgb = np.array([102, 153, 153])
        )
    ]

    with open(args.config_file, "r") as cfg:
        spheres =cfg.read().splitlines()

    for sphere in spheres:
        values = [int(x) for x in sphere.split()]
        objects.append(
            Sphere(
                center= np.array(values[:3]),
                rgb =  np.array(values[3:6]),
                radius = values[-1]
            )
        )


    light = np.array([500, 500, 500])

    image = np.zeros(shape = (width, height, 3))

    rays = np.zeros(shape = (width, height), dtype=Ray)


    for row_id, row in enumerate(image):
        for column_id, column in enumerate(row):
            # Convert 0,1000 pixel coordinates to -50, +50 screen coordinates
            x_screen, y_screen = raster2screen(column_id, row_id) #columns are x axis rows are y-axis 

            # The eye is at 0,0,0
            cam_vector = np.array([0, 0, 0])

            # Direction of each ray
            dir = np.array([x_screen, y_screen, screen_position])

            current_ray = Ray(source = cam_vector, dir = dir/np.linalg.norm(dir)) # Normalize direction

            rays[row_id][column_id] = current_ray

            # t keeps track of the closest intersection point of the ray with any object
            t = np.inf

            for object in objects:

                t_old = t

                intersect, t_close = object.intersect(current_ray)

                if intersect:
                    t = t_close

                if intersect and t < t_old:

                    image[row_id][column_id] = object.rgb
                    
                    # Create shadow ray starting from the point on the object to the light source
                    secondary_origin = current_ray.source + current_ray.dir*t

                    secondary_dir = light - secondary_origin

                    secondary_ray = Ray(secondary_origin, secondary_dir/np.linalg.norm(secondary_dir))

                    secondary_t = 0

                    for other_obj in objects:

                        secondary_intersect, secondary_t = other_obj.intersect(secondary_ray)

                        if secondary_intersect and secondary_t > -0.1:
                            # 0.1 is ambient shadowing coefficient
                            image[row_id][column_id] *= 0.1
                    


    plt.imsave("result.png", image/255.0)
    


