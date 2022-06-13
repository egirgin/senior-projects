import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils import raster2screen
from object import Sphere, Plane
from ray import Ray

argparser = argparse.ArgumentParser()

argparser.add_argument("--config_file", default="config.txt")

args = argparser.parse_args()

width = 1000
height = 1000

scale = 50 # Screen is from -50 to +50

screen_position = 100

max_depth = 2

ambient_coeff = 0.1
diffuse_coeff = 1.0

light = np.array([500, 500, 300])

image = np.zeros(shape = (width, height, 3), dtype="float64")

rays = np.zeros(shape = (width, height), dtype=Ray)


def shading(ray, t, object, objects):
    color = object.rgb

    secondary_origin = ray.source + ray.dir*t

    secondary_dir = light - secondary_origin
    secondary_dir = secondary_dir/np.linalg.norm(secondary_dir)

    secondary_ray = Ray(secondary_origin, secondary_dir)

    secondary_t = 0

    ever_intersect = False

    
    try:
        # Diffuse (Lambertian) shading
        surface_normal = secondary_origin - object.center
    except Exception as e:
        surface_normal = -object.normal

    surface_normal = surface_normal/np.linalg.norm(surface_normal)

    color = color * max(0, surface_normal.dot(secondary_dir)) * object.diffuse_component# object diffuse coefficient
    
    # Ambiant shading
    
    for other_obj in objects: 

        secondary_intersect, secondary_t = other_obj.intersect(secondary_ray)

        if secondary_intersect and secondary_t > -0.1:
            # 0.1 is ambient shadowing coefficient
            color = color * object.ambient_component # object ambient coefficient
            ever_intersect = True
    

    # Diffuse Shading
    if not ever_intersect:
        pass
 
    
    return color
                    


def trace(ray, depth):
    if depth > max_depth:
        return 0, 1 if depth == 0 else depth
    
    closest_object = None
    closest_t = np.inf

    for object in objects:

        intersect, t_close = object.intersect(ray)

        if intersect and t_close < closest_t and t_close > -0.1:
            closest_t = t_close
            closest_object = object
            closest_intersection = ray.source + ray.dir * closest_t

    
    if closest_object == None:
        return 0, 1 if depth == 0 else depth
    else:
        
        color = shading(ray, closest_t, closest_object, objects)

        # Reflection

        reflected_ray_origin = closest_intersection

        if type(closest_object) == Sphere:
            # Diffuse (Lambertian) shading
            surface_normal = reflected_ray_origin - closest_object.center
        else:
            surface_normal = closest_object.normal

        surface_normal = surface_normal/np.linalg.norm(surface_normal)
        # R = D - 2(D . N)N -> ray.dir - 2(ray.dir * surface.normal)*surface.normal
        reflected_ray_dir = ray.dir - 2*(ray.dir.dot(surface_normal))*surface_normal

        reflected_ray_dir = reflected_ray_dir / np.linalg.norm(reflected_ray_dir)

        reflected_ray = Ray(source=reflected_ray_origin, dir=reflected_ray_dir)

        reflected_color, reflected_depth = trace(reflected_ray, depth+1)


        color = color + reflected_color
        

    depth = reflected_depth
    return color, depth






if __name__ == '__main__':

    # Create spheres and the plane is default
    objects = [
        Plane(
            normal=np.array([0, -1, 0]),
            point= np.array([0, -50, 100]),
            rgb = np.array([102, 153, 153]),
            ambient_component=ambient_coeff,
            diffuse_component=diffuse_coeff
        )
    ]

    with open(args.config_file, "r") as cfg:
        spheres =cfg.read().splitlines()

    for sphere in spheres:
        values = [int(x) for x in sphere.split()[:-2]]
        values += [float(x) for x in sphere.split()[-2:]]
        # If z of sphere is less than 200 or more than 1000 discard it.
        if values[2] < 200 or values[2] > 1000:
            continue
        objects.append(
            Sphere(
                center= np.array(values[:3]),
                rgb =  np.array(values[3:6]),
                radius = values[6],
                ambient_component=values[7],
                diffuse_component=values[8]
            )
        )



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

            color, depth = trace(ray=current_ray, depth=0)

            image[row_id][column_id] = np.clip(color/depth, 0.0, 255.0)

    plt.imsave("result.png", image/255.0)
    


