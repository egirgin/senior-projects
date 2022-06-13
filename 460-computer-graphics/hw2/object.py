import numpy as np


class Object:

    def __init__(self, rgb, ambient_component=0.1, diffuse_component=0.9):
        self.rgb = rgb

        self.ambient_component = ambient_component

        self.diffuse_component = diffuse_component



class Sphere(Object):

    def __init__(self, center, radius, rgb, ambient_component=0.1, diffuse_component=0.9):
        super(Sphere, self).__init__(rgb, ambient_component, diffuse_component)
        
        self.x = center[0]
        self.y = center[1]
        self.z = center[2]

        self.center = center

        self.radius = radius


    def intersect(self, ray):
        
        B = ray.dir.dot(ray.source - self.center)

        four_AC = ray.dir.dot(ray.dir) * ((ray.source - self.center).dot(ray.source - self.center) - self.radius**2) 
        
        discriminant = B**2 - four_AC

        if discriminant < 0:
            return False, 0

        elif discriminant == 0:
            t = -1*ray.dir.dot(ray.source - self.center) / ray.dir.dot(ray.dir)

            return True, t

        elif discriminant > 0:
            # Substract the discriminant bcs we are looking for the closest intersection point and t is always positive in our setup
            t = (-1*ray.dir.dot(ray.source - self.center) - np.sqrt(discriminant) ) / ray.dir.dot(ray.dir)

            return True, t


class Plane(Object):
    def __init__(self, normal, point, rgb, ambient_component=0.1, diffuse_component=0.9):
        super(Plane, self).__init__(rgb, ambient_component, diffuse_component)

        self.normal = normal # 0, 1, 0

        self.point =  point


    
    def intersect(self, ray):
        denom = np.dot(self.normal, ray.dir)

        if denom > 1e-6:
            on_plane = self.point - ray.source

            t = np.dot(on_plane, self.normal) / denom

            return True, t

        return False, 0