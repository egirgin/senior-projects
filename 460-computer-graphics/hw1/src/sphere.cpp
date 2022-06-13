#include<tuple> // for tuple
#include "ray.cpp"
#include <iostream>
#include<cmath>

using namespace std;

class Sphere{
    public:
        int x;
        int y;
        int z;
        Vec3 center;
        tuple <int, int, int> rgb;
        int radius;


        Sphere(Vec3 center, tuple <int, int, int> rgb, int radius){
            
            this->x = center.x;
            this->y = center.y;
            this->z = center.z;
            this->center = center;

            this->rgb = rgb;

            this->radius = radius;

        }

        bool intersect(Ray ray, float& tnear){
            
            float B = ray.dir.dot(ray.source - this->center);

            float AC = ray.dir.dot(ray.dir) * ((ray.source - this->center).dot(ray.source - this->center) - pow(this->radius, 2));

            float discriminant = pow(B, 2) - AC;

            if (discriminant < 0){
                return false;
            }
            else if (discriminant == 0){ // tangent line
                float t = float(-1)*ray.dir.dot(ray.source - this->center) / ray.dir.dot(ray.dir);
                tnear = t;
                return true;
            }
            
            else if (discriminant > 0){ // disc is positive
                float t = (float(-1)*ray.dir.dot(ray.source - this->center) - sqrt(discriminant)) / ray.dir.dot(ray.dir);
                tnear = t;
                return true;
            }
        }
};
