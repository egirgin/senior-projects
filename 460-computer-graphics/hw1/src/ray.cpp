#include<tuple> // for tuple
#include "vector3d.cpp"
using namespace std;

class Ray{
    public:

        float t_min;
        float t_max;

        Ray(){
            this->source = Vec3();
            this->source = Vec3();
        }
        

        Ray(const Vec3 &source, const Vec3 &dir){
            this->source = source;
            this->dir = dir;
        }

        Vec3 source;
        Vec3 dir;

};