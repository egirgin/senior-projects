
#include <cmath>
  
using namespace std;

class Vec3{
    public:
        float x, y, z;
        float len = 0;

        Vec3(){
            this->x = 0;
            this->y = 0;
            this->z = 0;
        }

        Vec3(float x, float y, float z){
            this->x = x;
            this->y = y;
            this->z = z;
        }


        float length(){
            return sqrt(x*x + y*y + z*z);
        }

        Vec3 normalize(){
            float length = this->length();

            if(length > 0){
                Vec3 norm(
                    this->x/length,
                    this->y/length,
                    this->z/length
                );
                return norm;
            }

            
        }

        float dot(const Vec3& other){
            return this->x * other.x + this->y * other.y + this->z * other.z;
        }

        Vec3 operator+ (const Vec3& other ){
            Vec3 sum(
                this->x + other.x,
                this->y + other.y,
                this->z + other.z
            );
            return sum;
        }

        Vec3 operator- (const Vec3& other ){
            Vec3 sum(
                this->x - other.x,
                this->y - other.y,
                this->z - other.z
            );
            return sum;
        }

        Vec3 operator* (const float other ){
            Vec3 sum(
                this->x * other,
                this->y * other,
                this->z * other
            );
            return sum;
        }
        /*
        friend std::ostream& operator<<(std::ostream& os, const Vec3& vec)
        {
            return os << "X: " << vec.x << "Y: " << vec.y << "Z: " << vec.z << endl;
        }
        */
        
};