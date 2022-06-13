#include <stdio.h>
#include <iostream>
#include <vector>
#include "sphere.cpp"
#include <tuple>
#include <fstream>
  
using namespace std;

int width = 1000;
int height = 1000;

int scale = 50;

float screen_position = 100;


vector<float> screen_space(int row, int column){
    // Raster space to NDC space
    float new_row = (float(row) + 0.5) / width;
    float new_column = (float(column) + 0.5) / height;

    // NDC space to screen space
    new_row = 2 * new_row -1;
    new_column = 1- 2 * new_column;

    // Scale
    new_row = new_row * scale;
    new_column = new_column * scale;

    vector<float> output {new_row, new_column};

    return output;
}


int main ()
{

    Vec3 light_source(500, 100, 300);

    vector<vector<vector<int>>> image(
        width,
        vector<vector<int>>(
            height,
            vector<int>(3, 0)
        )

    );

    vector<vector<Ray>> rays(
        width,
        vector<Ray>(
            height
        )
    );

    vector<Sphere> spheres;

    spheres.push_back(
        Sphere(
            Vec3(0, 100, 300),
            tuple<int, int, int>(
                255,0,0
            ),
            35
        )
    );
    
    spheres.push_back(
        Sphere(
            Vec3(100, 100, 300),
            tuple<int, int, int>(
                0,255,0
            ),
            20
        )
    );
    
    int count = 0;


    // Primary Rays
    for(int row=0; row < image.size(); row++){
        for(int column=0; column < image[0].size(); column++){

            // Column represents X-axis
            // Row represents Y-axis

            vector<float> screen_coords = screen_space(column, row); // {X-Value, Y-Value}
            
            float x_screen = screen_coords[0];
            float y_screen = screen_coords[1];

            Vec3 origin(0, 0, 0);
            Vec3 dir(x_screen, y_screen, screen_position);

            rays[row][column] = Ray(origin, dir.normalize());
            float t = INFINITY;

            bool hit = false;

            for (Sphere s : spheres){
                //cout << rays[row][column].dir.x << endl;
                float t_old = t;

                if (s.intersect(rays[row][column], t)){
                    if (t < t_old){
                        image[row][column][0] = get<0>(s.rgb);
                        image[row][column][1] = get<1>(s.rgb);
                        image[row][column][2] = get<2>(s.rgb);

                        Ray current_ray = rays[row][column];

                        Vec3 secondary_origin = current_ray.source + current_ray.dir*t;

                        Vec3 secondary_dir = light_source - secondary_origin;

                        Ray secondary_ray(secondary_origin, secondary_dir.normalize());

                        float secondary_t = 0;


                        for (Sphere q : spheres){
                            if (q.intersect(secondary_ray, secondary_t)){
                                if (secondary_t > -0.1){
                                    image[row][column][0] = get<0>(s.rgb)*0.1;
                                    image[row][column][1] = get<1>(s.rgb)*0.1;
                                    image[row][column][2] = get<2>(s.rgb)*0.1;
                                }
                            }
                        
                        }
                    }



                }
            }

            
            
        }

    }



    ofstream out("test.pgm", ios::binary | ios::out | ios::trunc);

    out << "P6\n" << width << " " << height << "\n255\n";
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            out << (unsigned char)(image[x][y][0]) <<
                    (unsigned char)(image[x][y][1]) <<
                    (unsigned char)(image[x][y][2]);
        }
    }

    out.close();





}

/*

See 4.3.2 Perspective Views : Ray generation
4.4.1 : Ray-Sphere intersection

*/