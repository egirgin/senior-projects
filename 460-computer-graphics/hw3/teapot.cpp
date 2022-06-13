#include <stdio.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include "bezier.h"
#include "vector3d.h"
#include <algorithm>

using namespace std;

int width = 1000;
int height = 1000;

int scale = 50;


float screen_position = -100;

Vec3f light(300, 500, 0);

vector<vector<vector<int>>> image(
    width,
    vector<vector<int>>(
        height,
        vector<int>(3, 0)
    )

);


bool fast_intersection(
    const Vec3f &orig, const Vec3f &dir,
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
    float &t)
{
    float epsilon,u,v = 0.000001;
    Vec3f v0v1 = v1 - v0;
    Vec3f v0v2 = v2 - v0;
    Vec3f pvec = dir.crossProduct(v0v2);

    float det = v0v1.dotProduct(pvec);

    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < epsilon) return false;

    float invDet = 1.0 / det;

    Vec3f tvec = orig - v0;
    u = tvec.dotProduct(pvec) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    Vec3f qvec = tvec.crossProduct(v0v1);
    v = dir.dotProduct(qvec) * invDet;
    if (v < 0.0|| u + v > 1.0) return false;

    t = v0v2.dotProduct(qvec) * invDet;

    return true;
}


vector<float> screen_space(int row, int column){
    // Raster space to NDC space
    float new_row = (float(row) + 0.5) / width;
    float new_column = (float(column) + 0.5) / height;

    // NDC space to screen space
    new_row = 2 * new_row - 1;
    new_column = 1 - 2 * new_column;

    // Scale
    new_row = new_row * scale;
    new_column = new_column * scale;

    vector<float> output {new_row, new_column};

    return output;
}

Vec3f tris_points[TEAPOT_NB_PATCHES * RESU * RESV * 2][3];

Vec3f rotate_x(const Vec3f &v, float theta) {

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    Vec3f rotated(Vec3f(1, 0, 0).dotProduct(v),
                  Vec3f(0, cos_theta, -sin_theta).dotProduct(v),
                  Vec3f(0, sin_theta, cos_theta).dotProduct(v));

    return rotated;
}

Vec3f rotate_y(const Vec3f &v, float theta) {

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    Vec3f rotated(Vec3f(cos_theta, 0, sin_theta).dotProduct(v),
                  Vec3f(0, 1, 0).dotProduct(v),
                  Vec3f(-sin_theta, 0, cos_theta).dotProduct(v));

    return rotated;
}

Vec3f translate(const Vec3f &v, float x_, float y_, float z_) {

    Vec3f translated(v.x + x_, v.y + y_, v.z+z_);

    return translated;
}


void build_triangles(double theta){

    int triangle_id = 0;

    for (int patch_id = 0; patch_id < TEAPOT_NB_PATCHES; patch_id++) {

        for (int row_id = 0; row_id < RESU; row_id ++){
            for (int column_id = 0; column_id < RESV; column_id ++){
                triangle_id = patch_id * RESU * RESV + row_id * RESV + column_id ;


                struct vertex top_left_vertex = teapot_vertices[patch_id][row_id][column_id];

                Vec3f top_left(top_left_vertex.x, top_left_vertex.y, top_left_vertex.z);
                top_left = rotate_x(top_left, theta);
                //top_left = rotate_y(top_left, theta);
                top_left = top_left * 30;
                top_left = translate(top_left, 0, -50, -300);

                struct vertex bottom_left_vertex = teapot_vertices[patch_id][row_id+1][column_id];

                Vec3f bottom_left(bottom_left_vertex.x, bottom_left_vertex.y, bottom_left_vertex.z);
                bottom_left = rotate_x(bottom_left, theta);
                //bottom_left = rotate_y(bottom_left, theta);
                bottom_left = bottom_left * 30;
                bottom_left = translate(bottom_left, 0, -50, -300);

                struct vertex top_right_vertex = teapot_vertices[patch_id][row_id][column_id+1];

                Vec3f top_right(top_right_vertex.x, top_right_vertex.y, top_right_vertex.z);
                top_right = rotate_x(top_right, theta);
                //top_right = rotate_y(top_right, theta);
                top_right = top_right * 30;
                top_right = translate(top_right, 0, -50, -300);

                struct vertex bottom_right_vertex = teapot_vertices[patch_id][row_id+1][column_id+1];

                Vec3f bottom_right(bottom_right_vertex.x, bottom_right_vertex.y, bottom_right_vertex.z);
                bottom_right = rotate_x(bottom_right, theta);
                //bottom_right = rotate_y(bottom_right, theta);
                bottom_right = bottom_right * 30;
                bottom_right = translate(bottom_right, 0, -50, -300);

                
                tris_points[triangle_id*2][0] = top_left;
                tris_points[triangle_id*2][1] = bottom_right;
                tris_points[triangle_id*2][2] = top_right;
                     
                tris_points[triangle_id*2+1][0] = top_left;
                tris_points[triangle_id*2+1][1] = bottom_left;
                tris_points[triangle_id*2+1][2] = bottom_right;
                
            }
        }

    }
}

Vec3f shading(
    const Vec3f &orig, const Vec3f &dir, 
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, 
    const float &t, const int id){

    int total_tris = TEAPOT_NB_PATCHES * RESU*RESV*2; 

    Vec3f secondary_origin = orig + dir * t;
    
    Vec3f secondary_dir = light - secondary_origin;
    
    secondary_dir = secondary_dir.normalize();

    float secondary_t, u, v = 0;

    bool ever_intersect = false;

    Vec3f v0v1 = v1 - v0; 
    Vec3f v0v2 = v2 - v0; 
    Vec3f tri_normal = v0v1.crossProduct(v0v2); // N 
    tri_normal = tri_normal.normalize();
    
    Vec3f color(255,255,255);
    /*
    if (i%2==0){
        color = Vec3f(255,0,0);
    }
    else{
        color = Vec3f(0,0,255);
    }
    */

    // Diffuse
    color = color * std::max((float)0.0, tri_normal.dotProduct(-secondary_dir));
    
    // Ambient
    for(int i=0; i<total_tris;i++){
        Vec3f p0 = tris_points[i][0];
        Vec3f p1 = tris_points[i][1];
        Vec3f p2 = tris_points[i][2];
        float t_old = secondary_t;
        if (id == i){
            continue;
        }

        if (fast_intersection(secondary_origin, -secondary_dir, p0, p1, p2, secondary_t)){ 
            if (!ever_intersect && secondary_t < t_old ){
                color = color * 0;
                ever_intersect = true;
            }
        }

    }
    
    
    
    // Specular
    Vec3f V = -secondary_origin;
    V = V.normalize();
    Vec3f R = 2 * (tri_normal.dotProduct(secondary_dir)) * tri_normal - secondary_dir;
    //R = R.normalize();
    Vec3f specular = color * (pow( std::max( (float) 0.0, V.dotProduct(R) ), 250 ) );

    color += (specular * 0.8);
    

    return color;    
}


int main(){

    
    int total_tris = TEAPOT_NB_PATCHES * RESU*RESV*2; 

    build_teapot();

    build_triangles(-(3.14)/2);

    int count = 0;
    // Primary Rays
    for(int row=0; row < image.size(); row++){
        for(int column=0; column < image[0].size(); column++){

            count++;

            // Column represents X-axis
            // Row represents Y-axis

            vector<float> screen_coords = screen_space(column, row); // {X-Value, Y-Value}
            
            float x_screen = screen_coords[0];
            float y_screen = screen_coords[1];

            Vec3f cam_vector(0, 0, 0);
            Vec3f dir(x_screen, y_screen, screen_position);
            dir = dir.normalize();

            float t, u, v = 9999;

           bool ever_intersect = false;
            
            
            
            for(int i=0; i<total_tris;i++){
                Vec3f p0 = tris_points[i][0];
                Vec3f p1 = tris_points[i][1];
                Vec3f p2 = tris_points[i][2];
                float t_old = t;

                if (fast_intersection(cam_vector, dir, p0, p1, p2, t)){ 

                    if (!ever_intersect || t < t_old){
                        Vec3f color = shading(cam_vector, dir, p0, p1, p2, t, i);

                        image[row][column][0] = std::min(color.x, (float) 255.0);
                        image[row][column][1] = std::min(color.y, (float) 255.0);
                        image[row][column][2] = std::min(color.z, (float) 255.0);
                        ever_intersect = true;

                    }

                }
            }
            

            
            if (count % 10000 == 0)
                cout << 1.0 * count / 1000000 << endl;
            
            
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
