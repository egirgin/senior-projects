#include "TeaPotRayTrace.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>


#define RESU 10
#define RESV 10


void build_control_points_k(int p, struct vertex control_points_k[][ORDER+1]);
struct vertex compute_position(struct vertex control_points_k[][ORDER+1], float u, float v);
float bernstein_polynomial(int i, int n, float u);
float binomial_coefficient(int i, int n);
int factorial(int n);

struct vertex teapot_vertices[TEAPOT_NB_PATCHES][RESU+1][RESV+1];

void build_teapot() {
  // Vertices
  for (int p = 0; p < TEAPOT_NB_PATCHES; p++) {
    struct vertex control_points_k[ORDER+1][ORDER+1];
    build_control_points_k(p, control_points_k);
    for (int ru = 0; ru <= RESU; ru++) {
      float u = 1.0 * ru / (RESU);
      for (int rv = 0; rv <= RESV; rv++) {
        float v = 1.0 * rv / (RESV);
        teapot_vertices[p][ru][rv] = compute_position(control_points_k, u, v);
      }
    }
  }

}

void build_control_points_k(int p, struct vertex control_points_k[][ORDER+1]) {
  for (int i = 0; i <= ORDER; i++)
    for (int j = 0; j <= ORDER; j++)
      control_points_k[i][j] = teapot_cp_vertices[teapot_patches[p][i][j] - 1];
}

struct vertex compute_position(struct vertex control_points_k[][ORDER+1], float u, float v) {
  struct vertex result = { 0.0, 0.0, 0.0 };
  for (int i = 0; i <= ORDER; i++) {
    float poly_i = bernstein_polynomial(i, ORDER, u);
    for (int j = 0; j <= ORDER; j++) {
      float poly_j = bernstein_polynomial(j, ORDER, v);
      result.x += poly_i * poly_j * control_points_k[i][j].x;
      result.y += poly_i * poly_j * control_points_k[i][j].y;
      result.z += poly_i * poly_j * control_points_k[i][j].z;
    }
  }
  return result;
}

float bernstein_polynomial(int i, int n, float u) {
  return binomial_coefficient(i, n) * pow(u, i) * pow(1-u, n-i);
}

float binomial_coefficient(int i, int n) {
    assert(i >= 0); assert(n >= 0);
    return 1.0f * factorial(n) / (factorial(i) * factorial(n-i));
}
int factorial(int n) {
    assert(n >= 0);
    int result = 1;
    for (int i = n; i > 1; i--)
        result *= i;
    return result;
}
