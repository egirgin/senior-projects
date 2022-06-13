# -*- coding: utf-8 -*-
"""538-hw1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zIUCWqlYOvvNcBrrCqInZXkTTHR-_ce0
"""

import numpy as np
import matplotlib.pyplot as plt

[height, width] = [250, 250]
cube_size = 30

def form_rotation_matrix(roll, pitch, yaw, euler=False):

    if euler:
         [roll, pitch, yaw] = np.radians([roll, pitch, yaw])
    
    rotate_x = np.array([
                         [1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]
                         ])

    rotate_y = np.array([
                         [np.cos(pitch), 0, np.sin(pitch)],
                         [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]
                         ])
    
    rotate_z = np.array([
                         [np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]
                         ])

    return rotate_x @ rotate_y @ rotate_z


def form_calibration_matrix(rotation, translation):
    K = np.eye(3) # intrinsic matrix
    return K @ np.hstack((rotation, translation))

def project(calibration_matrix, points):
    """
        calibration matrix: 3x4
        points: Nx4 in world coordinates
    """

    points = points#/points[:, 3:4] # scale by homogeneous coordinates
    projected = (calibration_matrix @ (points.T)).T # apply transformation (T is to make Nx3)
    return projected/projected[:, 2:3] # scale by homogeneous coordinates

def generate_line(start, end, precision):
    """
        start and end coordinates of the line
        3d points
    """

    line_points = np.zeros((precision, 4))

    # find the axis to be filled
    flow_idx = np.flatnonzero(start-end)

    # max value of the axis
    size = np.abs((start-end)[flow_idx])

    # generate floating points
    flow = np.linspace(0, size, num=precision).reshape(precision, 1)

    # fill the variable axis
    line_points[:, flow_idx] = flow

    # fill the constant axices
    for i in range(3):
        if i != flow_idx:
            line_points[:, i] = start[i]

    # set homogeneous coordinates
    line_points[:, -1] = 1
    
    return line_points

def generate_corners(size=5):

    corners = np.array([
                     [0,0,0],
                     #[size,0,0],
                     [0,size,0],
                     [size,size,0],
                     [0,0,size],
                     [size,0,size],
                     [0,size,size],
                     [size,size,size],
    ])

    connections = np.array([
                            [0, size, size/3],
                            [0, size, size*2/3],
                            [0, 0, size/3],
                            [0, 0, size*2/3],

                            [0, size/3, size],
                            [0, size*2/3, size],
                            [0, size/3, 0],
                            [0, size*2/3, 0],

                            [size/3, size, size],
                            [size*2/3, size, size],
                            [size/3, size, 0],
                            [size*2/3, size, 0],

                            [size, size, size/3],
                            [size, size, size*2/3],
                            [0, size, size/3],
                            [0, size, size*2/3],

                            [size/3, size, size],
                            [size*2/3, size, size],
                            [size/3, 0, size],
                            [size*2/3, 0, size],

                            [size, size/3, size],
                            [size, size*2/3, size],
                            [0, size/3, size],
                            [0, size*2/3, size],

    ])

    return np.vstack((corners, connections))

def generate_cube_coordinates(size=5, precision=25):
    
    points = np.zeros((0, 4))
    
    corners = generate_corners(size)

    for start in corners:
        for end in corners:
            if np.count_nonzero(start-end) == 1:
                line_points = generate_line(start, end, precision=precision)
                points = np.vstack((points, line_points))

    return points

def render(points, img_height=500, img_width=500):
    """
    points: Nx2 in x, y format not row, column
    """

    #test = np.array([[0.4, -0.4],[-0.45, 0.45]])
    
    points = points.copy()
    
    #points = np.vstack((points, test))

    # move the origin from center to top left
    points[:, 0] += 0.5
    points[:, 1] -= 0.5
    points[:, 1] *= -1

    
    # scale by image size
    points[:, 0] *= img_width
    points[:, 1] *= img_height

    canvas = np.zeros((img_height, img_width))

    for p in points:
        # here we round the floating point
        # this may need a better solution
        if p[0] < img_width and p[1] < img_height:
            canvas[int(p[1]), int(p[0])] = 1
            p[0] = int(p[0])
            p[1] = int(p[1])


    return canvas, points

def ls_helper(xi, yi, x0, y0, z0):
    # only arranges the rows for correspondances
    return np.array(
        [
         [x0, y0, z0, 1, 0, 0, 0, 0, -xi*x0, -xi*y0, -xi*z0, -xi],
         [0, 0, 0, 0, x0, y0, z0, 1, -yi*x0, -yi*y0, -yi*z0, -yi]
        ]
    )

def form_ls_matrix(point_pairs):
    """
    form matrix to be applied SVD
    point_pairs: Nx2x3
    """
    Q = np.zeros((0, 12))

    for corr in point_pairs:
        img_coords = corr[0] # x, y, dummy
        world_coords = corr[1] # x, y, z

        pair = ls_helper(img_coords[0], img_coords[1], world_coords[0], world_coords[1], world_coords[2])
        Q = np.vstack((Q, pair))

    return Q

def approximate_calibration(Q_matrix):
    # apply svd
    U, S, V_t = np.linalg.svd(Q_matrix)

    # take the eigenvector corresponding to the smallest eigenvalue
    eigenvector = V_t[-1]

    return eigenvector.reshape(3,4)

def distance(real, approximated):
    # frobenius norm between two matrices
    return np.linalg.norm(real-approximated)

size = cube_size

world_points = np.array([
            [0,0,0], # 1
            #[size,0,0],
            [0,size,0], # 2
            [size,size,0], # 3
            [0,0,size], # 4
            [size,0,size], # 5
            [0,size,size], # 6
            [size,size,size], # 7

            [0, size, size/3], # 8
            [0, size, size*2/3], # 9
            [0, 0, size/3], # 10
            [0, 0, size*2/3], # 11

            [0, size/3, size], # 12
            [0, size*2/3, size], # 13
            [0, size/3, 0], # 14
            [0, size*2/3, 0], # 15

            [size/3, size, size], # 16
            [size*2/3, size, size], # 17
            [size/3, size, 0], # 18
            [size*2/3, size, 0], # 19

            [size, size, size/3], # 20
            [size, size, size*2/3], # 21

            [size/3, 0, size], # 22
            [size*2/3, 0, size], # 23

            [size, size/3, size], # 24
            [size, size*2/3, size], # 25

            # inside
            [0, size/3, size/3], # 26
            [0, size*2/3, size/3], # 27
            [0, size*2/3, size*2/3], # 28
            [0, size/3, size*2/3], # 29

            [size/3, size*2/3, size], # 30
            [size/3, size/3, size], # 31
            [size*2/3, size/3, size], # 32
            [size*2/3, size*2/3, size], # 33


            [size*2/3, size, size/3], # 34
            [size/3, size, size/3], # 35
            [size/3, size, size*2/3], # 36
            [size*2/3, size, size*2/3] # 37


    ])
#world_points[:, 2] -= 100
world_points = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

# generate camera geometry
r = form_rotation_matrix(45, 45, 0, euler=True)
t = np.array([[-15, -15, -100]])
c = form_calibration_matrix(r, t.T)

corners = project(c, world_points)
# generate cube
cube_points = generate_cube_coordinates(size=size, precision=100)
coords_2d = project(c, cube_points)


# render
corners_img, pixel_coordinates = render(corners[:,:-1], height, width)
img, _ = render(coords_2d[:,:-1], height, width)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

ax[0].imshow(img, cmap="gray")
ax[0].set_title("Rubic Cube")
ax[1].imshow(corners_img, cmap="gray")
ax[1].set_title("Selected Corners")
n_pixels = pixel_coordinates.shape[0]
plt.show()

# add dummy dimension to the pixel coordinates
pixel_coordinates = np.asarray(pixel_coordinates)
pixel_coordinates = np.hstack((pixel_coordinates, np.ones((n_pixels, 1)))) # Nx3

plt.clf()
plt.cla()
plt.close()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
for i, N in enumerate([6, 20, 30, 37]):
    # do least squares

    corr_matrix = np.zeros((N, 2, 3))

    corr_matrix[:, 0] = pixel_coordinates[:N]
    corr_matrix[:, 1] = world_points[:N, :-1]

    Q = form_ls_matrix(corr_matrix)

    c_hat = approximate_calibration(Q)


    # render

    cube_points = generate_cube_coordinates(size=size, precision=100)

    coords_2d = project(c_hat, cube_points)

    canvas = np.zeros((height, width))

    for p in coords_2d:
        # here we round the floating point
        # this may need a better solution
        if p[0] < height and p[1] < width:
            canvas[int(p[1]), int(p[0])] = 1

    ax[i//2, i%2].imshow(canvas, cmap="gray")
    ax[i//2, i%2].set_title("N:{} / D:{:.2f}".format(N, distance(img, canvas)))

    
plt.show()

