from matplotlib import pyplot as plt
import numpy as np
#from scipy.interpolate import interp2d
import os



debug = True
normalize = True
noise = False
all_imgs = True

def formCorrespondenceMatrix(point1, point2):
    return np.array([
        [point1[0], point1[1], 1, 0, 0, 0, -point1[0]*point2[0], -point1[1]*point2[0], -point2[0]],
        [0, 0, 0, point1[0], point1[1], 1, -point1[0]*point2[1], -point1[1]*point2[1], -point2[1]]
    ])

def computeH(imPoints1, imPoints2):

    point_number = imPoints1.shape[1]

    # form data matrix
    data_matrix = np.zeros((1,9), dtype="float32") # dummy row

    for i in range(point_number):
        corr_matrix = formCorrespondenceMatrix(imPoints1[:, i], imPoints2[:, i])

        data_matrix = np.vstack((data_matrix, corr_matrix))


    data_matrix = data_matrix[1:] # get rid of the dummy row

    # compute eigenvector which minimizes the A where A*h_hat = 0
    _, s, v_T = np.linalg.svd(data_matrix)

    h_hat = v_T[s.argmin()] # minimize least square

    return h_hat.reshape(3, 3)

def normalize_canvas(shape, points):
    height, width = shape
    new_points = points.copy().astype("float64")

    # make image size 1x1
    new_points[0, :] = new_points[0, :] / width
    new_points[1, :] = new_points[1, :] / -height

    # move the origin to the center
    new_points[0, :] = new_points[0, :] - 0.5
    new_points[1, :] = new_points[1, :] + 0.5

    return new_points

def back2image(shape, points):
    height, width = shape
    new_points = points.copy().astype("float64")

    # move the origin back to the top left
    new_points[0, :] = new_points[0, :] + 0.5
    new_points[1, :] = new_points[1, :] - 0.5

    # turn back to original size
    new_points[0, :] = new_points[0, :] * width
    new_points[1, :] = new_points[1, :] * -height

    return new_points

def partial_interp_2d(patch):

    # its just an averaging filter.
    return patch.mean(axis=(0, 1))
    """
    # the rest is unused interpolation part, idk why it doesnt working
    [n_rows, n_cols] = patch.shape[:-1]

    center_row = int(n_rows/2)
    center_col = int(n_cols / 2)

    if patch.mean(axis=(0, 1)).all() < 1/255:
        return [1, 0, 0]

    r_interp = interp2d(x=np.arange(0, n_cols, 1), y=np.arange(0, n_rows, 1), z=patch[:, :, 0], kind="cubic")
    g_interp = interp2d(x=np.arange(0, n_cols, 1), y=np.arange(0, n_rows, 1), z=patch[:, :, 1], kind="cubic")
    b_interp = interp2d(x=np.arange(0, n_cols, 1), y=np.arange(0, n_rows, 1), z=patch[:, :, 2], kind="cubic")

    rgb_value = np.array([r_interp(center_col, center_row), g_interp(center_col, center_row), b_interp(center_col, center_row)])

    return rgb_value[:, 0]
    """


def warp_pts(image, H):

    img_flatten = image.reshape(-1, 3)

    # create coordinate mesh for a given image
    if normalize:
        xx, yy = np.meshgrid(np.arange(-0.5, 0.5, 1 / image.shape[1]), np.arange(0.5, -0.5, -1 / image.shape[0]))
    else:
        xx, yy = np.meshgrid(np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1))
    matrix_coords = np.stack((xx, yy), axis=2)  # from [-0.5, 0.5] to [0.5, -0.5]

    flatten_coords = matrix_coords.reshape(-1, 2).T

    # add homogeneous coordinate
    coords_homo = np.vstack((flatten_coords, np.ones(flatten_coords.shape[1])))

    # warp the coordinates
    new_coords = H.dot(coords_homo)
    new_coords /= new_coords[-1]

    #return back from normalization
    if normalize:
        image_frame = back2image((image.shape[0], image.shape[1]), new_coords[:-1, :])
    else:
        image_frame = new_coords[:-1, :]

    #fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

    # shift the image so that the upper left corner fits on the (0,0)
    min_x = image_frame[0].min()
    min_y = image_frame[1].min()

    shifted_image = image_frame - [[min_x], [min_y]]

    wrap_canvas = np.zeros((round(shifted_image[1].max()) + 1, round(shifted_image[0].max()) + 1, 3))

    for pt_id, pts in enumerate(shifted_image.T):

        wrap_canvas[round(pts[1]), round(pts[0])] = img_flatten[pt_id] / 255.0

    #ax0.imshow(wrap_canvas)
    #print("Interpolating black pixels...")
    interpolation_range = 9
    # if a value of a pixel is zero then interpolate it.

    for row_id in range(interpolation_range, wrap_canvas.shape[0]-interpolation_range):
        for column_id in range(interpolation_range, wrap_canvas.shape[1]-interpolation_range):
            if wrap_canvas[row_id, column_id].all() == 0:
                new_rgb = partial_interp_2d(wrap_canvas[row_id - interpolation_range: row_id + interpolation_range + 1,
                                  column_id - interpolation_range: column_id + interpolation_range + 1])

                wrap_canvas[row_id, column_id] = new_rgb

    #ax1.imshow(wrap_canvas)
    #plt.show()

    return image_frame, wrap_canvas, [min_x, min_y]


def blend(wrapped_imgs, shifts_horizontal, shifts_vertical):

    base_id = int((len(wrapped_imgs) - 1) / 2)

    # calculate how much do i have to shift each image
    shifts_horizontal_abs = np.abs(shifts_horizontal)
    cumsum_shifts_horizontal = np.cumsum(shifts_horizontal_abs, axis=0)

    shifts_vertical_abs = np.abs(shifts_vertical)

    canvas_height = 0
    for im_id, img in enumerate(wrapped_imgs):
        img_vertical_cost = img.shape[0] + shifts_vertical_abs[im_id]
        if img_vertical_cost > canvas_height:
            canvas_height = img_vertical_cost

    rightmost_img_shape = wrapped_imgs[-1].shape

    new_blended = np.zeros((int(canvas_height) + 1, int(rightmost_img_shape[1] + shifts_horizontal_abs[1] + shifts_horizontal_abs[-1]) + 1, 3))

    # put warped images on the final canvas except the base img
    if all_imgs:
        traverse_list = [0, 1, 2, 4, 3]
        h_shifts = [0, shifts_horizontal_abs[1] - shifts_horizontal_abs[2], shifts_horizontal_abs[1], shifts_horizontal_abs[1] + shifts_horizontal_abs[3], shifts_horizontal_abs[1] + shifts_horizontal_abs[4] ]
    else:
        traverse_list = [0, 1, 2]
        h_shifts = cumsum_shifts_horizontal


    for im_id in traverse_list:
        img = wrapped_imgs[im_id]
        if im_id == base_id:
            continue
        for row_id in range(img.shape[0]):
            for col_id in range(img.shape[1]):
                if img[row_id, col_id].all() > 0:
                    new_blended[int(row_id + (shifts_vertical[im_id] - shifts_vertical.min())), int(col_id + h_shifts[im_id])] = img[row_id, col_id]

    # put base img at the end
    base_img = wrapped_imgs[base_id]

    for row_id in range(base_img.shape[0]):
        for col_id in range(base_img.shape[1]):
            new_blended[int(row_id + (shifts_vertical[base_id] - shifts_vertical.min())), int(col_id + h_shifts[base_id])] = base_img[row_id, col_id]

    return new_blended

def experiment(folder_name, n_points, exp_folder):
    noise_std = 1
    print(exp_folder)
    print("Experiment Config: Normalize:{}, Noise:{}, Panorama:{}, #Points:{}".format(normalize, noise, all_imgs, n_points))

    # read imgs
    im_names = ["left-1.jpg", "middle.jpg", "right-1.jpg"]  # ["left-2.jpg", "left-1.jpg", "middle.jpg", "right-1.jpg", "right-2.jpg"]

    if all_imgs:
        im_names = ["left-2.jpg", "left-1.jpg", "middle.jpg", "right-1.jpg", "right-2.jpg"]

    imgs = []
    for im in im_names:
        imgs.append(plt.imread("{}/{}".format(folder_name, im)))

    # read pts

    if all_imgs:
        # 0, 1, 1, 2, 3, 2, 3, 4
        point_names = ["l22l1", "l12l2", "l12m", "m2l1", "r12m", "m2r1", "r22r1", "r12r2"]
        related_im_ids = [0, 1, 1, 2, 3, 2, 4, 3]
    else:
        point_names = ["l12m", "m2l1", "r12m", "m2r1"]
        related_im_ids = [0, 1, 2, 1]

    # read points
    all_pts = []

    for homogprahpy_id, name in enumerate(point_names):
        img_size = (imgs[related_im_ids[homogprahpy_id]].shape[0], imgs[related_im_ids[homogprahpy_id]].shape[1])
        if debug and "{}.npy".format(name) in os.listdir("./{}".format(exp_folder)):
            with open("{}/{}.npy".format(exp_folder, name), 'rb') as file_pointer:
                points = np.load(file_pointer)
                if noise:
                    points += np.random.normal(0, noise_std, points.shape)
                if normalize:
                    all_pts.append(normalize_canvas(shape=img_size, points=points))
                else:
                    all_pts.append(points)
        else:
            plt.imshow(imgs[related_im_ids[homogprahpy_id]])
            points_temp = np.asarray(plt.ginput(n_points, show_clicks=True)).T
            with open("{}/{}.npy".format(exp_folder, name), 'wb+') as file_pointer:
                np.save(file_pointer, points_temp)
            points = points_temp
            if noise:
                points += np.random.normal(0, noise_std, points.shape)
            if normalize:
                all_pts.append(normalize_canvas(shape=img_size, points=points))
            else:
                all_pts.append(points)


    print("Calculating homograpy matrix...")
    homographies = []

    n_homography_matrices = 4 if all_imgs else 2

    for i in range(n_homography_matrices):
        homographies.append(computeH(all_pts[i*2], all_pts[i*2 + 1]))

    # h_l22l1, h_l12m, h_r12m, h_r2r1

    print("Warping images...")
    warped_imgs = []
    shifts_horizontal = [0]
    shifts_vertical = []

    for im_id, img in enumerate(imgs):
        #ignore base img
        if im_id == int((len(imgs) - 1) /2):
            warped_imgs.append(img/255.0)
            shifts_vertical.append(0)
            continue

        if all_imgs:
            if im_id == 0:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[1].dot(homographies[0]))
            if im_id == 1:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[1])
            if im_id == 3:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[2])
            if im_id == 4:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[2].dot(homographies[3]))
        else:
            if im_id == 0:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[0])
            if im_id == 2:
                warped_pts, warped_img, shifts = warp_pts(img, homographies[1])


        warped_imgs.append(warped_img)
        shifts_horizontal.append(shifts[0])
        shifts_vertical.append(shifts[1])

    shifts_horizontal = np.asarray(shifts_horizontal)
    shifts_vertical = np.asarray(shifts_vertical)

    print("Blending images...")
    blended_img = blend(warped_imgs, shifts_horizontal, shifts_vertical)
    plt.imsave("{}/blended.png".format(exp_folder), blended_img)

def main():
    global normalize, noise, all_imgs

    # ---------------5 POINTS-------------------------------------------------

    normalize = True
    noise = False
    all_imgs = False
    n_points = 5

    exp_folder = "experiments/exp1/5_pts/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"  # "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp1/5_pts/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"  # "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    #--------------------12 POINTS--------------------------------------------

    n_points = 12
    exp_folder = "experiments/exp1/12_pts/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp1/12_pts/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    # -------------------3 WRONG WITHOUT NORMALIZATION---------------------------------------------

    normalize = False
    exp_folder = "experiments/exp2/3_wrg/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp2/3_wrg/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    # -----------------3 WRONG WITH NORMALIZATON-----------------------------------------------
    normalize = True

    exp_folder = "experiments/exp2/3_wrg_norm/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp2/3_wrg_norm/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    # -----------------5 WRONG WITH NORMALIZATON-----------------------------------------------

    exp_folder = "experiments/exp2/5_wrg_norm/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp2/5_wrg_norm/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    #experiment(folder_name, n_points, exp_folder)

    # -----------------NOISE WITH NORMALIZATION-----------------------------------------------
    noise = True

    exp_folder = "experiments/exp3/norm/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp3/norm/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    experiment(folder_name, n_points, exp_folder)

    # -----------------NOISE WITHOUT NORMALIZATION-----------------------------------------------
    noise = True
    normalize = False

    exp_folder = "experiments/exp3/wo_norm/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp3/wo_norm/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    experiment(folder_name, n_points, exp_folder)


    # -----------------PANOROMA-----------------------------------------------
    noise = False
    normalize = True
    all_imgs = True

    exp_folder = "experiments/exp4/cmpe_building"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "cmpe-building/cmpe-building"
    #experiment(folder_name, n_points, exp_folder)

    exp_folder = "experiments/exp4/north_campus"
    os.makedirs(exp_folder, exist_ok=True)
    folder_name = "north_campus"
    #experiment(folder_name, n_points, exp_folder)

if __name__ == "__main__":
    main()
