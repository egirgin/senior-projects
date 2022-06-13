import cv2.cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time

manual = False

def read_img(img_name):

    im = cv2.imread(img_name)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def generate_initals(K=2):
    return np.random.uniform(0, 255, (K, 3))

def manual_initials(im, K):
    plt.imshow(im)
    pts = np.array(plt.ginput(K, show_clicks=True), dtype=int)
    centroids = []
    for p in pts:
        centroids.append(im[p[1], p[0]])
    plt.cla()
    plt.clf()
    plt.close()
    return np.array(centroids).astype(np.float64)

def kmeans(data, centroids):

    dist_matrix = np.zeros((data.shape[0], centroids.shape[0]))

    # assign the closest centroid
    for idx, c in enumerate(centroids):
        dist_matrix[:, idx] = np.linalg.norm(data-c, axis=1) # l2 as distance metric

    cluster_matrix = np.argmin(dist_matrix, axis=1)


    # shift centroids
    for idx, c in enumerate(centroids):

        if len(cluster_matrix[cluster_matrix == idx]) != 0:
            centroids[idx] = np.mean(data[cluster_matrix == idx], axis=0)

    return cluster_matrix, np.asarray(centroids)


def quantize(img, k):

    shape = img.shape

    img = img.reshape(-1, 3)

    if manual:
        centroids = manual_initials(img.reshape(shape), K=k)
        #print(centroids)
    else:
        centroids = generate_initals(K=k)

    for iter in range(10):
        clusters, centroids = kmeans(img, centroids)

    output = np.zeros_like(img)

    # color based on clusters
    for idx, _ in enumerate(centroids):
        mask = clusters == idx
        output[mask] = centroids[idx]

    output = output.reshape(shape)

    return output


def main():
    img_list = ["frog.jpg", "panda.jpg", "boun.jpg"]
    os.makedirs("output", exist_ok=True)
    global manual

    for img_name in img_list:
        img = read_img("./{}".format(img_name))

        # pick initials randomly
        
        manual = False
        fig = plt.figure()

        for id, k in enumerate([2, 4, 8, 16, 32]):
            output = quantize(img, k)
            ax = fig.add_subplot(1, 5, id+1)
            ax.imshow(output)
            ax.set_title("K:{}".format(k))
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        fig.savefig("output/random_{}".format(img_name), dpi=300, bbox_inches="tight")


        fig.clf()
        plt.close(fig)


        # pick initials manually
        manual = True

        fig = plt.figure()

        for id, k in enumerate([2, 4, 8, 16, 32]):
            output = quantize(img, k)
            ax = fig.add_subplot(1, 5, id+1)
            ax.imshow(output)
            ax.set_title("K:{}".format(k))
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        fig.savefig("output/manual_{}".format(img_name), dpi=300, bbox_inches="tight")

        fig.clf()
        plt.close(fig)


if __name__ == "__main__":
    main()
