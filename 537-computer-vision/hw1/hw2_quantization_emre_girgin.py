import sys

import cv2.cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

manual = True


def read_img(img_name):
    im = cv2.imread(img_name)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    return im


def preprocess(gray, name):

    im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    if name == "birds3.jpg":
        im = cv2.dilate(im, np.ones((7, 7), np.uint8), iterations=1)
        im = cv2.erode(im, np.ones((5, 5), np.uint8), iterations=1)
    elif name == "demo4.png":

        im = cv2.dilate(im, np.ones((2, 2), np.uint8), iterations=5)

        im = cv2.erode(im, np.ones((3, 3), np.uint8), iterations=4)

        im = cv2.erode(im, np.ones((2, 2), np.uint8), iterations=7)

        im = cv2.dilate(im, np.ones((3, 3), np.uint8), iterations=3)

    elif name == "dice5.PNG" or name == "dice6.PNG":
        im = cv2.dilate(im, np.ones((3, 3), np.uint8), iterations=1)

    #plt.imshow(im)
    #plt.show()

    return im / 255.0


def checkNeighbors(img, row, column):
    """
        7 0 1
        6 * 2
        5 4 3

    :param img: [width, height] 2D np array
    :param row: int
    :param column: int
    :return: [0, 1, 2, 3, 4, 5, 6, 7]
    """

    neighbor_classes = []
    neighbor_classes.append(img[row - 1, column] if row > 0 else 0)  # 0
    neighbor_classes.append(img[row - 1, column + 1] if (row > 0 and column < img.shape[1] - 1) else 0)  # 1
    neighbor_classes.append(img[row, column + 1] if column < img.shape[1] - 1 else 0)  # 2
    neighbor_classes.append(
        img[row + 1, column + 1] if (row < img.shape[0] - 1 and column < img.shape[1] - 1) else 0)  # 3
    neighbor_classes.append(img[row + 1, column] if row < img.shape[0] - 1 else 0)  # 4
    neighbor_classes.append(img[row + 1, column - 1] if (row < img.shape[0] - 1 and column > 0) else 0)  # 5
    neighbor_classes.append(img[row, column - 1] if column > 0 else 0)  # 6
    neighbor_classes.append(img[row - 1, column - 1] if (row > 0 and column > 0) else 0)  # 7

    return np.array(neighbor_classes)


# Implementation of row-by-row cca algorithm from https://youtu.be/hMIrQdX4BkE
def countConnectedComponents(img):
    components = []

    graph = np.copy(img)

    # give each non zero point a class
    cc = 0
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):

            if column > 0 and img[row, column - 1] == 0 and img[row, column] == 1:
                cc += 1
            if img[row, column] == 0:
                continue
            elif img[row, column] == 1:
                graph[row, column] = cc

    # check neighbor classes
    for row in range(graph.shape[0]):
        for column in range(graph.shape[1]):
            if graph[row, column] == 0:
                continue
            else:
                neighbors = checkNeighbors(graph, row, column)

                neighbors = np.append(neighbors, graph[row, column])

                neighbors = list(set(neighbors[neighbors > 0]))

                flag = False
                for neigh in neighbors:
                    for idx, comp in enumerate(components):
                        if neigh in comp:
                            comp += neighbors
                            comp = list(set(comp))
                            components[idx] = comp
                            flag = True

                if not flag:
                    components.append(neighbors)

    comp_final = components.copy()

    # match the classes that belongs to the same component
    pairs = []
    for idx1, comp1 in enumerate(components):
        for idx2, comp2 in enumerate(components):
            for elem in comp2:
                if elem in comp1 and idx1 != idx2:
                    pairs.append([idx1, idx2] if idx1 < idx2 else [idx2, idx1])

    # remove duplicate pairs
    if len(pairs) != 0:
        pairs = np.unique(pairs, axis=0)

        for pair in pairs:
            comp_final[pair[0]] = list(set(comp_final[pair[0]] + comp_final[pair[1]]))

        comp_final = comp_final

        comp_newest = []
        for idx, comp in enumerate(comp_final):
            if idx not in pairs[:, 1]:
                comp_newest.append(comp)
    else:
        comp_newest = comp_final

    # assign new class id to each component
    for row in range(graph.shape[0]):
        for column in range(graph.shape[1]):
            if graph[row, column] == 0:
                continue
            else:
                val = graph[row, column]
                for idx, comp in enumerate(comp_newest):
                    if val in comp:
                        graph[row, column] = idx + 1

    obj_count, colored = color(graph)

    return obj_count, colored


def color(img):
    uniques = np.unique(img)
    uniques = uniques[uniques > 0]

    unique_count = np.zeros_like(uniques)

    for idx, distinct in enumerate(uniques):
        unique_count[idx] = sum(sum(img == distinct)) / (img.shape[0] * img.shape[1])

    colored = np.zeros((img.shape[0], img.shape[1], 3))

    obj_counter = 0

    for idx, distinct in enumerate(uniques):
        if unique_count[idx] < 0.4:
            colored[img == distinct] = np.random.randint(0, 255, 3)
            obj_counter += 1

    return obj_counter, colored / 255.0


def main():
    names = ["birds1.jpg", "birds2.jpg", "birds3.jpg", "demo4.png", "dice5.PNG", "dice6.PNG"]


    for name in names:
        fig = plt.figure()
        im = read_img("./{}".format(name))

        im = preprocess(im, name)

        os.makedirs("output", exist_ok=True)

        obj_count, colored = countConnectedComponents(im)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(im, cmap="Greys")
        ax1.set_title("Grey")
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(colored)
        ax2.set_title("Colored:{}".format(obj_count))
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])

        fig.savefig("output/{}".format(name), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
