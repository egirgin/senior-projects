import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import ticker, cm, colors


class CustomDataset(Dataset):
    def __init__(self, path, train=True):
        self.root = path

        if train:
            self.X_path = self.root + "/X_train.npy"
            self.y_path = self.root + "/y_train.npy"

        else:
            self.X_path = self.root + "/X_test.npy"
            self.y_path = self.root + "/y_test.npy"

        self.data = np.load(self.X_path)
        self.labels = np.load(self.y_path)


    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        x = torch.from_numpy(x).float()
        return x, y

    def __len__(self):
        return len(self.data)


def plot_decision_boundary(dataset, labels, model, path="./decision-boundary.png"):
    min0 = np.min(dataset[:, 0])
    max0 = np.max(dataset[:, 0])

    min1 = np.min(dataset[:, 1])
    max1 = np.max(dataset[:, 1])

    precision = 100
    x = np.linspace(min0, max0, precision)
    y = np.linspace(min1, max1, precision)

    X, Y = np.meshgrid(x, y)

    z = np.zeros((x.shape[0], y.shape[0]))
    for x_val in range(len(x)):
        for y_val in range(len(y)):
            with torch.no_grad():
                input = torch.Tensor([x[x_val], y[y_val]])
                output = model(input)
                pred = torch.argmax(output).item()
                z[x_val, y_val] = pred


    cmap = colors.ListedColormap(['orange', 'deepskyblue'])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    cs = plt.contourf(X, Y, z, cmap=cmap, norm=norm)

    plt.colorbar(cs, ticks=[2, 1, 0])
    #plt.colorbar(cs)

    class0 = dataset[labels == 0]
    class1 = dataset[labels == 1]

    plt.scatter(class0[:, 0], class0[:, 1], label="class-0", c="b")
    plt.scatter(class1[:, 0], class1[:, 1], label="class-1", c="r")

    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()

    plt.title('Decision Boundary')

    plt.savefig(path)