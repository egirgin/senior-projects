import os, time

import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from skimage.feature import hog

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.models as models


class CustomClassifier(torch.nn.Module):
    def __init__(self, D_in):

        super(CustomClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=D_in, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=5)
        self.dropout = nn.Dropout(P=0.5)

    def forward(self, x):

        x = x.view(-1, 256 * 2 * 2)  # FLATTEN
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class EmotionDataset(Dataset):

    def __init__(self, dataset_list, label_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset_list
        self.label = label_list
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.dataset[idx], self.label[idx]]

        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample


def read_dataset(filename="train", n=5):
    classes = os.listdir("./{}".format(filename))

    dataset = []
    labels = []

    for emotion in classes:
        img_names = os.listdir("./{}/{}".format(filename, emotion))

        if n != -1:
            img_names = np.random.choice(img_names, size=n)

        for img_name in img_names:
            img_obj = cv2.imread("./{}/{}/{}".format(filename, emotion, img_name))

            img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)

            dataset.append(img_obj)
            if emotion == "Angry":
                labels.append(0)
            elif emotion == "Upset":
                labels.append(1)
            elif emotion == "Happy":
                labels.append(2)
            elif emotion == "Sleepy":
                labels.append(3)
            elif emotion == "Suprised":
                labels.append(4)
            else:
                print(emotion)

    return dataset, labels

def create_dataset(filename="train", n=5, transform=None, crop=False, hog=False):

    if crop:
        dataset, labels = cropped_dataset(filename, "./models", rejection_th=0.1) # TODO ENTER REJECTION THRESHOLD HERE
    else:
        dataset, labels = read_dataset(filename, n)

    if hog:
        dataset = hog_transform(dataset)

    print("creating dataset objects")

    if filename == "train":
        X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=42)
        emotion_dataset_train = EmotionDataset(X_train, y_train, transform=transform)
        emotion_dataset_val = EmotionDataset(X_val, y_val, transform=transform)

        return emotion_dataset_train, emotion_dataset_val
    else:
        emotion_dataset_test = EmotionDataset(dataset, labels, transform=transform)
        return emotion_dataset_test


def create_model(model_name="vgg", device="cpu"):
    if model_name == "vgg":
        model = models.vgg16(pretrained=True, progress=False).to(device)
        model = nn.Sequential(model, nn.Linear(1000, 5), nn.Softmax(dim=1)).to(device)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True, progress=False).to(device)
        model = nn.Sequential(model, nn.Linear(1000, 5), nn.Softmax(dim=1)).to(device)
    elif model_name == "resnet":
        model = models.resnet18(pretrained=True).to(device)
        model = nn.Sequential(model, nn.Linear(1000, 5), nn.Softmax(dim=1)).to(device)
    elif model_name == "custom":
        model = CustomClassifier(D_in=500).to(device) # TODO

    return model


def accuracy(groundtruths, preds):
    return sum(groundtruths==preds)/len(preds)


def print_losses(train_loss, val_loss):
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss[:,1], val_loss[:,0], label="Val Loss")
    plt.legend()
    plt.title("Loss Value")
    plt.imsave("loss.png")

def cropped_dataset(dataset_dir, models_dir, rejection_th):
    def makeItSquare(x_start, y_start, x_end, y_end, h, w):
        y_diff = y_end - y_start
        x_diff = x_end - x_start
        dim_diff = x_diff - y_diff

        if dim_diff > 0:
            if dim_diff % 2 == 0:
                y_start -= dim_diff // 2
                y_end += dim_diff // 2
            else:
                y_start -= dim_diff // 2 + 1
                y_end += dim_diff // 2
        elif dim_diff < 0:
            dim_diff *= -1
            if dim_diff % 2 == 0:
                x_start -= dim_diff // 2
                x_end += dim_diff // 2
            else:
                x_start -= dim_diff // 2 + 1
                x_end += dim_diff // 2

        if x_start < 0:
            x_end -= x_start
            x_start -= x_start
        elif x_end > w - 1:
            x_start -= x_end - (w - 1)
            x_end -= x_end - (w - 1)
        if x_start < 0:
            x_end -= x_start
            x_start -= x_start
        elif y_end > h - 1:
            y_start -= y_end - (h - 1)
            y_end -= y_end - (h - 1)

        return np.array([x_start, y_start, x_end, y_end])

    # import the model
    modelFile = os.path.join(models_dir, 'opencv_face_detector_uint8.pb')
    configFile = os.path.join(models_dir, 'opencv_face_detector.pbtxt')
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    classes = os.listdir("./{}".format(dataset_dir))

    dataset = []
    labels = []

    for emotion in classes:
        img_names = os.listdir("./{}/{}".format(dataset_dir, emotion))

        for img_name in img_names:
            img_obj = cv2.imread("./{}/{}/{}".format(dataset_dir, emotion, img_name))

            h, w = img_obj.shape[:-1]

            blob = cv2.dnn.blobFromImage(img_obj, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            confidence = detections[0, 0, 0, 2]  # highest confidence

            if confidence < rejection_th:
                continue

            x1 = int(detections[0, 0, 0, 3] * w)
            y1 = int(detections[0, 0, 0, 4] * h)
            x2 = int(detections[0, 0, 0, 5] * w)
            y2 = int(detections[0, 0, 0, 6] * h)
            x1, y1, x2, y2 = makeItSquare(x1, y1, x2, y2, h, w)

            cropped_img = img_obj[y1:y2, x1:x2, :]

            dataset.append(cropped_img)

            if emotion == "Angry":
                labels.append(0)
            elif emotion == "Upset":
                labels.append(1)
            elif emotion == "Happy":
                labels.append(2)
            elif emotion == "Sleepy":
                labels.append(3)
            elif emotion == "Suprised":
                labels.append(4)
            else:
                print(emotion)

    return np.array(dataset), np.array(labels)


def hog_transform(imgs):
    result = []
    for id, img in enumerate(imgs):
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(4, 4),
                          cells_per_block=(1, 1), visualize=True)

        if id % 100 == 0:
            print("Progress: {:.0f}%".format(id * 100 / len(imgs)))

        result.append(fd)

    return np.asarray(result)