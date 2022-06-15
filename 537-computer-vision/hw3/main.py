import math
import os, random, pickle

import cv2
import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
np.random.seed(42)

class dataset():
    def __init__(self, train_path="./Caltech20/training", test_path="./Caltech20/testing", sample=False, sample_number=100):
        self.train_path = train_path
        self.test_path = test_path
        self.class_names = None
        self.trainset = None
        self.testset = None
        self.img_size = 256
        self.sample = sample
        self.sample_number = sample_number

        self._read_datasets()

    def _read_datasets(self):

        print("Reading dataset...")

        self.class_names = [(name, idx) for idx, name in enumerate(os.listdir(self.train_path))]

        train_set = []

        for cls_name, cls_id in self.class_names:
            class_counter = 0

            if self.sample:
                while class_counter < self.sample_number:
                    for img_name in os.listdir(self.train_path + "/" + cls_name):
                        if self.sample and class_counter >= self.sample_number:
                            break
                        img = cv2.imread(self.train_path + "/" + cls_name + "/" + img_name)
                        img = self._normalize(img)
                        train_set.append((img, cls_id))
                        class_counter += 1
            else:
                for img_name in os.listdir(self.train_path + "/" + cls_name):
                    img = cv2.imread(self.train_path + "/" + cls_name + "/" + img_name)
                    img = self._normalize(img)
                    train_set.append((img, cls_id))

        random.shuffle(train_set)
        x_train = np.asarray([sample[0] for sample in train_set])
        y_train = np.asarray([sample[1] for sample in train_set])

        self.trainset = (x_train, y_train)

        test_set = []

        for cls_name, cls_id in self.class_names:

            if cls_name not in os.listdir(self.test_path):
                # ignore background class
                continue

            for img_name in os.listdir(self.test_path + "/" + cls_name):
                img = cv2.imread(self.test_path + "/" + cls_name + "/" + img_name)
                img = self._normalize(img)
                test_set.append((img, cls_id))

        random.shuffle(test_set)
        x_test = np.asarray([sample[0] for sample in test_set])
        y_test = np.asarray([sample[1] for sample in test_set])

        self.testset = (x_test, y_test)


    def _normalize(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (self.img_size, self.img_size))

        return img

def sift_features(X):

    sift = cv2.SIFT_create()

    descriptors = np.zeros((1,128))

    n_descs = []

    for img_id, img in enumerate(X):
        kp = sift.detect(img, None)
        kp, des = sift.compute(img, kp)
        if des is None:
            n_descs.append(0)
        else:
            n_descs.append(des.shape[0])
            descriptors = np.vstack((descriptors, des))

    descriptors = descriptors[1:]

    return descriptors, n_descs


def quantize(preds, n_descs, K):

    features = np.zeros((len(n_descs), K))

    index = 0
    for sample_id, n_kp in enumerate(n_descs):
        if n_kp != 0:
            new_descs = preds[index:index+n_kp]

            hist = np.histogram(new_descs, bins=range(K+1), density=True)[0]

            features[sample_id, :] = hist

        index += n_kp

    return features


def show_img(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram(array, name, title):

    unique, counts = np.unique(array, return_counts=True)
    a = dict(zip(unique, counts))

    for idx, k in enumerate(a.keys()):
        plt.bar(k, a[k], color="tab:blue")
        plt.text(k, a[k], str(a[k]), ha="center")

    plt.xlabel("Class ID")
    plt.ylabel("Frequency")
    plt.xticks(unique)
    plt.title(title)
    plt.savefig(name)
    plt.close()

def main():

    # read dataset
    data = dataset(sample=True, sample_number=100)

    x_train, y_train = data.trainset
    x_test, y_test = data.testset

    histogram(y_train, "train_data.png", "Training Data")
    histogram(y_test, "test_data.png", "Test Data")

    ############################SIFT######################################################
    print("Calculating SIFT features...")
    # calculate descriptors
    train_desc, train_n_descs = sift_features(x_train)

    # calculate test descriptors
    test_desc, test_n_descs = sift_features(x_test)

    ############################KMEANS######################################################
    print("Transforming using K-Means and Histogram...")
    # calculate clusters based on train descriptors
    K = 500  # TODO
    kmeans = KMeans(n_clusters=K).fit(train_desc)

    train_kmeans_preds = kmeans.labels_
    test_kmeans_preds = kmeans.predict(test_desc)

    # Convert each img descriptors to the hists
    X_train = quantize(train_kmeans_preds, train_n_descs, K)

    X_test = quantize(test_kmeans_preds, test_n_descs, K)

    ############################SVM######################################################
    # Classify
    parameters = {
        'kernel': [chi2_kernel],
        'C': [math.pow(10, -1)],#[math.pow(10, i) for i in np.arange(-2, 3, 1).astype(float)]
    }
    svc = svm.SVC(class_weight = "balanced")
    clf = GridSearchCV(svc, parameters, verbose=0) #RandomizedSearchCV(svc, parameters, verbose=3)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    train_preds = clf.predict(X_train)
    print("Test Acc: {}".format(sum(preds == y_test) / len(preds)))

    #print("Train Acc: {}".format(sum(train_preds == y_train) / len(train_preds)))

    histogram(preds, "preds.png", "Test Predictions")
    
    ############################EVALUATION######################################################


    x_ticks_int = [i[1] for i in data.class_names] + [len(data.class_names)]
    x_ticks_str = [i[0] + ":" + str(i[1]) for i in data.class_names] + ["Mean"]

    f1_scores = f1_score(y_test, preds, average=None, labels=[i[1] for i in data.class_names], zero_division=0.0)

    macro_avg = f1_scores.mean()
    f1_scores = f1_scores.tolist()
    f1_scores.append(macro_avg)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(32,6), gridspec_kw={'width_ratios': [4, 1]})

    for idx, score in enumerate(f1_scores):
        ax[0].bar(x_ticks_int[idx], score, color="tab:blue")
        ax[0].text(x_ticks_int[idx], score, str(score)[:4], ha="center")

    ax[0].set_xlabel("Class Names")
    ax[0].set_ylabel("F1-Score")
    ax[0].set_xticks(x_ticks_int, x_ticks_str, size='small')
    ax[0].title.set_text("F1 Scores")


    conf_matrix = confusion_matrix(y_test, preds, labels=range(len(data.class_names)))
    ax[1] = sns.heatmap(data=conf_matrix, ax=ax[1])
    ax[1].title.set_text("Confusion Matrix")
    fig.set_tight_layout(True)
    fig.savefig("confusion_matrix.png")

    ############################PLOTTING######################################################
    """
    mis_classifications = x_test[preds != y_test]

    fig, ax = plt.subplots(nrows= 5 , ncols= 5, figsize=(100, 100))

    for idx, im_id in enumerate(np.random.choice(range(len(mis_classifications)), 25)):
        mis_img = mis_classifications[im_id]
        ax[idx//5, idx%5].imshow(mis_img, cmap="gray")

    fig.savefig("misclassified.png")
    """

if __name__ == "__main__":
    main()