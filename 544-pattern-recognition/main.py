#!gdown --id 1c4UDghtTbniTynyz_bbkhyC2DX5-7tBF
#!unzip cropped.zip

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

from skimage.feature import hog, local_binary_pattern
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

from scipy.stats import norm

np.random.seed(42)

# Feature extraction techniques
_lda = False
_eigenfaces = False
_sift = False
_haar = False
_hog = True
_lbp = False
_gabor = False
# do not use feature e
_vanilla = False
# use a deep model as face cropper
_deepmodel = True
# use center cropping
_hard_cropped = False

"""# normalize, read_imgs, dump_processed, pick sample"""

def construct_dataset(dataset_dir, models_dir):
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

    dataset = []
    labels = []

    count = 0
    for angle in [0, 45, 90, 135, 180]:
        angle_dir = os.path.join(dataset_dir, str(angle))
        for filename in os.listdir(angle_dir):
            if not filename.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(angle_dir, filename))
            h, w = img.shape[:-1]

            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            detections = net.forward()
            confidence = detections[0, 0, 0, 2]  # highest confidence
            x1 = int(detections[0, 0, 0, 3] * w)
            y1 = int(detections[0, 0, 0, 4] * h)
            x2 = int(detections[0, 0, 0, 5] * w)
            y2 = int(detections[0, 0, 0, 6] * h)
            x1, y1, x2, y2 = makeItSquare(x1, y1, x2, y2, h, w)

            cropped_img = img[y1:y2, x1:x2, :]
            cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resized_cropped_gray = cv2.resize(cropped_gray, (150, 150))

            dataset.append(resized_cropped_gray)
            labels.append(angle)

    return np.array(dataset), np.array(labels)

def normalize(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def read_imgs(filename="train", n=5):
    classes = os.listdir("./{}".format(filename))

    dataset = []
    labels = []

    for degree in classes:
        img_names = os.listdir("./{}/{}".format(filename, degree))

        if n != -1:
            img_names = np.random.choice(img_names, size=n)

        for img_name in img_names:
            img_obj = cv2.imread("./{}/{}/{}".format(filename, degree, img_name))
            if filename == "train":
                img_obj = normalize(img_obj)
                img_obj = crop_hard(img_obj)
            if filename == "cropped":
                img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)
            dataset.append(img_obj)
            labels.append(int(degree))

    return np.asarray(dataset), np.asarray(labels)

def dump_processed():

    os.makedirs("cropped", exist_ok=True)

    classes = os.listdir("./train")

    for degree in classes:
        os.makedirs("./cropped/{}".format(degree), exist_ok=True)

        img_names = os.listdir("./train/{}".format(degree))

        for img_name in img_names:
            img_obj = cv2.imread("./train/{}/{}".format(degree, img_name))
            img_obj = normalize(img_obj)
            img_obj = crop_hard(img_obj)
            cv2.imwrite("./cropped/{}/{}".format(degree, img_name), img_obj)


def pick_sample(degree):

    img_name = np.random.choice(os.listdir("./cropped/{}".format(degree)))

    img = cv2.imread("./cropped/{}/{}".format(degree, img_name))

    img = normalize(img)
    return img

"""#Face Crop"""

### face cropping

def crop_hard(img):
    row_c = int(img.shape[0]/2)
    col_c = int(img.shape[1]/2)

    return img[row_c-75:row_c+75, col_c-75:col_c+75]

def crop_faces(img):
    def getLargestBox(boxes):
        i = np.argmax([width * height for (column, row, width, height) in detected_faces])
        return boxes[i]

    model_dir = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(model_dir)

    detected_faces = face_cascade.detectMultiScale(img, 1.1, 3)
    if len(detected_faces) == 0:
        print("hard crop")
        return crop_hard(img)

    (x, y, w, h) = getLargestBox(detected_faces)
    cropped_img = img[y:y + h, x:x + w]

    return cropped_img

"""# Feature Extraction"""

def gabor_filtering(imgs):
    def build_filters():
        filters = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 16):
            kern = cv2.getGaborKernel(ksize=(ksize, ksize),
                                      sigma=4.0, theta=theta,
                                      lambd=10.0, gamma=0.5,
                                      psi=0, ktype=cv2.CV_32F)

            kern /= 1.5 * kern.sum()
            filters.append(kern)
        return filters

    filters = build_filters()

    result = []
    for id, img in enumerate(imgs):
        accum = np.zeros_like(img)
        for kern in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            accum = np.maximum(accum, fimg)
        if id %100 == 0:
            print("Progress: {:.0f}%".format(id*100/len(imgs)))
        result.append(accum)
    return np.asarray(result)


def hog_transform(imgs):
    result = []
    for id, img in enumerate(imgs):
        fd, hog_img = hog(img, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True)

        if id %100 == 0:
            print("Progress: {:.0f}%".format(id*100/len(imgs)))
        
        result.append(fd)

    return np.asarray(result)

def lbp(imgs, radius=1):

    n_points = 8 * radius
    result = []

    for id, img in enumerate(imgs):
        lbp = local_binary_pattern(img, n_points, radius)
        
        n_bins = int(lbp.max() + 1)

        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        
        
        if id %100 == 0:
            print("Progress: {:.0f}%".format(id*100/len(imgs)))
        result.append(lbp.reshape(-1))

    return np.asarray(result)

def sift_transform(imgs):
    sift = cv2.SIFT_create()

    result = []
    for id, img in enumerate(imgs):

        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        kp, des = sift.detectAndCompute(img, None)
        if id % 100 == 0:
            print("Progress: {:.0f}%".format(id * 100 / len(imgs)))
        result.append(np.mean(des,axis=0))

    return np.asarray(result)


def haar_features(imgs):

    feature_types = ['type-4']

    def extract_feature_image(img, feature_type, feature_coord=None):
        ii = integral_image(img)
        return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=feature_type,
                                 feature_coord=feature_coord)

    result = []
    for id, img in enumerate(imgs):
        features = extract_feature_image(img, feature_types)
        if id %100 == 0:
            print("Progress: {:.0f}%".format(id*100/len(imgs)))
        result.append(features)


    return np.asarray(result)


def sift_bovw(train_x, train_y, test_x, test_y, n_classes=5): # not used
    # cluster all descriptors to create a visual vocab
    k = 10 * n_classes
    sift = cv2.SIFT_create()

    dico = []
    for img in train_x:
        kp, des = sift.detectAndCompute(img, None)
        for d in des:
            dico.append(d)
    kmeans = KMeans(n_clusters=k).fit(dico)
    kmeans.verbose = False

    histo_list = []
    for img in X_train:
        kp, des = sift.detectAndCompute(img, None)
        histo = np.zeros(k)
        nkp = np.size(kp)
        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly
        histo_list.append(histo)

    # map test input to visual words and count occurances
    histo_list2 = []
    for img in X_test:
        kp, des = sift.detectAndCompute(img, None)
        histo = np.zeros(k)
        nkp = np.size(kp)
        for d in des:
            idx = kmeans.predict([d])
            histo[idx] += 1 / nkp  # Because we need normalized histograms, I prefere to add 1/nkp directly
        histo_list2.append(histo)

    # construct transformed test data___
    X_train_trans = np.array(histo_list)
    X_test_trans = np.array(histo_list2)

    return X_train_trans, X_test_trans

"""# Cluster, Classify"""

def cluster(X_train, y_train, X_test, y_test):

    kmeans = KMeans(n_clusters=5).fit(X_train)

    train_preds = kmeans.predict(X_train)


    cluster_id_list = range(5)

    test_preds = kmeans.predict(X_test)

    preds_0 = X_test[test_preds == cluster_id_list[0]]
    preds_1 = X_test[test_preds == cluster_id_list[1]]
    preds_2 = X_test[test_preds == cluster_id_list[2]]
    preds_3 = X_test[test_preds == cluster_id_list[3]]
    preds_4 = X_test[test_preds == cluster_id_list[4]]

    all_preds = [preds_0, preds_1, preds_2, preds_3, preds_4]

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))


    for cls_id, cls in enumerate(all_preds):
        dist2center = []
        max_dist = 0
        min_dist = np.inf
        for point1 in cls:
            for point2 in cls: # other points in that cluster
                max_dist_temp = np.linalg.norm(point1-point2)
                if max_dist_temp > max_dist:
                    max_dist = max_dist_temp

            for other_point in X_test[test_preds != cluster_id_list[cls_id]]: # all other points that does not belong to that cluster
                min_dist_temp = np.linalg.norm(point1 - other_point)
                if min_dist_temp < min_dist:
                    min_dist = min_dist_temp

            dist2center.append(np.linalg.norm(point1 - kmeans.cluster_centers_[cluster_id_list[cls_id]]))

        dist2center = np.asarray(dist2center)

        xs = np.arange(0, max_dist, 0.01)
        ax[cls_id].plot(xs, norm.pdf(xs, dist2center.mean(), np.sqrt(np.var(dist2center))), label="Avg. Intra-Class Dist.")
        ax[cls_id].axvline(x=min_dist, label="Min. Inter-Class Dist", c="r")
        ax[cls_id].axvline(x=max_dist, label="Max. Intra-Class Dist", c="g")

        print("Cluster ID: {}".format(cls_id))
        print("Intra-class distance mean: {:.4f}".format(dist2center.mean()))
        print("Intra-class distance variance: {:.4f}".format(np.var(dist2center)))
        print("Intra-class max distance: {:.4f}".format(max_dist))
        print("Inter-class min dist: {:.4f}".format(min_dist))
        print("-----------------------------------------------")
    plt.legend(loc="upper right")
    plt.show()
    if X_train.shape[1] == 2:

        plt.scatter(X_train[:, 0], X_train[:, 1], c=(train_preds/45).astype(int))
        plt.show()

        plt.scatter(X_test[:, 0], X_test[:, 1], c=(test_preds / 45).astype(int))
        plt.show()





def classify(X_train, y_train, X_test, y_test, X_test_original):

    svm = SVC(C=1.0, kernel='linear', degree=2, gamma="auto")

    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)

    mis_preds = X_test_original[preds != y_test]
    mis_pred_gts = y_test[preds != y_test]
    mis_pred_preds = preds[preds != y_test]

    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(12, 12))

    for idx, img_id in enumerate(np.random.choice(range(mis_preds.shape[0]), size=25)):
        img_reshaped = mis_preds[img_id].reshape(150, 150)
        ax[int(idx//5), idx % 5].imshow(X=img_reshaped, cmap="gray")
        ax[int(idx // 5), idx % 5].axis("off")
        ax[int(idx//5), idx % 5].set_title("GT:{}/P:{}".format(mis_pred_gts[img_id], mis_pred_preds[img_id]))

    plt.show()

    accs = []
    for cls_id in np.unique(y_test):
        cls_preds = preds[y_test == cls_id]
        cls_gts = y_test[y_test == cls_id]

        cls_acc = np.sum(cls_preds == cls_gts) / cls_preds.shape[0]

        accs.append(cls_acc)
        print("Class {} acc: {:.4f}".format(cls_id, cls_acc))


    print("Overall acc: {:.4f}".format(1 - (mis_preds.shape[0] / X_test.shape[0])))
    print("F1 per class: {}".format(f1_score(y_test, preds, average=None)))
    print("F1 overall: {:.4f}".format(f1_score(y_test, preds, average="macro")))

"""# Eigenfaces, Fisherfaces"""

def eigenfaces(dataset, rank=1):
    #[h, w] = dataset.shape[2:]

    pca = PCA(n_components=rank, whiten=True).fit(dataset)

    #eigenfaces = pca.components_.reshape((rank, h, w))

    #transformed = pca.transform(dataset)

    return pca

def fisherfaces(X_train, y_train):

    lda = LinearDiscriminantAnalysis()

    lda.fit(X_train, y_train)

    return lda

"""# Save, read"""

def save_transformed(data, name):
    with open("./{}.npy".format(name), "bw+") as npy_file:
        np.save(npy_file, data)

def read_transformed(name):
    with open("./{}.npy".format(name), "br") as npy_file:
        return np.load(npy_file, allow_pickle=True)

"""# Do the thing!"""

if _deepmodel:
    print("Cropping Faces using deep model...")
    dataset, labels = construct_dataset("./train", "./models")
elif _hard_cropped:
    print("Using hard cropped version.")
    dataset, labels = read_imgs(filename="cropped", n=-1)
else:
    print("Using raw data to be cropped.")
    dataset, labels = read_imgs(filename="train", n=-1) # retrieve all of the images


X_train, X_test, y_train, y_test = train_test_split(
        dataset/255.0, labels, test_size=0.33, random_state=1)

X_test_original = X_test

print("Dataset has been loaded.")

if _lbp:
    print("LBP Active")
    X_train = lbp(X_train)
    X_test = lbp(X_test)

if _hog: # need dimensionality reduction
    print("HOG Active")
    """
    if "x_train.npy" in os.listdir("."):
        X_train = read_transformed("x_train")
    if "x_test.npy" in os.listdir("."):
        X_test = read_transformed("x_test")

    else:
        X_train = hog_transform(X_train)
        X_test = hog_transform(X_test)
        save_transformed(X_train, "x_train")
        save_transformed(X_test, "x_test")
    """

    X_train = hog_transform(X_train)
    X_test = hog_transform(X_test)

if _gabor: # need dimensionality reduction
    print("Gabor Active")
    X_train = gabor_filtering(X_train*255).reshape(X_train.shape[0], -1)/255.0
    X_test = gabor_filtering(X_test*255).reshape(X_test.shape[0], -1)/255.0
    
if _sift:  # TODO: each output has different size
    print("SIFT Active")
    X_train = np.asarray(X_train * 255, dtype="uint8")
    X_test = np.asarray(X_test * 255, dtype="uint8")
    X_train = sift_transform(X_train).reshape(X_train.shape[0], -1)
    X_test = sift_transform(X_test).reshape(X_test.shape[0], -1)

if _haar: # bad performance
    print("Haar Active")
    X_train = hog_transform(X_train)
    X_test = hog_transform(X_test)

if _eigenfaces: # bad clustring, good classification,
    print("Eigenfaces Active")
    eigenface_transformer = eigenfaces(X_train.reshape(X_train.shape[0], -1), rank=100)

    X_train = eigenface_transformer.transform(X_train.reshape(X_train.shape[0], -1))
    X_test = eigenface_transformer.transform(X_test.reshape(X_test.shape[0], -1))

if _lda:
    print("LDA Active")
    lda_transformer = fisherfaces(X_train.reshape(X_train.shape[0], -1), y_train)

    X_train = lda_transformer.transform(X_train.reshape(X_train.shape[0], -1))
    X_test = lda_transformer.transform(X_test.reshape(X_test.shape[0], -1))

if _vanilla:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

print("Transformation completed.")
print("Sample size: {}".format(X_train[0].shape))
cluster(X_train, y_train, X_test, y_test)

print("Accuracy:")
classify(X_train, y_train, X_test, y_test, X_test_original)

