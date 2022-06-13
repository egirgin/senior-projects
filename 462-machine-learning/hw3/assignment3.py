import argparse
import random

import numpy as np
import pandas as pd

from libsvm.svmutil import *

argparser = argparse.ArgumentParser()

argparser.add_argument("part")
argparser.add_argument("step")

args = argparser.parse_args()

step1_column_names = None


def normalize(dataset):
    if len(dataset.shape) == 1:
        normalized_vector = dataset - dataset.min()
        normalized_vector = normalized_vector / normalized_vector.max()
        return normalized_vector
    else:
        normalized_transpose = np.array([column - column.min() for column in dataset.T])
        normalized_transpose = np.array([column/column.max() for column in normalized_transpose])
        return normalized_transpose.T

def read_data(path, part):
    csv_data = pd.read_csv(path)

    if part == "part1":
        global step1_column_names
        step1_column_names = csv_data.columns

        # Get iris-setosa and iris-versicolor samples
        csv_data_setosa = csv_data[csv_data["class"] == "Iris-setosa"]
        csv_data_setosa = csv_data_setosa.replace("Iris-setosa", -1)
        csv_data_versicolor = csv_data[csv_data["class"] == "Iris-versicolor"]
        csv_data_versicolor = csv_data_versicolor.replace("Iris-versicolor", 1)

        setosa_size = csv_data_setosa.shape[0]


        csv_data = pd.concat([csv_data_setosa, csv_data_versicolor]).to_numpy().astype("float128")
        csv_data[:,:-1] = normalize(csv_data[:,:-1])

        # Train test split
        setosa_train = csv_data[:(setosa_size-10)]
        setosa_test = csv_data[(setosa_size-10):setosa_size]

        versicolor_train = csv_data[setosa_size:-10]
        versicolor_test = csv_data[-10:]

        # Merge
        dataset_train = np.concatenate((setosa_train, versicolor_train), axis=0 )
        dataset_test = np.concatenate((setosa_test, versicolor_test), axis=0 )
    elif part == "part2":
        csv_data = csv_data.drop(columns=["id", "Unnamed: 32"])

        csv_data = csv_data.replace("M", -1)
        csv_data = csv_data.replace("B", 1)

        csv_data = csv_data.to_numpy().astype("float128")
        csv_data[:, 1:] = normalize(csv_data[:, 1:])


        dataset_train = csv_data[:400]
        dataset_test = csv_data[400:]

    
    # Shuffle dataset
    np.random.shuffle(dataset_train)
    np.random.shuffle(dataset_test)

    return dataset_train, dataset_test


class Node:
    def __init__(self):
        self.feature = None
        self.separator = None

        self.is_leaf = None

        self.left = None
        self.right = None

        self.data = None

        self.class_probs = [0, 0]
        self.depth = None

    def set_probs(self):
        self._parse_probs()

    def _parse_probs(self):
        
        self.class_probs[0] = sum(self.data[:,-1] == -1) / len(self.data)

        self.class_probs[1] = sum(self.data[:,-1] == 1) / len(self.data)


class DecisionTree:
    def __init__(self, gain_ratio = False, max_depth=3):
        self.gain_ratio = gain_ratio
        self.root = None
        self.max_depth = max_depth

    def _entropy(self, node):

        if len(np.unique(node[:, -1])) == 1:
            return 0
        elif sum(node[:,-1] == 1) == sum(node[:,-1] == -1):
            return 1
        elif len(node) == 0:
            return 0
        else:
            setosa_size = sum(node[:,-1] == -1)
            versicolor_size = sum(node[:,-1] == 1)
            all_size = setosa_size + versicolor_size

            setosa_ent = -(setosa_size) / all_size

            setosa_ent = setosa_ent * np.log2(-setosa_ent)

            versicolor_ent = -(versicolor_size) / all_size

            versicolor_ent = versicolor_ent * np.log2(-versicolor_ent)

            return setosa_ent + versicolor_ent

    def _separator(self, column):
        return np.mean(column)#random.choice(column)


    def _gain_ratio(self, info_gain, node_sizes):

        total_items = sum(node_sizes)

        denominator = 0

        for node_size in node_sizes:
            node = node_size / total_items

            node = node * np.log2(node)

            node = - node

            denominator += node


        return info_gain / denominator

    def _step(self, subset, node):

        if node.depth >= self.max_depth:
            # if max depth is reached, then the node is lead
            node.is_leaf = True
            return

        if sum(subset[:,-1] == -1) == subset.shape[0] or sum(subset[:,-1] == 1) == subset.shape[0]:
            # if node is pure, then it is a leaf
            node.is_leaf = True
            return

        if subset.shape[0] == 0:
            # if a node is empty, then it is a leaf
            node.is_leaf = True
            return
        
        min_ent = np.inf
        max_gain_ratio = 0
        feature = -1

        last_left = None
        last_right = None
        last_separator = None

        prev_entropy = self._entropy(subset)


        for column_id in range(subset.shape[1]-1):
            separator = self._separator(subset[:, column_id])   

            left_leaf = subset[subset[:, column_id] <= separator]

            right_leaf = subset[subset[:, column_id] > separator]

            entropy = ( len(left_leaf) / (len(left_leaf) + len(right_leaf)) ) * self._entropy(left_leaf) + \
                ( len(right_leaf) / (len(left_leaf) + len(right_leaf)) ) * self._entropy(right_leaf)

            if self.gain_ratio:
                current_gain_ratio = self._gain_ratio(info_gain=prev_entropy-entropy, node_sizes=[len(left_leaf), len(right_leaf)])
            
                if current_gain_ratio > max_gain_ratio:
                    max_gain_ratio = current_gain_ratio
                    feature = column_id
                    last_left = left_leaf
                    last_right = right_leaf
                    last_separator = separator
            else:
                if entropy < min_ent:
                    min_ent = entropy
                    feature = column_id
                    last_left = left_leaf
                    last_right = right_leaf
                    last_separator = separator
                

        if left_leaf.shape[0] == 0 or right_leaf.shape[0] == 0:
            node.is_leaf = True
            return

        # Set parent node details
        node.feature = feature
        node.separator = last_separator

        # Create right node
        right_node = Node()
        right_node.depth = node.depth + 1
        right_node.data = last_right

        # If a child node is empty, then set the parent probs
        if len(last_right) == 0:
            right_node.class_probs = node.class_probs
        else:
            right_node.set_probs()

        # Create left node
        left_node = Node()
        left_node.depth = node.depth + 1
        left_node.data = last_left
        # If a child node is empty, then set the parent probs
        if len(last_left) == 0:
            left_node.class_probs = node.class_probs
        else:
            left_node.set_probs()

        # Set children nodes
        node.right = right_node
        node.left = left_node

        self._step(subset=last_left, node=left_node)
        self._step(subset=last_right, node=right_node)        



    def train(self, trainset):

        self.root = Node()
        self.root.depth = 1

        self.root.data = trainset
        self.root.set_probs()
        
        self._step(subset=trainset, node=self.root)


    def test(self, testset):
        
        preds = []

        for sample in testset:
            current_node = self.root
            while not current_node.is_leaf:
                feature = current_node.feature
                separator = current_node.separator
                
                if sample[feature] <= separator:
                    current_node = current_node.left
                else:
                    current_node = current_node.right


            pred = np.argmax(current_node.class_probs)
            if pred == 0:
                preds.append(-1)
            else:
                preds.append(1)
        return preds



############### PART 2 #############################    

class SVM():

    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C
        self.model = None


    def train(self, trainset):
        m = svm_train(trainset[:,0], trainset[:, 1:], '-t {} -c {} -q'.format(self.kernel, self.C))
        self.model = m


    def test(self, testset):
        p_label, p_acc, p_val = svm_predict(testset[:,0], testset[:, 1:], self.model, "-q")

        print("SVM kernel={} C={} acc={:.2f} n={}".format(self.kernel, self.C, p_acc[0]/100, self.model.get_nr_sv()))

            
        

if __name__ == "__main__":

    if args.part == "part1":

        trainset, testset = read_data("./iris.csv", part=args.part)

        if args.step == "step1":

            decision_tree = DecisionTree(gain_ratio=False, max_depth=3)
            decision_tree.train(trainset=trainset)
            preds = decision_tree.test(testset=testset)
            acc = sum(preds == testset[:,-1]) / len(preds)
            feature_name = step1_column_names[decision_tree.root.feature]
            
            print("DT {} {:.2f}".format(feature_name, acc))


        elif args.step == "step2":

            decision_tree = DecisionTree(gain_ratio=True, max_depth=3)
            decision_tree.train(trainset=trainset)
            preds = decision_tree.test(testset=testset)
            acc = sum(preds == testset[:,-1]) / len(preds)
            feature_name = step1_column_names[decision_tree.root.feature]
            
            print("DT {} {:.2f}".format(feature_name, acc))

    elif args.part == "part2":

        trainset, testset = read_data("./wbcd.csv", part=args.part)    
        
        kernels = [0, 1, 2, 3]

        Cs = [1, 5, 10, 50, 100]


        if args.step == "step1":
            for c in Cs:
                svm = SVM(kernel=1, C=c)
                svm.train(trainset=trainset)
                svm.test(testset=testset)

        elif args.step == "step2":
            for k in kernels:
                svm = SVM(kernel=k, C=1)
                svm.train(trainset=trainset)
                svm.test(testset=testset)
