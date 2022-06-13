import argparse
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

argparser = argparse.ArgumentParser()

argparser.add_argument("part")
argparser.add_argument("step")

args = argparser.parse_args()

np.random.seed(42)

debug = True

if debug:
    os.makedirs("debug", exist_ok=True)

def normalize(dataset):
    if len(dataset.shape) == 1:
        normalized_vector = dataset - dataset.min()
        normalized_vector = normalized_vector / normalized_vector.max()
        return normalized_vector
    else:
        normalized_transpose = np.array([column - column.min() for column in dataset.T])
        normalized_transpose = np.array([column/column.max() for column in normalized_transpose])
        return normalized_transpose.T

def read_data(path):
    csv_data = pd.read_csv(path)

    # Get saab and van samples
    csv_data_saab = csv_data[csv_data["Class"] == "saab"]
    csv_data_saab = csv_data_saab.assign(Class=-1) 
    csv_data_van = csv_data[csv_data["Class"] == "van"]
    csv_data_van = csv_data_van.assign(Class=1)

    # Merge
    dataset = [csv_data_van, csv_data_saab]
    csv_dataset = pd.concat(dataset)
    
    # Shuffle dataset
    csv_dataset = csv_dataset.sample(frac=1).reset_index(drop=True)                                                             

    dataset_numpy = csv_dataset.to_numpy().astype("float128")

    dataset_numpy[:,:-1] = normalize(dataset_numpy[:,:-1])

    return dataset_numpy
    
class LogisticRegression:
    def __init__(self, kfold, regularization=None, name=None) -> None:
        self.kfold = kfold
        self.regularization = regularization
        self.weights = None
        self.avg_train_loss = []
        self.train_losses = []
        self.val_losses = []
        self.accuracy = []
        self.test_losses = []
        self.num_batches = None
        self.name = name # Ex: part2_step1
        self.duration = 0

    def _add_bias(self, dataset):
        # Add column of ones to the feature vectors as bias
        bias_column = np.ones_like(dataset.T[0].reshape(-1, 1))
        dataset = np.hstack((dataset, bias_column))
        return dataset

    def _init_weights(self, num_features = None):
        # Standard Normal initialization
        weights = np.random.standard_normal(size=(self.kfold, num_features, 1)).astype("float128")
        
        self.weights = weights

    def _gradient(self, data, labels, weights):

        batchsize = data.shape[0]

        grad = (-labels * data) * self._sigmoid(-labels*data.dot(weights))

        grad = (1/batchsize) * sum(grad) 

        return grad.reshape(-1,1)


    def dataloader(self, data, kfold = None):

        labels = data[:,-1]
        labels = labels.reshape(-1, 1)
        data = data[:,:-1]
        data = self._add_bias(data)
        
        if kfold:
            datasize = data.shape[0]
            
            # Discard last few samples to equally divide dataset
            excess = datasize%kfold
            data = data[:-excess]
            labels = labels[:-excess]
            
            folds = np.split(data, kfold) 
            labels = np.split(labels, kfold)

            return np.array(folds).astype("float128"), np.array(labels).astype("float128")
        else:
            return np.array(data).astype("float128"), np.array(labels).astype("float128")



    def _sigmoid(self,value):
        return 1/(1 + np.exp(-value))
    
    def _step(self, grad, lr, dataset):
        
        # Learning rate is scaled to not explode the gradients
        
        data_size = dataset.shape[0]
        
        # W =  W - (alpha/N) * grad
        self.weights = self.weights - (lr/data_size) * grad

    def _loss(self, data, labels, weights):
        
        datasize = data.shape[0]

        loss = np.log(1 + np.exp(-labels * data.dot(weights)))

        loss = (1/datasize) * sum(loss)

        return loss[0]

    def _accuracy(self, data, labels, weights):
        pred = self._sigmoid(data.dot(weights))

        pred[pred <= 0.5] = -1
        pred[pred > 0.5] = 1

        mask = pred == labels
        
        return (sum(mask) / len(mask))[0]

    def train(self, dataset, lr, epochs = 10, batch_size=-1):
        
        threshold = 0.00001

        # divide data into folds
        x, y = self.dataloader(dataset, self.kfold)

        if batch_size == -1:
            # full batch
            batch_size = x.shape[1] * (self.kfold -1)

        num_batches = int((x.shape[1] * (self.kfold -1)) / batch_size)
        self.num_batches = num_batches

        folds = []

        for i, fold in enumerate(x):
            mask = np.ones(shape=(x.shape[0]), dtype=bool)
            mask[i] = False
            # merge each fold's training folds and split to batches 
            x_train = np.array_split(x[mask].reshape(-1, x.shape[-1]), num_batches)
            y_train = np.array_split(y[mask].reshape(-1, 1), num_batches)
            # no split for the validation set
            x_test = x[i].reshape(-1, x.shape[-1])
            y_test = y[i].reshape(-1, 1)

            folds.append(np.array([x_train, y_train, x_test, y_test], dtype=object))
            
        folds = np.array(folds)

        if self.weights == None:
            # Random initialize weights
            self._init_weights(dataset.shape[1])

        
        start = time.time()
        for fold_id in range(self.kfold):

            train_loss = []
            val_loss = []
            val_acc = []
            e = 1
            iter = 1

            while True:
            
                # iterate over batches
                for batch_id in range(num_batches):

                    # each fold brings its own batch and labels
                    data = np.array(folds[fold_id][0][batch_id])
                    labels = np.array(folds[fold_id][1][batch_id])

                    # calculate gradient for batches of each fold and update the weights of each fold

                    grad = np.array(
                        self._gradient(data=data, labels=labels, weights=self.weights[fold_id])
                    )
                    self.weights[fold_id] = self.weights[fold_id] - lr * grad

                    train_loss.append(
                        self._loss(data=data, labels=labels, weights=self.weights[fold_id])
                    )
                    iter += 1


                # Calculate Validation loss and acc
                val_loss.append(
                    self._loss(data=folds[fold_id][2], labels=folds[fold_id][3], weights=self.weights[fold_id])
                )
                val_acc.append(
                    self._accuracy(data=folds[fold_id][2], labels=folds[fold_id][3], weights=self.weights[fold_id])
                )

                # Check for convergence
                epoch_loss = sum(train_loss[-num_batches:])
                prev_epoch_loss = sum(train_loss[-2*num_batches:-num_batches])
                diff = np.abs(epoch_loss - prev_epoch_loss)

                if debug:
                    print("Iteration: {} | FoldID: {} | LearningRate: {} | Batch Size: {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Accuracy: {:.2f} | Diff: {:.5f}".format(
                            iter, fold_id+1, lr, batch_size, train_loss[-1], val_loss[-1], val_acc[-1], diff
                            )
                        )

                e += 1
                if e > 1 and diff <= threshold:
                    break

            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracy.append(val_acc)

        end = time.time()

        self.duration = int(end - start)


if __name__ == '__main__':

    dataset = read_data("./vehicle.csv")

    lrs = [0.01, 0.5, 1.0]
    if args.step == "step1":

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
        axes[0][0].set_title("First Fold Logistic Loss")
        axes[0][1].set_title("First Fold Accuracy")


        for i, lr in enumerate(lrs):
            if not debug:
                print("Training Full Batch Logistic Regression with LR:{}".format(lr))
            my_lr = LogisticRegression(kfold = 5)
            my_lr.train(dataset, lr)
            if not debug:
                print("Took {:.0f}secs".format(my_lr.duration))

            if debug:
                fig_debug, axes_debug = plt.subplots(nrows=my_lr.kfold, ncols=2, figsize=(12,8))
                axes_debug[0][0].set_title("Logistic Loss")
                axes_debug[0][1].set_title("Accuracy")


                for fold in range(my_lr.kfold):
                    axes_debug[fold, 0].set_ylabel("Fold {}".format(fold+1))
                    val_axis = np.arange(start=0, step=1, stop=len(my_lr.val_losses[fold])) * my_lr.num_batches
                    axes_debug[fold][0].plot(my_lr.train_losses[fold], label="train loss")
                    axes_debug[fold][0].plot(val_axis, my_lr.val_losses[fold], label="val loss")
                    axes_debug[fold][0].legend()

                    axes_debug[fold][1].plot(val_axis, my_lr.accuracy[fold], label="acc")
                    axes_debug[fold][1].legend()

                fig_debug.tight_layout()
                plt.figure(fig_debug.number)
                plt.savefig("./debug/part1_step1_lr{}.png".format(i), dpi=300)



            axes[i, 0].set_ylabel("LR: {} | Time: {:.0f}secs".format(lr, my_lr.duration))
            val_axis = np.arange(start=0, step=1, stop=len(my_lr.val_losses[0])) * my_lr.num_batches
            axes[i][0].plot(my_lr.train_losses[0], label="train loss")
            axes[i][0].plot(val_axis, my_lr.val_losses[0], label="val loss")
            axes[i][0].legend()

            
            axes[i][1].plot(val_axis, my_lr.accuracy[0], label="acc")
            axes[i][1].legend()


        fig.tight_layout()
        plt.figure(fig.number)
        plt.savefig("./part1_step1.png", dpi=300)

    elif args.step == "step2":
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,8))
        axes[0][0].set_title("First Fold Logistic Loss")
        axes[0][1].set_title("First Fold Accuracy")


        for i, lr in enumerate(lrs):
            if not debug:
                print("Training Mini-Batch Logistic Regression with LR:{}".format(lr))
            my_lr = LogisticRegression(kfold = 5)
            my_lr.train(dataset, lr, batch_size=64)
            if not debug:
                print("Took {:.0f}secs".format(my_lr.duration))

            if debug:
                fig_debug, axes_debug = plt.subplots(nrows=my_lr.kfold, ncols=2, figsize=(12,8))
                axes_debug[0][0].set_title("Logistic Loss")
                axes_debug[0][1].set_title("Accuracy")


                for fold in range(my_lr.kfold):
                    axes_debug[fold, 0].set_ylabel("Fold {}".format(fold+1))
                    val_axis = np.arange(start=0, step=1, stop=len(my_lr.val_losses[fold])) * my_lr.num_batches
                    axes_debug[fold][0].plot(my_lr.train_losses[fold], label="train loss")
                    axes_debug[fold][0].plot(val_axis, my_lr.val_losses[fold], label="val loss")
                    axes_debug[fold][0].legend()

                    axes_debug[fold][1].plot(val_axis, my_lr.accuracy[fold], label="acc")
                    axes_debug[fold][1].legend()

                fig_debug.tight_layout()
                plt.figure(fig_debug.number)
                plt.savefig("./debug/part1_step2_lr{}.png".format(i), dpi=300)



            axes[i, 0].set_ylabel("LR: {} | Time: {:.0f}secs".format(lr, my_lr.duration))
            val_axis = np.arange(start=0, step=1, stop=len(my_lr.val_losses[0])) * my_lr.num_batches
            axes[i][0].plot(my_lr.train_losses[0], label="train loss")
            axes[i][0].plot(val_axis, my_lr.val_losses[0], label="val loss")
            axes[i][0].legend()

            
            axes[i][1].plot(val_axis, my_lr.accuracy[0], label="acc")
            axes[i][1].legend()


        fig.tight_layout()
        plt.figure(fig.number)
        plt.savefig("./part1_step2.png", dpi=300)

  
