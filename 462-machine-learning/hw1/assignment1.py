import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

argparser = argparse.ArgumentParser()

argparser.add_argument("part")
argparser.add_argument("step")

args = argparser.parse_args()

step2_dataset1 = "ds1.csv"
step2_dataset2 = "ds2.csv"

part1_iter = 5000
part2_iter = 100 

weight_decay = 559 # lambda

np.random.seed(0)

# -----------------PART1------------------------------
def generate_dataset(size):
    x = np.random.uniform(size=size)
    y = np.random.uniform(size=size)
    
    # Label -1 is used for class 0
    labels = np.array([-1 if y < (-3*x+1) else 1 for (x,y) in zip(x,y)])
    return np.array(list(zip(x, y, labels)))


class PLA:
    
    def __init__(self, delta=None):
        self.delta = delta
        self.weights = np.array([0, 0, 0], dtype="float64") # x, y, bias
        
    def train(self, data, iters):
    
        weights_hist = []
        
        for i in range(iters):
            # Store current classifier to plot
            weights_hist.append(self.weights.copy())
            
            np.random.shuffle(data) # shuffle data so that in every iter get a random example first

            
            for sample in data:
                
                # add bias term to sample
                data_bias = np.concatenate((sample[:2],[1]))
                
                # make prediction
                pred = self.weights.dot(data_bias)
                
                # If delta is provided
                if self.delta:
                    
                    # If class 0
                    if sample[-1] == -1:
                        # Mis classified or low confidence
                        if pred * sample[-1] <= 0 or pred > -self.delta:
                            self.weights += data_bias*sample[-1]
                            break
                        else:
                            continue # correctly classified

                    elif sample[-1] == 1:
                        # Mis classified or low confidence
                        if pred * sample[-1] <= 0 or pred < self.delta:
                            self.weights += data_bias*sample[-1]
                            break
                        else:
                            continue # correctly classified

                # If delta is not provided only check misclassification
                if pred * sample[-1] <= 0:
                    self.weights += data_bias*sample[-1]
                    break
        
        return weights_hist

def plt_classifier(dataset, weights, step):

    x_classifier = 2*(np.random.uniform(size=100)-0.5)
    y_classifier = (x_classifier*weights[0] + weights[2]) / -weights[1]
    
    x_seperator = 2*(np.random.uniform(size=100)-0.5)
    y_seperator = -3*x_seperator+1
    
    plt.plot(x_seperator, y_seperator, c="g", linewidth=3, label="Seperating Function")
        
    plt.plot(x_classifier, y_classifier, c="purple", linewidth=3, label="Classifier")

    ones = dataset[(dataset[:,-1] == 1)]
    zeros = dataset[(dataset[:,-1] == -1)]
    
    plt.scatter(ones[:, 0], ones[:, 1], c="b", label="Class 1", s=5)
    plt.scatter(zeros[:, 0], zeros[:, 1], c="r", label="Class 0", s=5)
    
    
    
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(loc=1)
    title = args.part.upper() + " " + args.step.upper() + " Iter: {}".format(step)
    plt.title(title)
    plt.savefig(args.part + "_" + args.step +  ".png")

# -----------------PART2------------------------------

def read_datasets():
    dataset1 = pd.read_csv(step2_dataset1).to_numpy()
    dataset2 = pd.read_csv(step2_dataset2).to_numpy()

    return dataset1, dataset2

def normalize(dataset):
    if len(dataset.shape) == 1:
        normalized_vector = dataset - dataset.min()
        normalized_vector = normalized_vector / normalized_vector.max()
        return normalized_vector
    else:
        normalized_transpose = np.array([column - column.min() for column in dataset.T])
        normalized_transpose = np.array([column/column.max() for column in normalized_transpose])
        return normalized_transpose.T


class LinearRegression:
    
    def __init__(self, regularization = None, name="Linear_Regression"):
        self.regularization = regularization
        self.weights = None
        self.losses = []
        self.accuracy = []
        self.grads = []
        self.test_losses = []
        self.name = name # Ex: part2_step1
        
    def _init_weights(self, num_features = None):
        # Standard Normal initialization
        weights = np.random.standard_normal(size=(num_features, 1))
        
        self.weights = weights
        
    def _add_bias(self, dataset):
        # Add column of ones to the feature vectors as bias
        bias_column = np.ones_like(dataset.T[0].reshape(-1, 1))
        dataset = np.hstack((dataset, bias_column))
        return dataset
        
    def _loss(self, dataset, labels):
        
        data_size = dataset.shape[0]
        
        # X @ weights
        hypothesis = dataset.dot(self.weights)
                    
        # loss = (1/2*N) * L2_Norm(y_hat - y)^2
        loss = 0.5 * (np.linalg.norm(hypothesis - labels)**2) 
        
        loss /= data_size
        # Loss is scaled in order to approximate per sample loss
        
        if self.regularization:
            # loss = loss + lambda/2 * L2_norm(weights)^2
            loss += self.regularization/2 * np.linalg.norm(self.weights)**2
        
        return loss
    
    def rmse(self, x, y):
        data_size = x.shape[0]
        
        # X @ weights
        hypothesis = x.dot(self.weights)
                    
        mse = sum((hypothesis - y)**2)/data_size 
        
        rmse = np.sqrt(mse)

        return rmse

    def _grad(self, dataset, labels):
        
        # X.transpose @ X @ weights - X.transpose @ labels : X.transpose @ X @ w - X.transpose @ t 
        
        grad = dataset.T.dot(dataset).dot(self.weights) - dataset.T.dot(labels)
        
        if self.regularization:
            # grad + lambda * weights
            grad += self.regularization * self.weights
        
        return grad
    
    def _step(self, grad, lr, dataset):
        
        # Learning rate is scaled to not explode the gradients
        
        data_size = dataset.shape[0]
        
        # W =  W - (alpha/N) * grad
        self.weights = self.weights - (lr/data_size) * grad 
        
    
    def train(self, dataset, labels, epochs, x_test, y_test, lr=0.01):
        labels = labels.reshape(-1, 1)
        dataset = self._add_bias(dataset)
        x_test = self._add_bias(x_test)
        
        if self.weights == None:
            # Random initialize weights
            self._init_weights(dataset.shape[1])
        
        for epoch in range(epochs):
            
            loss = self._loss(dataset, labels) # Calculate loss
            
            grad = self._grad(dataset, labels) # Calculate grad
                        
            self._step(grad, lr, dataset) # Update weights
            
            if epoch > 5:

                self.losses.append(loss)
                
                #print("Epoch: {} -> Loss: {:.3f}".format(epoch, loss))

                self.test_losses.append(self._loss(x_test, y_test))
                
        #self._evaluate(dataset, labels)
        self._save_loss()
                        
    
    def closed_form(self, dataset, labels, x_test, y_test):
        dataset = self._add_bias(dataset)
        num_features = dataset.shape[1]
        x_test = self._add_bias(x_test)
        start = time.time()
    
        if self.regularization:
            # w_star = ((X.transpose @ X + lambda * I ) ^-1) @ X.transpose @ label
            self.weights = np.linalg.inv(dataset.T.dot(dataset) + self.regularization * np.eye(num_features)).dot(dataset.T).dot(labels)
        else:
            # w_star = ((X.transpose @ X) ^-1) @ X.transpose @ label
            self.weights = np.linalg.inv(dataset.T.dot(dataset)).dot(dataset.T).dot(labels)
            
        end = time.time()
        
        print("Time to complete {}: {}msec".format(self.name, int((end-start)*1000) ))
        
        self._evaluate(dataset, labels, True)
        self._evaluate(x_test, y_test, False)
        
    def _evaluate(self, dataset, labels, isTrain):
        loss = self.rmse(dataset, labels)#self._loss(dataset, labels)
        if isTrain:
            print("RMSE for Train Set: {:.5f}".format(loss))
        else:
            print("RMSE for Test Set: {:.5f}".format(loss))
        return loss

    def _save_loss(self):

        size = len(self.losses)

        plt.plot(np.linspace(5, size+5, size), self.losses, label="Train Loss")
        plt.plot(np.linspace(5, size+5, size), self.test_losses, label="Test Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        title = " ".join(self.name.split("_")).upper()

        plt.title(title)
        plt.legend(loc=1)
        plt.savefig(self.name + ".png")
               
if __name__ == "__main__":
    if args.part == "part1":
        # do pla
        if args.step == "step1":
            dataset = generate_dataset(50)
        elif args.step == "step2":
            dataset = generate_dataset(100)
        elif args.step == "step3":
            dataset = generate_dataset(5000)
        else:
            print("Unknown step!")
            exit()

        pla = PLA()

        hist = pla.train(data = dataset, iters = part1_iter)

        plt_classifier(dataset=dataset, weights=hist[-1], step=str(len(hist)))



    elif args.part == "part2":
        ds1, ds2 = read_datasets()
    
        #ds1 = normalize(ds1)
        #ds2 = normalize(ds2)


        if args.step == "step1":
            dataset = ds1
            split = int(dataset.shape[0]*0.2)
            np.random.shuffle(dataset)
            x_train = dataset[split:, :-1]
            y_train = dataset[split:, -1]

            x_test = dataset[:split, :-1]
            y_test = dataset[:split, -1]

            x_dataset = dataset[:,:-1]
            y_dataset = dataset[:,-1]
            lin_reg_closed = LinearRegression(name=args.part+"_"+args.step)
            lin_reg_closed.closed_form(x_train, y_train, x_test, y_test)
            #lin_reg_iter = LinearRegression(name=args.part+"_"+args.step)
            #lin_reg_iter.train(x_train, y_train, part2_iter,  x_test, y_test)

        elif args.step == "step2":
            dataset = ds2
            split = int(dataset.shape[0]*0.2)
            np.random.shuffle(dataset)
            x_train = dataset[split:, :-1]
            y_train = dataset[split:, -1]

            x_test = dataset[:split, :-1]
            y_test = dataset[:split, -1]

            x_dataset = dataset[:,:-1]
            y_dataset = dataset[:,-1]
            lin_reg_closed = LinearRegression(name=args.part+"_"+args.step)
            lin_reg_closed.closed_form(x_train, y_train, x_test, y_test)
            #lin_reg_iter = LinearRegression(name=args.part+"_"+args.step)
            #lin_reg_iter.train(x_train, y_train, part2_iter,  x_test, y_test)

        elif args.step == "step3":
            dataset = ds2
            split = int(dataset.shape[0]*0.2)
            np.random.shuffle(dataset)
            x_train = dataset[split:, :-1]
            y_train = dataset[split:, -1]

            x_test = dataset[:split, :-1]
            y_test = dataset[:split, -1]

            x_dataset = dataset[:,:-1]
            y_dataset = dataset[:,-1]
            lin_reg_closed = LinearRegression(regularization = weight_decay, name=args.part+"_"+args.step)
            lin_reg_closed.closed_form(x_train, y_train, x_test, y_test)
            #lin_reg_iter = LinearRegression(regularization = weight_decay, name=args.part+"_"+args.step)
            #lin_reg_iter.train(x_train, y_train, part2_iter,  x_test, y_test)

        else:
            print("Unknown step!")
            exit()