import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

step2_dataset1 = "ds1.csv"
step2_dataset2 = "ds2.csv"

part1_iter = 5000
part2_iter = 5000 

weight_decay = 1e-18 # lambda

np.random.seed(0)


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
        """
        if self.regularization:
            # loss = loss + lambda/2 * L2_norm(weights)^2
            loss += self.regularization/2 * np.linalg.norm(self.weights)**2
        """
        return loss

    def rmse(self, x, y):
        x = self._add_bias(x)
        data_size = x.shape[0]
        
        # X @ weights
        hypothesis = x.dot(self.weights)
                    
        # loss = (1/2*N) * L2_Norm(y_hat - y)^2
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
                        
    
    def closed_form(self, dataset, labels):
        dataset = self._add_bias(dataset)
        num_features = dataset.shape[1]
    
        if self.regularization:
            # w_star = ((X.transpose @ X + lambda * I ) ^-1) @ X.transpose @ label
            self.weights = np.linalg.inv(dataset.T.dot(dataset) + self.regularization * np.eye(num_features)).dot(dataset.T).dot(labels)
        else:
            # w_star = ((X.transpose @ X) ^-1) @ X.transpose @ label
            self.weights = np.linalg.inv(dataset.T.dot(dataset)).dot(dataset.T).dot(labels)


    def _evaluate(self, dataset, labels):
        dataset = self._add_bias(dataset)
        loss = self._loss(dataset, labels)
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


ds1, ds2 = read_datasets()

#ds2 = normalize(ds2)
dataset = ds2
split = int(dataset.shape[0]*0.2)
np.random.shuffle(dataset)
x_train = dataset[split:, :-1]
y_train = dataset[split:, -1]

x_test = dataset[:split, :-1]
y_test = dataset[:split, -1]

x_dataset = dataset[:,:-1]
y_dataset = dataset[:,-1]

train_losses = []
test_losses = []

rng = np.linspace(1, 1000, 1000)

lambdas = np.linspace(1, 1000, 1000)

min_lambda = 0
min_loss = 99999

for i in rng:
    weight_decay = i
    linreg = LinearRegression(regularization= weight_decay)
    linreg.closed_form(x_train, y_train)

    
    train_loss = linreg.rmse(x_train, y_train)
    train_losses.append(train_loss)

    test_loss = linreg.rmse(x_test, y_test)
    test_losses.append(test_loss)

    if test_loss < min_loss:
        min_loss = test_loss
        min_lambda = i


xs = rng # np.linspace(0, -rng, rng) #lambdas

print(min_lambda)
print(min_loss)


plt.plot(xs, train_losses, label = "train")
plt.plot(xs, test_losses, label = "test")
plt.plot(min_lambda, min_loss, 'go', label="Min Lambda: {}".format(min_lambda))

plt.xlabel("lambda")
plt.ylabel("Rmse")

plt.legend(loc=1)
plt.savefig("./lambda.png")