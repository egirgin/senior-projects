import pickle
import time
import torch
import torchvision
from torchvision import transforms
import numpy as np
from alternativelib.network import Network
from alternativelib.dense import FullyConnected, Flatten
from alternativelib.activations import *
from alternativelib.losses import CrossEntropy
from alternativelib.conv import conv2d, MaxPool2d

lightweight = False

np.random.seed(42)
if lightweight:
    params = {
        "epochs": 2,
        "batch_size": 8,
        "lr": 0.01,
        "momentum": 0.9
    }
else:
    params = {
        "epochs": 2,
        "batch_size": 8,
        "lr": 0.01,
        "momentum": 0.9
    }
def save_model(model, path):
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"],
                                          shuffle=True, num_workers=2)

my_network = Network()

if lightweight:
    flatten = Flatten(batch_size=params["batch_size"])

    fc1 = FullyConnected(in_dims=28*28, out_dims=784, batch_size=params["batch_size"])
    relu3 = relu()
    fc2 = FullyConnected(in_dims=784, out_dims=10, batch_size=params["batch_size"])
    linear = linear()
else:
    conv1 = conv2d(in_channels=1, out_channels=4, filter_size=5, batch_size=params["batch_size"])
    relu1 = relu()
    maxpool1 = MaxPool2d(kernel_size=2)

    conv2 = conv2d(in_channels=4, out_channels=8, filter_size=5, batch_size=params["batch_size"])
    relu2 = relu()
    maxpool2 = MaxPool2d(kernel_size=2)

    flatten = Flatten(batch_size=params["batch_size"])

    fc1 = FullyConnected(in_dims=4*4*8, out_dims=128, batch_size=params["batch_size"])
    relu3 = relu()
    fc2 = FullyConnected(in_dims=128, out_dims=10, batch_size=params["batch_size"])
    linear = linear()

loss = CrossEntropy(use_softmax=True, batch_size=params["batch_size"])

# -------------------------------------------------------
if not lightweight:
    my_network.add_layer(conv1)
    my_network.add_layer(relu1)
    my_network.add_layer(maxpool1)

    my_network.add_layer(conv2)
    my_network.add_layer(relu2)
    my_network.add_layer(maxpool2)

my_network.add_layer(flatten)

my_network.add_layer(fc1)
my_network.add_layer(relu3)

my_network.add_layer(fc2)
my_network.add_layer(linear)

total_iter = len(trainloader) * params["epochs"]

counter = 0
cumulative_time = 0
try:
    for epoch in range(params["epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            counter += 1
            start = time.time()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.detach().numpy()
            labels = labels.detach().numpy()

            o = my_network(inputs)

            #print(np.argmax(o, axis=1))

            loss_deriv = loss._backward(o, labels)

            my_network.backwards(loss_deriv)

            my_network.update(lr=params["lr"], momentum=params["momentum"])

            end = time.time()

            cumulative_time += (end-start)

            remaining_time = cumulative_time/counter * (total_iter - i*(epoch+1))

            print("Epochs: {} | Iter:{} | Loss: {} | ETA: {:.0f}mins".format(epoch, i, loss(o, labels), remaining_time/60))

except Exception as e:
    print(e)
finally:
    save_model(my_network, "./model.pickle")
    print('Model saved successfully. Finished Training')