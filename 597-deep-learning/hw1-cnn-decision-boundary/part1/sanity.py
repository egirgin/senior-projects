import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np

params = {
    "epochs": 2,
    "batch_size": 8,
    "lr": 0.01,
    "momentum": 0.9
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:{}".format(device))

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"],
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"],
                                         shuffle=False, num_workers=2)


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # LAYER 1&2
        x = self.pool(F.relu(self.conv2(x)))  # LAYER 3&4
        x = x.view(-1, 8 * 4 * 4)  # FLATTEN
        x = F.relu(self.fc1(x))  # LAYER 5
        x = self.fc2(x)  # LAYER 6

        return x


net = CustomNetwork()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=params["lr"], momentum=params["momentum"])


for epoch in range(params["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # clear the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        # calculate loss
        loss = criterion(outputs, labels)

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print("Epochs: {}, Iteration: {}, Loss: {}".format(epoch + 1, i, running_loss / 100))
            running_loss = 0.0

    acc = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + optimize
        outputs = net(inputs)

        preds = torch.argmax(outputs, dim=1)

        successes = torch.sum(preds == labels)

        acc += successes.item()

    print("Accuracy: {}".format(acc/testset.data.shape[0]))

print('Finished Training')