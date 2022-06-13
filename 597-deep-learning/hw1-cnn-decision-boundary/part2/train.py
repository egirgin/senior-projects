import numpy as np
import torch
import torch.nn as nn
from network import CustomNetwork
import torch.optim as optim

from torchvision import transforms
from dataset import CustomDataset, plot_decision_boundary

np.random.seed(42)
torch.manual_seed(42)

debug = False

data_path = "./data/data"

params = {
    "epochs": 100,
    "batch_size": 16,
    "lr": 0.1,
    "momentum": 0.9
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:{}".format(device))

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = CustomDataset(path=data_path, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"],
                                          shuffle=True, num_workers=2)

testset = CustomDataset(path=data_path, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"],
                                         shuffle=False, num_workers=2)

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

    

    print("Epoch:{} | Loss: {}, Accuracy: {}".format(epoch, running_loss / len(trainloader), acc/testset.data.shape[0]))
    running_loss = 0.0


plot_decision_boundary(dataset=testset.data, labels=testset.labels, model=net)
print('Finished Training')

