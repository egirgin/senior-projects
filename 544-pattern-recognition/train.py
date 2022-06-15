import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from sklearn.svm import SVC

from utils import create_dataset, create_model, accuracy, print_losses

np.random.seed(42)

# MACROS
batch_size = 32
epochs = 100
im_size = 256
model_name = "vgg"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained = False
model_path = "path_to_model.pth"
svm = False
hog = False
rejection = False
#print(device)


def train(model, trainloader, valloader, optimizer, criterion):
    train_losses = []
    val_losses = []
    iteration = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        model.train()

        step_size = len(trainloader)/10

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # clear the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))

            # calculate loss
            loss = criterion(outputs, labels.to(device))
            train_losses.append(loss.item())

            # backprop
            loss.backward()
            iteration += 1

            # update weights
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % step_size == 0:  # print every 2000 mini-batches
                print("Epochs: {}, Iteration: {}, Loss: {}".format(epoch + 1, i, running_loss / step_size))
                running_loss = 0.0

        print("Evaluating on validation set...")
        val_acc = 0
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                outputs = torch.argmax(outputs, dim=1)

                val_acc += accuracy(labels.to(device), outputs)
                val_loss += loss.item()
            mean_val_loss = val_loss / (i + 1)
            val_losses.append([mean_val_loss, iteration])
            print("Val loss:{} | Val Acc: {}".format(mean_val_loss, val_acc / i))
        end = time.time()

        print("Took {} secs.".format(end - start))

        print_losses(train_losses, val_losses)

    print('Finished Training')

    return model, train_losses, val_losses


def main():
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
                             std=0.2),
        transforms.Resize((im_size, im_size))
    ])
    if hog:
        trainset, valset = create_dataset(filename="train", n=-1, transform=None, crop=rejection, hog=True)
    else:
        trainset, valset = create_dataset(filename="train", n=-1, transform=my_transforms, crop=rejection, hog=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)

    if svm:
        svm = SVC(C=1.0, kernel='linear', degree=2, gamma="auto")

        svm.fit(trainset.dataset, trainset.label)

        preds = svm.predict(valset.dataset)

        accuracy(valset.label, preds)

    else:

        model = create_model(model_name=model_name, device=device)

        if pretrained:
            model.load_state_dict(torch.load(model_path))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        model, train_loss, val_loss = train(model, trainloader, valloader, optimizer, criterion)


        preds = evaluate(model, testloader)








