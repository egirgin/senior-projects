import time

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

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
#print(device)

def evaluate(model, testloader):
    start = time.time()

    model.eval()

    test_acc = 0
    result = []

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            outputs = model(inputs.to(device))
            outputs = torch.argmax(outputs, dim=1)
            result.append(outputs)
            test_acc += accuracy(labels.to(device), outputs)

        print("Test Acc: {}".format(test_acc / i))
    end = time.time()

    print("Took {} secs.".format(end - start))

    return torch.as_tensor(outputs)

def main():
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5,
                             std=0.2),
        transforms.Resize((im_size, im_size))
    ])

    testset = create_dataset(filename="test", n=-1, transform=my_transforms, crop=True)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = create_model(model_name="vgg", device=device)



    if pretrained:
        model.load_state_dict(torch.load(model_path))


    preds = evaluate(model, testloader)








