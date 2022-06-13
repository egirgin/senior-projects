import pickle, time
import torch
import torchvision
from torchvision import transforms
import numpy as np

def load_model(path):

    with open(path, 'rb') as handle:
        model = pickle.load(handle)

    return model


transform = transforms.Compose(
    [transforms.ToTensor()])

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

my_network = load_model("./model.pickle")

total_iter = len(testloader)

counter = 0
cumulative_time = 0

acc = 0
print("Starting Evaluation...")
for i, data in enumerate(testloader):
    counter += 1
    start = time.time()
    inputs, labels = data
    inputs = inputs.detach().numpy()
    labels = labels.detach().numpy()

    o = my_network(inputs, test=True)

    preds = np.argmax(o, axis=1)

    successes = np.sum(preds == labels)

    acc += successes.item()

    end = time.time()

    cumulative_time += (end - start)

    remaining_time = cumulative_time / counter * (total_iter - i)

    #print("ETA: {:.0f}mins".format(remaining_time / 60))

print("Accuracy: {}".format(acc / testset.data.shape[0]))

