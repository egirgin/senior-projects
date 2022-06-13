import time
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F

from utils import Generator, Discriminator, save_model, visualize, WassersteinLoss

import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:{}".format(device))

params = {
    "epochs": 100,
    "batch_size": 32,
    "lrG": 0.0002,
    "lrD": 0.0002,
    "latent_size": 2,
    "n_samples": 100,
    "wasserstein": True,
    "k": 3
}
params["k"] = params["k"] if params["wasserstein"] else 1

transform = transforms.Compose(
    [transforms.ToTensor()])


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"],
                                          shuffle=True, num_workers=2)

# Create the generator
netG = Generator(latent_dim=params["latent_size"]).to(device)
netD = Discriminator(wasserstein=params["wasserstein"]).to(device)

fixed_noise = torch.randn(params["n_samples"], params["latent_size"], device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = -1. if params["wasserstein"] else 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=params["lrD"], betas=(0.5, 0.999), amsgrad=True)
optimizerG = optim.Adam(netG.parameters(), lr=params["lrG"], betas=(0.5, 0.999), amsgrad=True)

if params["wasserstein"]:
    loss = WassersteinLoss
else:
    loss = F.binary_cross_entropy

lossG_history = []
lossD_history = []

predD_history_fake = []
predD_history_real = []

time_history = []
for epoch in range(params["epochs"]):  # loop over the dataset multiple times

    running_lossG = 0.0
    running_lossD = 0.0
    running_predR = []
    running_predF = []

    start = time.time()
    for i, data in enumerate(trainloader):
        # PREPARE DATA

        # create real labels
        labels_real = torch.full((params["batch_size"],), real_label, dtype=torch.float, device=device)

        # create fake labels
        labels_fake = torch.full((params["batch_size"],), fake_label, dtype=torch.float, device=device)

        # create random noise
        noise = torch.randn(params["batch_size"], params["latent_size"], device=device)

        # get real inputs
        inputs, _ = data
        inputs = inputs.to(device)

        # TRAIN GENERATOR -> maximize log(D(G(z)))

        # clear generator params
        netG.zero_grad()

        # create random noise
        noise = torch.randn(params["batch_size"], params["latent_size"], device=device)

        # create fake input
        fake_input = netG(noise)

        # discriminate fake inputs
        outputs = netD(fake_input).view(-1)

        # calculate error
        if params["wasserstein"]:
            lossG = - torch.mean(outputs)
        else:
            lossG = loss(outputs, labels_real)

        # backprop
        lossG.backward()
        running_lossG += lossG.item()  # EXTRA


        optimizerG.step()

        for step in range(params["k"]):

            # create fake data
            fake_input = netG(noise)

            # TRAIN DISCRIMINATOR -> maximize log(D(x)) + log(1 - D(G(z)))

            # clear discriminator params
            optimizerD.zero_grad()

            ## REAL DATA

            # discriminate real data
            outputs_real = netD(inputs).view(-1)

            # calculate error
            lossD_real = loss(outputs_real, labels_real)
            running_lossD += lossD_real.item()  # EXTRA

            # keep track of the discriminator bias
            running_predR.append(outputs_real.mean().item())  # EXTRA

            ## FAKE DATA
            # discriminate fake data
            outputs_fake = netD(fake_input.detach()).view(-1)

            # calculate error
            lossD_fake = loss(outputs_fake, labels_fake)
            running_lossD += lossD_fake.item()  # EXTRA

            # keep track of the discriminator bias
            running_predF.append(outputs_fake.mean().item())  # EXTRA

            # Error
            #lossD_fake.backward()
            #lossD_real.backward()
            if params["wasserstein"]:
                lossD = - (torch.mean(outputs_real) - torch.mean(outputs_fake))
            else:
                lossD = (lossD_real + lossD_fake) / 2

            # backprop
            lossD.backward()

            # update weights
            optimizerD.step()

            # gradient clipping
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

        if ((i / len(trainloader) ) * 100) % 1 == 0:
            print("Progress: {:.2f} | Disc. Loss: {:.2f} | Gen. Loss: {:.2f} | Disc. Fake Avg.: {:.2f} | Disc. Real Avg.: {:.2f}".format(
                i / len(trainloader),
                running_lossD / (2*(i+1)*params["k"]),
                running_lossG / (i+1),
                np.mean(running_predF),
                np.mean(running_predR)
            ))

    end = time.time()

    time_history.append((end-start) / 60)

    epoch_lossD = running_lossD / (2*len(trainloader)*params["k"])  # times 2 bcs for real and fake data
    epoch_lossG = running_lossG / len(trainloader)

    print(
        "Epoch:{} | Discriminator Loss: {:.2f} | Generator Loss: {:.2f}| ETA: {:.2f}mins".format(
            epoch,
            epoch_lossD,
            epoch_lossG,
            np.mean(time_history) * (params["epochs"] - epoch)))

    lossD_history.append(epoch_lossD)
    lossG_history.append(epoch_lossG)
    predD_history_real.append(np.mean(running_predR))
    predD_history_fake.append(np.mean(running_predF))

    with torch.no_grad():

        generated = netG.generate(latent_codes=fixed_noise)

        visualize(generated, path="gan_samples.png")

    save_model(netG, "generator.pickle")
    save_model(netD, "discriminator.pickle")

    plt.cla()
    plt.plot(predD_history_fake, label="Fake")
    plt.plot(predD_history_real, label="Real")
    plt.xlabel("Iterations")
    plt.ylabel("Prediction")
    plt.title("Discriminator Predictions")
    plt.legend()
    plt.savefig("disc_pred.png")



    plt.cla()
    plt.plot(lossD_history, label="Discriminator")
    plt.plot(lossG_history, label="Generator")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("GAN Loss")
    plt.legend()
    plt.savefig("gan_loss.png")

    plt.close("all")

