import time, os
import numpy as np
import torch
import torchvision
import torch.optim as optim
from torchvision import transforms

from utils import VAE, Encoder, Decoder, BCE_KL_loss, save_model, visualize

import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:{}".format(device))

params = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "latent_size": 2,
    "encoder_hidden_size": 16,
    "n_samples": 100,
    "kl_regularization": False
}

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"],
                                          shuffle=True, num_workers=2)


encoder = Encoder(hidden_dim=params["encoder_hidden_size"],latent_dim=params["latent_size"]).to(device)

decoder = Decoder(input_dim=params["latent_size"]).to(device)

vae = VAE(encoder=encoder, decoder=decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=params["lr"], betas=(0.5, 0.999))

fixed_noise = torch.randn(params["n_samples"], params["latent_size"], device=device)

loss_history = []
time_history = []
for epoch in range(params["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0

    start = time.time()
    for i, data in enumerate(trainloader, 0):

        inputs, _ = data
        inputs = torch.squeeze(inputs, dim=1).to(device)

        # clear the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, latent_mu, latent_var = vae(inputs)

        outputs = torch.squeeze(outputs, dim=1) # remove the dummy channel dimension since it is grayscale

        # calculate loss
        loss = BCE_KL_loss(outputs, latent_mu, latent_var, inputs, kl=params["kl_regularization"])

        # backprop
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()


    end = time.time()

    time_history.append((end-start) / 60)

    print(
        "Epoch:{} | Loss: {:.2f} | ETA: {:.2f}mins".format(
            epoch,
            running_loss / len(trainloader),
            np.mean(time_history) * (params["epochs"] - epoch)))

    loss_history.append(running_loss / len(trainloader))
    running_loss = 0.0

    with torch.no_grad():

        generated = decoder.generate(latent_codes=fixed_noise)

        visualize(generated, path="vae_samples.png")

    save_model(vae, "vae_model.pickle")

plt.cla()
plt.plot(loss_history)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("VAE Loss")
plt.savefig("vae_loss.png")

