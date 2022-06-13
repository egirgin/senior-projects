import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 28, 28)
        return img

    def generate(self, latent_codes=None):
        latent_code = latent_codes.reshape((latent_codes.shape[0], int(self.latent_dim)))

        with torch.no_grad():
            outputs = self.forward(latent_code)

        return outputs


class Discriminator(nn.Module):
    def __init__(self, wasserstein=False):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            #nn.Sigmoid(),
        )
        if not wasserstein:
            self.model.append(nn.Sigmoid())

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def WassersteinLoss(preds, labels):

    if labels.mean() == 0:
        return 1 - torch.mean(preds)
    else:
        return torch.mean(preds)

def visualize(generated, path):

    plt.cla()

    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(16, 16))

    for i in range(100):
        ax[i // 10, i % 10].imshow(generated[i].view(28,28).detach().cpu().numpy(), cmap="gray")
        ax[i // 10, i % 10].axis('off')

    fig.tight_layout()

    plt.savefig(path)
    plt.close(fig)

def save_model(model, path):
    with open(path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(path):

    with open(path, 'rb') as handle:
        model = pickle.load(handle)

    return model
