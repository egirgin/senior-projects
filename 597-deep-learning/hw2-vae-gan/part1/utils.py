import pickle
import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d, BCELoss, MSELoss, BatchNorm2d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=hidden_dim, num_layers=1, dropout=0.0, batch_first=True)

        self.mu = nn.Linear(hidden_dim, latent_dim) # from 16 to 2
        self.var = nn.Linear(hidden_dim, latent_dim) # from 16 to 2


    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)  # batch, Seq-length, Hidden

        #print(output.shape)  # B x 28 x H
        #print(h_n.shape)  # 1 x B x H
        #print(c_n.shape)  # 1 x B x H

        z_mu = self.mu(h_n[0])
        z_var = self.var(h_n[0])
        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self, input_dim = 2):
        super(Decoder, self).__init__()

        self.input_dim = input_dim

        self.linear = nn.Linear(in_features=self.input_dim, out_features=64*7*7)

        self.convT1 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.convT2 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.convT3 = ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear(x)

        x = x.view(x.shape[0], 64, 7, 7)
        #print(x.shape)
        x = F.relu(self.convT1(x))
        x = self.bn1(x)
        #print(x.shape)
        x = F.relu(self.convT2(x))
        x = self.bn2(x)
        #print(x.shape)
        x = self.sigmoid(self.convT3(x))
        #print(x.shape)
        return x

    def generate(self, latent_codes=None):

        with torch.no_grad():

            latent_codes = latent_codes.reshape((latent_codes.shape[0], self.input_dim))

            outputs = self.forward(latent_codes)

        return outputs


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):

        mu, var = self.encoder(x)

        # sampling
        std = torch.exp(var / 2)
        eps = torch.randn_like(std)
        latent_code = eps * std + mu

        #print(torch.mean(latent_code))
        #print(torch.std(latent_code))

        # generate
        predicted = self.decoder(latent_code)

        return predicted, mu, var


def BCE_KL_loss(prediction, latent_mu, latent_var, original, kl=False):

    """
    why do we compare with the standard normal:
     https://ai.stackexchange.com/questions/18390/why-do-we-regularize-the-variational-autoencoder-with-a-normal-distribution
    :param prediction:
    :param original:
    :param bce:
    :return:
    """
    mean_pred = torch.mean(latent_mu)

    std_pred = torch.exp(latent_var / 2)

    kl_loss = -0.5 * (1 + torch.log(std_pred**2) - latent_mu**2 - std_pred**2)
    kl_loss = torch.mean(torch.sum(kl_loss, dim=1))


    error = F.binary_cross_entropy(prediction, original)

    if kl:
        return error + kl_loss
    else:
        return error

def visualize(generated, path):

    plt.cla()

    fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(16, 16))

    for i in range(100):
        ax[i // 10, i % 10].imshow(generated[i, 0].detach().cpu().numpy(), cmap="gray")
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
