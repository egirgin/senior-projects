import numpy as np
import torch

from utils import load_model, visualize

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:{}".format(device))

params = {
    "n_samples": 100,
}

net = load_model("vae_model.pickle").to(device)

net.eval()

hidden_dim_size = net.encoder.mu.out_features

fixed_noise = torch.randn(params["n_samples"], hidden_dim_size, device=device)

with torch.no_grad():
    generated = net.decoder.generate(latent_codes=fixed_noise)

    visualize(generated, path="vae_result.png")

