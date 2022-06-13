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

generator = load_model("generator.pickle")

generator.train = False

latent_size = 2#generator.latent_dim

fixed_noise = torch.randn(params["n_samples"], latent_size, device=device)

with torch.no_grad():
    generated = generator.generate(latent_codes=fixed_noise)

    visualize(generated, path="gan_results.png")

