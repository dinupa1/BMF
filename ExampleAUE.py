import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Generate example data (signal and background events)
num_samples = 10**5
num_back = int(0.2 * num_samples)
signal_with_background = np.random.normal(1.0, 0.2, num_samples)
background_data = np.random.normal(1.0, 0.2, num_back)

# Assign weights (-1 for background, 1 for signal)
signal_back_weights = np.ones(num_samples)
background_weights = -1 * np.ones(num_back)

# Concatenate signal and background data and weights
data = np.concatenate([signal_with_background, background_data])
weights = np.concatenate([signal_back_weights, background_weights])

# Plot the distributions
bins = np.linspace(0., 2., 31)
plt.hist(signal_with_background, bins=bins, label="signal+background", histtype="step", color="red")
plt.hist(background_data, bins=bins, label="background", histtype="step", color="blue")
plt.hist(data, bins=bins, alpha=0.5, weights=weights, label="signal")
plt.legend(frameon=False)
plt.show()

# Batch size and num epochs
batch_size = 1024
num_epochs = 100

# Convert data and weights to PyTorch tensors
data = torch.tensor(data, dtype=torch.float).reshape(-1, 1)
weights = torch.tensor(weights, dtype=torch.float).reshape(-1, 1)

# Create dataset and data loader
dataset = TensorDataset(data, weights)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define the Variational Autoencoder (VAE) architecture
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h), self.fc22(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Define the loss function (VAE loss)
def vae_loss(recon_x, x, weight_batch, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    BCE_weight = torch.sum(weight_batch * BCE)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE_weight + KLD) / x.size(0)

# Create the VAE
input_dim = 1
hidden_dim = 32
latent_dim = 10
vae = VAE(input_dim, hidden_dim, latent_dim)

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    vae.train()
    for batch_data, batch_weight in dataloader:
        optimizer.zero_grad()
        recon_data, mu, logvar = vae(batch_data)  # Use batch_data
        loss = vae_loss(recon_data, batch_data, batch_weight, mu, logvar)  # Use batch_data and batch_weight
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")


# Generate test data (signal and background events)
num_samples = 10**4
num_back = int(0.2 * num_samples)
signal_with_background = np.random.normal(1.0, 0.2, num_samples)
background_data = np.random.normal(1.0, 0.2, num_back)

# Assign weights (-1 for background, 1 for signal)
signal_back_weights = np.ones(num_samples)
background_weights = -1 * np.ones(num_back)

# Concatenate signal and background data and weights
data = np.concatenate([signal_with_background, background_data])
weights = np.concatenate([signal_back_weights, background_weights])

# Generate background-subtracted data
vae.eval()
with torch.no_grad():
    # background_subtracted_data, _, _ = vae(torch.Tensor(data).reshape(-1, 1))
    background_subtracted_data = vae.decode(torch.normal(0.0, 0.5, size=(num_samples, latent_dim)).float())

background_subtracted_data = background_subtracted_data.detach().numpy()


# Plot the background subtracted events
bins = np.linspace(0., 2., 31)
# plt.hist(signal_with_background, bins=bins, label="signal+background", histtype="step", color="blue")
plt.hist(data, bins=bins, alpha=0.5, weights=weights, label="signal")
plt.hist(background_subtracted_data, bins=bins, label="VAE signal", histtype="step", color="red")
plt.legend(frameon=False)
plt.show()
