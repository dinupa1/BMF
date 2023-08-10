#
# dinupa3@gmail.com
# 08-08-2023
#

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import uproot
import awkward as ak

from BackgroundVDM import BackgroundVDM
from BackgroundVDM import VDMLoss

# Some conts.
batch_size = 1024
num_epochs = 25

# Load E906 LH2 data
tree = uproot.open("LH2Data.root:tree")
events = tree.arrays(["mass", "pT", "xF", "phi", "costh", "weight"])

X = np.array([(mass, pT, xF, phi, costh) for mass, pT, xF, phi, costh in zip(events.mass, events.pT, events.xF, events.phi, events.costh)])
weight = np.array([(weight) for weight in events.weight])

# Convert to torch tensor
X_tensor = torch.from_numpy(X).float()
weight_tensor = torch.from_numpy(weight.reshape(-1, 1)).float()

dataset = TensorDataset(X_tensor, weight_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create the VDM
input_dim = 5
hidden_dim = 10
latent_dim = 5
time_steps = 1000
vdm = BackgroundVDM(input_dim, hidden_dim, latent_dim)

# Define the optimizer and loss
optimizer = optim.Adam(vdm.parameters(), lr=0.001)
loss_vdm = VDMLoss()

# Training loop
for epoch in range(num_epochs):
    vdm.train()
    for batch_data, batch_weight in dataloader:
        optimizer.zero_grad()
        reco_data, mu, logvar = vdm(batch_data, time_steps)
        loss = loss_vdm(reco_data, batch_data, batch_weight, mu, logvar)
        loss.backward()
        optimizer.step()
    # if epoch % 10 == 0:
    print(f"===> Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Generate background-subtracted data

num_samples = events.mass[events.weight == 1].to_numpy().shape[0]

vdm.eval()
with torch.no_grad():
    # background_subtracted_data, _, _ = vae(torch.Tensor(X))
    background_subtracted_data = vdm.decode(torch.randn(num_samples, latent_dim)).float()

background_subtracted_data = background_subtracted_data.detach().numpy()


bins = np.linspace(4.0, 9.0, 31)
plt.hist(X[:, 0], bins=bins, weights=weight, alpha=0.5, label="unmixed - mix")
plt.hist(background_subtracted_data[:, 0], bins=bins, label="diffusion", histtype="step", color="red")
plt.xlabel("mass [GeV]")
plt.legend(frameon=False)
plt.show()

bins = np.linspace(0.0, 3.0, 31)
plt.hist(X[:, 1], bins=bins, weights=weight, alpha=0.5, label="unmixed - mix")
plt.hist(background_subtracted_data[:, 1], bins=bins, label="diffusion", histtype="step", color="red")
plt.xlabel("pT [GeV]")
plt.legend(frameon=False)
plt.show()


bins = np.linspace(0.0, 1.0, 31)
plt.hist(X[:, 2], bins=bins, weights=weight, alpha=0.5, label="unmixed - mix")
plt.hist(background_subtracted_data[:, 2], bins=bins, label="diffusion", histtype="step", color="red")
plt.xlabel("xF")
plt.legend(frameon=False)
plt.show()

bins = np.linspace(-np.pi, np.pi, 31)
plt.hist(X[:, 3], bins=bins, weights=weight, alpha=0.5, label="unmixed - mix")
plt.hist(background_subtracted_data[:, 3], bins=bins, label="diffusion", histtype="step", color="red")
plt.xlabel("phi [rad]")
plt.legend(frameon=False)
plt.show()

bins = np.linspace(-0.5, 0.5, 31)
plt.hist(X[:, 4], bins=bins, weights=weight, alpha=0.5, label="unmixed - mix")
plt.hist(background_subtracted_data[:, 4], bins=bins, label="diffusion", histtype="step", color="red")
plt.xlabel("costh")
plt.legend(frameon=False)
plt.show()