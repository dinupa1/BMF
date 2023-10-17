import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from DenoisingAEModels import DenoisingAE
from DenoisingAEModels import fit_denoising_ae, denoise_reco_hist

from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)


latent_dim = 32
batch_size = 64
learning_rate = 0.001
num_epochs = 100

model = DenoisingAE(latent_dim)

total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("total trainable params: {}".format(total_trainable_params))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_tree = uproot.open("vae.root:train_tree")
val_tree = uproot.open("vae.root:val_tree")

fit_denoising_ae(train_tree, batch_size, model, criterion, optimizer, num_epochs, device)

denoise_reco_hist(model, val_tree)
