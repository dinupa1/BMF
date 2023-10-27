import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from DenoisingUNetModels import DenoisingUNet
from DenoisingUNetModels import fit_denoising_unet, denoise_reco_hist

from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)

batch_size = 1024
learning_rate = 0.0001
num_epochs = 100

model = DenoisingUNet()

print("===> using decvice {}".format(device))
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("total trainable params: {}".format(total_trainable_params))

# criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_file = uproot.open("unet.root")

train_tree = data_file["train_tree"]
val_tree = data_file["val_tree"]

fit_denoising_unet(train_tree, batch_size, model, criterion, optimizer, num_epochs, device)

denoise_reco_hist(model, val_tree, device)
