import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from UNetModels import UNet
from UNetModels import fit_unet, unet_prediction

from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)

batch_size = 1024
learning_rate = 0.0001
num_epochs = 1000

model = UNet()

print("===> using decvice {}".format(device))
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("total trainable params: {}".format(total_trainable_params))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

tree = torch.load("unet-tensor.pt")

train_tree = tree["train_tree"]
val_tree = tree["val_tree"]
test_tree = tree["test_tree"]

fit_unet(train_tree, val_tree, batch_size, model, criterion, optimizer, num_epochs, device)

unet_prediction(model, test_tree, device)