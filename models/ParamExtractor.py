import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import ParamExtractor

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)

latent_size = 32
batch_size = 1024
learning_rate = 0.0001
num_epochs = 200
step_size = 50
gamma = 0.1


tree = torch.load("unet-tensor.pt")

train_tree = tree["train_tree"]
val_tree = tree["val_tree"]
test_tree = tree["test_tree"]

model = ParamExtractor(latent_size, learning_rate, step_size, gamma)
model.train(train_tree, val_tree, batch_size, num_epochs, device)
model.prediction(test_tree, batch_size)