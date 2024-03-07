import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from utils import data_loaders, train_model, reweight_fn, update_data
from models import ReweightModel, ReweightLoss

plt.rc("font", size=14)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

hdf="../e906-LH2-data/reweight.hdf5"

early_stopping_patience = 5,0
epochs = 500

train_loader, val_loader, X_test_tensor = data_loaders(hdf, batch_size=500)

model = ReweightModel(input_size=5, hidden_size=20, output_size=1)


criterion = ReweightLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_model_weights = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, early_stopping_patience)

model.load_state_dict(best_model_weights)

weights = reweight_fn(model, X_test_tensor)

print(weights)

update_data(hdf, weights)
