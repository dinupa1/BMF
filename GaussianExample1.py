import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from GaussianVAEModels import GaussianVAE, GaussianLoss

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # for m1 macs

# create data
n_data_points = 10**6
mu_min = -2.
mu_max = 2.
mu_values = np.random.uniform(mu_min, mu_max, n_data_points)

X_data = np.array([(np.random.normal(mu, 1), mu) for mu in mu_values])

learning_rate = 0.0001
latent_dim = 30
batch_size = 1000
num_epochs = 200

model = GaussianVAE(latent_dim)

print(model)

criterion = GaussianLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_model_weights = model.fit(X_data, batch_size, criterion, optimizer, num_epochs, device, plotson=True)

# print(best_model_weights)