#
# dinupa3@gmail.com
# 09-05-2023
#

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from ParamVAEModels import ParamVAE, ParamVAELoss, ParamOptimizer
from ParamVAEModels import train_param_model, train_optimizer, scan_fn

from sklearn.model_selection import train_test_split

import uproot
import awkward as ak


plt.rc('font', size=14)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data
n_data_points = 10**5

batch_size = 1024

mu_min = -2.
mu_max = 2.
mu_vals = mu_values = np.random.uniform(mu_min, mu_max, n_data_points)

X = np.array([(np.random.normal(mu, 1)) for mu in mu_values])

X_train, X_test, param_train, param_test = train_test_split(X, mu_vals, test_size=0.4, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().reshape(-1, 1)
param_train_tensor = torch.from_numpy(param_train).float().reshape(-1, 1)

X_test_tensor = torch.from_numpy(X_test).float().reshape(-1, 1)
param_test_tensor = torch.from_numpy(param_test).float().reshape(-1, 1)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, param_train_tensor)
test_dataset = TensorDataset(X_test_tensor, param_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 200
learning_rate = 0.001
early_stopping_patience = 20
latent_dim = 16
hidden_dim = 32
gamma = 0.1
step_size = 10

# Create the model
param_vae = ParamVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
criterion = ParamVAELoss()
optimizer = optim.Adam(param_vae.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in param_vae.parameters() if p.requires_grad)
print(param_vae)
print('total trainable params: {}'.format(total_trainable_params))

best_model_weights = train_param_model(param_vae, train_loader, test_loader, criterion, optimizer, device, num_epochs, early_stopping_patience, scheduler)

# Load the best model weights
param_vae.load_state_dict(best_model_weights)

# Set all weights in fit model to non-trainable
for param in param_vae.parameters():
    param.requires_grad = False

# Model summary
# print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in param_vae.parameters() if p.requires_grad)
print(param_vae)
print('total trainable params: {}'.format(total_trainable_params))

mu_init = [0.0]

param_optimizer = ParamOptimizer(mu_init)
criterion = nn.MSELoss()
optimizer = optim.Adam(param_optimizer.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Model summary
# print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in param_optimizer.parameters() if p.requires_grad)
print(param_optimizer)
print('total trainable params: {}'.format(total_trainable_params))

mu_secret = 1.5
X_mystery = np.random.normal(mu_secret, 1, n_data_points)

X_mystery_tensor = torch.from_numpy(X_mystery).float().reshape(-1, 1)
dataset = TensorDataset(X_mystery_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Plot the generated events
param_vae.eval()
with torch.no_grad():
    z = torch.randn(n_data_points, latent_dim)
    params = torch.full((n_data_points, 1), mu_secret)
    reco_x = param_vae.decode(z, params).float().ravel().detach().numpy()

# bins = np.linspace(-5, 6, 31)
# plt.hist(X_mystery, bins=bins, label="True events", alpha=0.5)
# plt.hist(reco_x, bins=bins, label="Generated events", histtype="step", color="red")
# plt.legend(frameon=True)
# plt.show()

print("True mean = {:.4f} std = {:.4f}".format(np.mean(X_mystery), np.std(X_mystery)))
print("Reco mean = {:.4f} std = {:.4f}".format(np.mean(reco_x), np.std(reco_x)))


scan_fn(param_vae, dataloader, criterion, device, latent_dim)

train_optimizer(param_optimizer, param_vae, dataloader, criterion, optimizer, device, num_epochs, scheduler, latent_dim)