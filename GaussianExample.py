#
# Gaussian example for reweighting and fitting
# dinupa3@gmail.com
# 05-26-2023
#


#
# Imports
#
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from GaussUtil import GaussClassifier, AddParams2Input
from GaussUtil import train_model, reweight_fn, fit_fn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# In this example we try to understand the algorithm using 1D gaussian distributions.
#

#
# Step 1: In this step we try to reweight N(0,1) to N(1, 1.3) distributions.
# We train the classifier f (binary classification) using the samples drawn from these 2 distributions.
# The reweighting formula;
# w = f(x)/(1 - f(x))
#

#
# Create data and data loaders
#

n_data_points = 10**5

batch_size = 1000

mu0 = 0
mu1 = 1
var0 = 1
var1 = 1.3

X0 = np.random.normal(mu0, var0, n_data_points)
X1 = np.random.normal(mu1, var1, n_data_points)

Y0 = np.zeros(n_data_points)
Y1 = np.ones(n_data_points)

X = np.concatenate((X0, X1)).reshape(-1, 1)
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).float()


# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
model = GaussClassifier(input_dim=1, hidden_dim=20)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
model = model.to(device=device)


# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print('total trainable params: {}'.format(total_trainable_params))

# Training loop
epochs = 100
early_stopping_patience = 10
best_model_weights = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

# Validate the trained classifier

X0_val = np.random.normal(mu0, var0, n_data_points)
X1_val = np.random.normal(mu1, var1, n_data_points)

# Load the best model weights
model.load_state_dict(best_model_weights)

weights = reweight_fn(model, X0_val.reshape(-1, 1))

bins = np.linspace(-6, 5, 31)
plt.hist(X0_val, bins=bins, alpha=0.5, label=r'$\mu=0$')
plt.hist(X0_val, bins=bins, label=r'$\mu=0$ weighted', weights=weights, histtype='step', color='k')
plt.hist(X1_val, bins=bins, alpha=0.5, label=r'$\mu=1$')
plt.legend(frameon=False)
# plt.savefig("imgs/reweight1.png")
plt.show()

#
# Step 2: In this step we try to reweight N(0,1) to N(mu, 1) with one model for any mu.
# We uniformly sample mu in some range.
# We will now parametrize our network by giving it a value in addition to x in N(mu, 1).
#


# Create data
n_data_points = 10**5
mu_min = -2
mu_max = 2
mu_values = np.random.uniform(mu_min, mu_max, n_data_points)

X0 = [(np.random.normal(0, 1), mu) for mu in mu_values] # Note the zero in normal(0, 1)
X1 = [(np.random.normal(mu, 1), mu) for mu in mu_values]

Y0 = np.zeros(n_data_points)
Y1 = np.ones(n_data_points)

X = np.concatenate((X0, X1))
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).float()

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


fit_model = GaussClassifier(input_dim=2, hidden_dim=50)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(fit_model.parameters(), lr=0.001)

# Move the model to GPU if available
fit_model = fit_model.to(device)

# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in fit_model.parameters() if p.requires_grad)
print(fit_model)
print('total trainable params: {}'.format(total_trainable_params))

# Training loop
epochs = 200
early_stopping_patience = 10
best_model_weights = train_model(fit_model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

# Validate

mu1 = -1.5
assert mu_min <= mu1 <= mu_max  # choose mu1 in valid range

X0_val = np.random.normal(0, 1, n_data_points)
X1_val = np.random.normal(mu1, 1, n_data_points)

X_input = np.array([(x, mu1) for x in X0_val])

# Load the best model weights
fit_model.load_state_dict(best_model_weights)

weights = reweight_fn(fit_model, X_input)

bins = np.linspace(-6, 5, 31)
plt.hist(X0_val, bins=bins, alpha=0.5, label=r'$\mu=0$')
plt.hist(X0_val, bins=bins, label=r'$\mu=0$ wgt.', weights=weights, histtype='step', color='k')
plt.hist(X1_val, bins=bins, alpha=0.5, label=r'$\mu={}$'.format(mu1))
plt.legend(frameon=False)
# plt.savefig("imgs/reweight2.png")
plt.show()


# Step 3: In this step we find the unknown parameter by gradient decent algorithm


# Create data
mu_secret = 1.3
X_mystery = np.random.normal(mu_secret, 1, n_data_points)


Y0 = np.zeros(n_data_points)
Y1 = np.ones(n_data_points)

X = np.concatenate((np.array(X0)[:,0], X_mystery)).reshape(-1, 1)
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)

# Create PyTorch datasets and dataloaders
dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).float())

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the best model weights
fit_model.load_state_dict(best_model_weights)

# Define the parameters
mu_fit_init = [0.]

# Create the AddParams2Input layer
add_params_layer = AddParams2Input(mu_fit_init)

# Set all weights in fit model to non-trainable
for param in fit_model.parameters():
    param.requires_grad = False

# Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(add_params_layer.parameters(), lr=0.001)

# Transfer models to GPU
add_params_layer = add_params_layer.to(device)
fit_model = fit_model.to(device)

# Model summary
print("using device : {}".format(device))
fit_trainable_params = sum(p.numel() for p in fit_model.parameters() if p.requires_grad)
print(fit_model)
print('total trainable params in fit model: {}'.format(fit_trainable_params))

total_trainable_params = sum(p.numel() for p in add_params_layer.parameters() if p.requires_grad)
print(add_params_layer)
print('total trainable params in fit model: {}'.format(total_trainable_params))

# Training loop
epochs = 50
losses, fit_vals = fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn)

plt.plot(fit_vals, label='Fit', color='r')
plt.hlines(mu_secret, 0, len(fit_vals), label='Truth')
plt.xlabel("Epochs")
plt.ylabel(r'$\mu_{fit}$')
plt.legend(frameon=False)
# plt.savefig("imgs/fit_net.png")
plt.show()