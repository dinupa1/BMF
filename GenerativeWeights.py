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

from GenerativeWeightsModels import GaussianVAE, GaussianVAELoss, Classifier
from GenerativeWeightsModels import train_vae, train_classifier

from sklearn.model_selection import train_test_split

import uproot
import awkward as ak


plt.rc('font', size=14)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data
n_data_points = 10**5

batch_size = 1000

mu0 = 0.
mu1 = 1.
sigma0 = 1.
sigma1 = 1.

X_real = np.random.normal(mu0, sigma0, n_data_points)
Y_real = np.ones(n_data_points)

X_train, X_test, Y_train, Y_test = train_test_split(X_real, Y_real, test_size=0.4, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().reshape(-1, 1)
Y_train_tensor = torch.from_numpy(Y_train).float().reshape(-1, 1)

X_test_tensor = torch.from_numpy(X_test).float().reshape(-1, 1)
Y_test_tensor = torch.from_numpy(Y_test).float().reshape(-1, 1)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 1000
learning_rate = 0.001
early_stopping_patience = 10

# Create the model
gaussian_vae = GaussianVAE()
criterion = GaussianVAELoss()
optimizer = optim.Adam(gaussian_vae.parameters(), lr=learning_rate)

# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in gaussian_vae.parameters() if p.requires_grad)
print(gaussian_vae)
print('total trainable params: {}'.format(total_trainable_params))

best_model_weights = train_vae(gaussian_vae, train_loader, test_loader, criterion, optimizer, device, num_epochs, early_stopping_patience)

# Load the best model weights
gaussian_vae.load_state_dict(best_model_weights)

# Plot the generated events
gaussian_vae.eval()
with torch.no_grad():
    X_fake = gaussian_vae.decode(torch.randn(n_data_points, 5)).float().ravel()

X_real = np.random.normal(mu0, sigma0, n_data_points)
X_fake = X_fake.detach().numpy()

bins = np.linspace(-5, 5, 31)
plt.hist(X_real, bins=bins, label="True events", histtype='step', color='red')
plt.hist(X_fake, bins=bins, label="Generated events", histtype='step', color='blue')
plt.xlabel("x [a.u.]")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.show()

# Fix the generated events with reweighting
Y_real = np.ones(n_data_points)
Y_fake = np.zeros(n_data_points)

X = np.concatenate((X_real, X_fake)).reshape(-1, 1)
Y = np.concatenate((Y_real, Y_fake)).reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float().reshape(-1, 1)
Y_train_tensor = torch.from_numpy(Y_train).float().reshape(-1, 1)

X_test_tensor = torch.from_numpy(X_test).float().reshape(-1, 1)
Y_test_tensor = torch.from_numpy(Y_test).float().reshape(-1, 1)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
classifier = Classifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

best_model_weights = train_classifier(classifier, train_loader, test_loader, criterion, optimizer, device, num_epochs, early_stopping_patience)

# Load the best model weights
classifier.load_state_dict(best_model_weights)

# Plot the generated events
gaussian_vae.eval()
with torch.no_grad():
    X_fake_val = gaussian_vae.decode(torch.randn(n_data_points, 5)).float().ravel()


X_fake_val = X_fake_val.detach().numpy()
X_real_val = np.random.normal(mu0, sigma0, n_data_points)

classifier.eval()
with torch.no_grad():
    preds = classifier(torch.Tensor(X_fake_val).reshape(-1, 1)).detach().numpy().ravel()
    weights = preds / (1.0 - preds)


# Plost the weighted distributions
bins = np.linspace(-5., 5., 31)
plt.hist(X_real_val, bins=bins, label="True events", alpha=0.5)
plt.hist(X_fake_val, bins=bins, label="Generated events", histtype='step', color='blue')
plt.hist(X_fake_val, bins=bins, weights=weights, label="Weighted generated events", histtype='step', color='red')
plt.xlabel("x [a.u.]")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.show()