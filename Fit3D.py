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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from BMFUtil import BMFClassifier, AddParams2Input,BMFLoss
from BMFUtil import weight_fn, reweight_fn, train_model, fit_fn


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# In this example we extract the lambda, mu, nu parameters from the messy MC data
#

#
# Step 1: In this step we try to reweight (lambda, mu, nu) = (0., 0., 0.) to (lambda, mu, nu) = (0.2, 0.2, 0.2) distributions.
# We train the classifier f(phi, costh) (binary classification) using the samples drawn from these 2 distributions.
# The reweighting formula;
# w = f(x)/(1 - f(x))
#

#
# Load E906 messy MC data
#

# Create train and test data

batch_size = 1024

lambda0, mu0, nu0 = 1., 0., 0.
lambda1, mu1, nu1 = 0.8, 0.1, 0.2

data = np.load("BMFData.npy", allow_pickle=True)

X0 = [(phi, costh) for phi, costh in data[["phi", "costh"]][:1000000]]
X1 = [(phi, costh) for phi, costh in data[["phi", "costh"]][1000000:2000000]]

Y0 = np.zeros(1000000)
Y1 = np.ones(1000000)

weight0 = [(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][:1000000]]
weight1 = [(weight_fn(lambda1, mu1, nu1, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][1000000:2000000]]

X = np.concatenate((X0, X1))
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)
weights = np.concatenate((weight0, weight1)).reshape(-1, 1)

X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size=0.3, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
weights_train_tensor = torch.from_numpy(weights_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).float()
weights_test_tensor = torch.from_numpy(weights_test).float()

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, weights_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor, weights_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Create the model
model = BMFClassifier(input_dim=2, hidden_dim=64)


# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print("total trainable params: {}".format(total_trainable_params))

# Training loop
epochs = 200
early_stopping_patience = 20

# Define the loss function and optimizer
# criterion = nn.BCELoss(weight=None)
criterion = BMFLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move the model to GPU if available
model = model.to(device=device)

best_model_weights = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

# Validation
X0_val = np.array([(phi, costh) for phi, costh in data[["phi", "costh"]][2000000:2015000]])
X1_val = np.array([(phi, costh) for phi, costh in data[["phi", "costh"]][2015000:2030000]])

weight0_val = np.array([(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][2000000:2015000]])
weight1_val = np.array([(weight_fn(lambda1, mu1, nu1, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][2015000:2030000]])

# Load the best model weights
model.load_state_dict(best_model_weights)

weights = reweight_fn(model, X0_val)

bins = np.linspace(-np.pi, np.pi, 31)
# plt.hist(X0_val[:, 0], bins=bins, label=r"$(\lambda, \mu, \nu) = (1., 0., 0.)$", weights=weight0_val, histtype="step", color="r")
plt.hist(X0_val[:, 0], bins=bins, label=r"$(\lambda, \mu, \nu) = (1., 0., 0.)$ weighted", weights=weights*weight0_val, histtype="step", color="k")
plt.hist(X1_val[:, 0], bins=bins, label=r"$(\lambda, \mu, \nu) = (0.8, 0.1, 0.2)$", weights=weight1_val, histtype="step", color="b")
plt.legend(frameon=False)
plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("Counts")
# plt.savefig("imgs/reweight1.png")
plt.show()

bins = np.linspace(-0.5, 0.5, 31)
# plt.hist(X0_val[:, 1], bins=bins, label=r"$(\lambda, \mu, \nu) = (1., 0., 0.)$", weights=weight0_val, histtype="step", color="r")
plt.hist(X0_val[:, 1], bins=bins, label=r"$(\lambda, \mu, \nu) = (1., 0., 0.)$ weighted", weights=weights*weight0_val, histtype="step", color="k")
plt.hist(X1_val[:, 1], bins=bins, label=r"$(\lambda, \mu, \nu) = (0.8, 0.1, 0.2)$", weights=weight1_val, histtype="step", color="b")
plt.legend(frameon=False)
plt.xlabel(r"$\cos\theta$")
plt.ylabel("Counts")
# plt.savefig("imgs/reweight1.png")
plt.show()


#
# Step 2: In this step we try to reweight (0., 0., 0.) to (lambda, mu, nu) with one model for any lambda, mu, nu
#

# Sample lambda, mu, nu values in the range (0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5)
lambda_vals = np.random.uniform(0.5, 1.5, 1000000)
mu_vals = np.random.uniform(-0.5, 0.5, 1000000)
nu_vals = np.random.uniform(-0.5, 0.5, 1000000)

X0 = [(phi, costh, theta0, theta1, theta2) for phi, costh, theta0, theta1, theta2 in zip(data["phi"][:1000000], data["costh"][:1000000], lambda_vals, mu_vals, nu_vals)]
X1 = [(phi, costh, theta0, theta1, theta2) for phi, costh, theta0, theta1, theta2 in zip(data["phi"][1000000:2000000], data["costh"][1000000:2000000], lambda_vals, mu_vals, nu_vals)]

Y0 = np.zeros(1000000)
Y1 = np.ones(1000000)

weight0 = [(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][:1000000]]
weight1 = [(weight_fn(theta0, theta1, theta2, phi, costh)) for theta0, theta1, theta2, phi, costh, in zip(lambda_vals, mu_vals, nu_vals, data["true_phi"][1000000:2000000], data["true_costh"][1000000:2000000])]


X = np.concatenate((X0, X1))
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)
weights = np.concatenate((weight0, weight1)).reshape(-1, 1)

X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size=0.3, shuffle=True)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
weights_train_tensor = torch.from_numpy(weights_train).float()

X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).float()
weights_test_tensor = torch.from_numpy(weights_test).float()

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, weights_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor, weights_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
fit_model = BMFClassifier(input_dim=5, hidden_dim=64)

# Define the loss function and optimizer
# criterion = nn.BCELoss(weight=None)
criterion = BMFLoss()
optimizer = optim.Adam(fit_model.parameters(), lr=0.001)

# Move the model to GPU if available
fit_model = fit_model.to(device=device)

# Model summary
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in fit_model.parameters() if p.requires_grad)
print(fit_model)
print('total trainable params: {}'.format(total_trainable_params))

# Training loop
epochs = 200
early_stopping_patience = 20

best_model_weights = train_model(fit_model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

# Validation
lambda_val, mu_val, nu_val = 1.2, -0.1, -0.2

X0_val = np.array([(phi, costh) for phi, costh in data[["phi", "costh"]][2000000:2015000]])
X1_val = np.array([(phi, costh) for phi, costh in data[["phi", "costh"]][2015000:2030000]])

X_input = np.array([(phi, costh, lambda_val, mu_val, nu_val) for phi, costh in X0_val])

weight0_val = np.array([(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][2000000:2015000]])
weight1_val = np.array([(weight_fn(lambda_val, mu_val, nu_val, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][2015000:2030000]])

# Load the best model weights
fit_model.load_state_dict(best_model_weights)

weights = reweight_fn(fit_model, X_input)

bins = np.linspace(-np.pi, np.pi, 31)
# plt.hist(X0_val[:, 0], bins=bins, label=r'$(\lambda, \mu, \nu) = (1., 0., 0.)$', weights=weight0_val, histtype='step', color='r')
plt.hist(X0_val[:, 0], bins=bins, label=r'$(\lambda, \mu, \nu) = (1., 0., 0.)$ weighted', weights=weights*weight0_val, histtype='step', color='k')
plt.hist(X1_val[:, 0], bins=bins, label=r'$(\lambda, \mu, \nu) = (1.2, -0.1, -0.2)$', weights=weight1_val, histtype='step', color='b')
plt.legend(frameon=False)
plt.xlabel(r"$\phi$ [rad]")
plt.ylabel("Counts")
# plt.savefig("imgs/reweight1.png")
plt.show()

bins = np.linspace(-0.5, 0.5, 31)
# plt.hist(X0_val[:, 1], bins=bins, label=r'$(\lambda, \mu, \nu) = (1., 0., 0.)$', weights=weight0_val, histtype='step', color='r')
plt.hist(X0_val[:, 1], bins=bins, label=r'$(\lambda, \mu, \nu) = (1., 0., 0.)$ weighted', weights=weights*weight0_val, histtype='step', color='k')
plt.hist(X1_val[:, 1], bins=bins, label=r'$(\lambda, \mu, \nu) = (1.2, -0.1, -0.2)$', weights=weight1_val, histtype='step', color='b')
plt.legend(frameon=False)
plt.xlabel(r"$\cos\theta$")
plt.ylabel("Counts")
# plt.savefig("imgs/reweight1.png")
plt.show()


# Step 3: In this step we find the unknown parameter by gradient decent algorithm


# Create data
lambda_secret, mu_secret, nu_secret = 0.9, 0.1, 0.2

X_mystery = np.array([(phi, costh) for phi, costh in data[["phi", "costh"]][2015000:2030000]])

Y0 = np.zeros(15000)
Y1 = np.ones(15000)

weight0 = [(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][:15000]]
weight_mystry = [(weight_fn(lambda_secret, mu_secret, nu_secret, phi, costh)) for phi, costh in data[["true_phi", "true_costh"]][2015000:2030000]]

X = np.concatenate((np.array(X0)[:15000, :2], X_mystery))
Y = np.concatenate((Y0, Y1)).reshape(-1, 1)
weights = np.concatenate((weight0, weight_mystry)).reshape(-1, 1)

# Create PyTorch datasets and dataloaders
dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).float(), torch.Tensor(weights))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the best model weights
fit_model.load_state_dict(best_model_weights)

# Define the parameters
mu_fit_init = [1., 0., 0.]

# Create the AddParams2Input layer
add_params_layer = AddParams2Input(mu_fit_init)

# Set all weights in fit model to non-trainable
for param in fit_model.parameters():
    param.requires_grad = False

# Define the loss function and optimizer
# loss_fn = nn.BCELoss(weight=None)
loss_fn = BMFLoss()
optimizer = torch.optim.Adam(add_params_layer.parameters(), lr=0.001)

# Transfer models to GPU
add_params_layer = add_params_layer.to(device)
fit_model = fit_model.to(device)

# Model summary
print("using device : {}".format(device))
fit_trainable_params = sum(p.numel() for p in fit_model.parameters() if p.requires_grad)
print(fit_model)
print("total trainable params in fit model: {}".format(fit_trainable_params))

total_trainable_params = sum(p.numel() for p in add_params_layer.parameters() if p.requires_grad)
print(add_params_layer)
print("total trainable params in fit model: {}".format(total_trainable_params))

# Fit vals
epochs = 200

losses, fit_vals = fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn)


# Plot the loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("loss")
# plt.legend(frameon=False)
# plt.savefig("imgs/fit_net.png")
plt.show()

# Plot the fitted values
plt.plot(np.array(fit_vals["lambda"])/lambda_secret, label=r"$\lambda$", color="r")
plt.plot(np.array(fit_vals["mu"])/mu_secret, label=r"$\mu$", color="b")
plt.plot(np.array(fit_vals["nu"])/nu_secret, label=r"$\nu$", color="g")
# plt.hlines(lambda_secret, 0, len(fit_vals["lambda"]), label="Truth", color="b")
plt.xlabel("Epochs")
plt.ylabel("Fit value/Injected value")
plt.legend(frameon=False)
# plt.savefig("imgs/fit_net.png")
plt.show()


# plt.plot(fit_vals["mu"], label=r"$\mu_{fit}$", color="r")
# plt.hlines(mu_secret, 0, len(fit_vals["mu"]), label="Truth", color="b")
# plt.xlabel("Epochs")
# plt.ylabel(r"$\mu_{fit}$")
# plt.legend(frameon=False)
# # plt.savefig("imgs/fit_net.png")
# plt.show()
#
#
# plt.plot(fit_vals["nu"], label=r"$\nu_{fit}$", color="r")
# plt.hlines(nu_secret, 0, len(fit_vals["nu"]), label="Truth", color="b")
# plt.xlabel("Epochs")
# plt.ylabel(r"$\nu_{fit}$")
# plt.legend(frameon=False)
# # plt.savefig("imgs/fit_net.png")
# plt.show()