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

from BMFUtil import BMFClassifier, AddParams2Input, BMFLoss, BMFLoader
from BMFUtil import weight_fn, reweight_fn, train_model, fit_fn


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
# In this example we extract the lambda, mu, nu parameters from the messy MC data
#

#
# The reweighting formula;
# w = f(x)/(1 - f(x))
#

#
# Load E906 messy MC data
#

# Create train and test data

batch_size = 1024

# lambda0, mu0, nu0 = 1., 0., 0.
#
# data = np.load("BMFData.npy", allow_pickle=True)


def minimizer_fn(lambda_inj, mu_inj, nu_inj, batch_size):

    #
    # Step 1: In this step we try to reweight (1., 0., 0.) to (lambda, mu, nu) with one model for any lambda, mu, nu
    #

    loaders = BMFLoader(lambda_inj, mu_inj, nu_inj, batch_size)

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Create the model
    fit_model = BMFClassifier(input_dim=8, hidden_dim=64)

    # Define the loss function and optimizer
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

    data_loader = loaders["val"]

    # Load the best model weights
    fit_model.load_state_dict(best_model_weights)

    # Define the parameters
    mu_fit_init = [np.random.uniform(0.5, 1.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0]]

    # Create the AddParams2Input layer
    add_params_layer = AddParams2Input(mu_fit_init)

    # Set all weights in fit model to non-trainable
    for param in fit_model.parameters():
        param.requires_grad = False

    # Define the loss function and optimizer
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

    return fit_vals


lambda_fit, mu_fit, nu_fit = [], [], []

# Injected values
lambda_inj, mu_inj, nu_inj = 1.33, 0.17, -0.34

for i in range(50):
    print("starting run : {}".format(i+1))
    fit_vals = minimizer_fn(lambda_inj, mu_inj, nu_inj, batch_size)
    lambda_fit.append(fit_vals["lambda"][-1])
    mu_fit.append(fit_vals["mu"][-1])
    nu_fit.append(fit_vals["nu"][-1])

# print("lambda injected = {:.4f}, lambda fit = {:.4f} +/- {:.4f}".format(lambda_inj, np.mean(lambda_fit), np.std(lambda_fit)))
# print("mu injected = {:.4f}, mu fit = {:.4f} +/- {:.4f}".format(mu_inj, np.mean(mu_fit), np.std(mu_fit)))
# print("nu injected = {:.4f}, nu fit = {:.4f} +/- {:.4f}".format(nu_inj, np.mean(nu_fit), np.std(nu_fit)))

bins = np.linspace(0.0, 2.0, 31)
plt.hist(lambda_fit, bins=bins, histtype='step')
plt.text(0.7, 0.7, "injected = {:.4f}, fit = {:.4f} +/- {:.4f}".format(lambda_inj, np.mean(lambda_fit), np.std(lambda_fit)))
plt.savefig("imgs/lambda_fit.png")
plt.close("all")


bins = np.linspace(0.0, 2.0, 31)
plt.hist(mu_fit, bins=bins, histtype='step')
plt.text(0.7, 0.7, "injected = {:.4f}, fit = {:.4f} +/- {:.4f}".format(mu_inj, np.mean(mu_fit), np.std(mu_fit)))
plt.savefig("imgs/mu_fit.png")
plt.close("all")

bins = np.linspace(0.0, 2.0, 31)
plt.hist(nu_fit, bins=bins, histtype='step')
plt.text(0.7, 0.7, "injected = {:.4f}, fit = {:.4f} +/- {:.4f}".format(nu_inj, np.mean(nu_fit), np.std(nu_fit)))
plt.savefig("imgs/nu_fit.png")
plt.close("all")