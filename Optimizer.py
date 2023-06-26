import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import uproot
import awkward as ak

from Model import CNNData, FitData, CNNClassifier, AddParams2Input
from Model import weight_fn, train_model, fit_fn

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LAMBDA_val, MU_val, NU_val = 0.92, -0.12, 0.34
LAMBDA0, MU0, NU0 = 1., 0., 0.

# Constants
iterations = 1
epochs = 2
learning_rate = 0.001
n_samples = 100000
n_events = 10000
n_val = 20000
batch_size = 1000
early_stopping_patience = 5


opt_vals = {
    "lambda": [],
    "mu": [],
    "nu": [],
}


# Load data
data0 = uproot.open("BinMCData.root:X0_train")
data1 = uproot.open("BinMCData.root:X1_train")
data1_val = uproot.open("BinMCData.root:X1_val")


X0 = data0["hist"].array(library="np")
THETA0 = data0["theta"].array(library="np")

X1 = data1["hist"].array(library="np")
THETA1 = data1["theta"].array(library="np")

X1_val = data1_val["hist"].array(library="np")
THETA1_val = data1_val["theta"].array(library="np")

Y0 = np.zeros(n_samples).reshape(-1, 1)
Y1 = np.ones(n_samples).reshape(-1, 1)

X = np.concatenate((X0, X1))
Y = np.concatenate((Y0, Y1))
THETA = np.concatenate((THETA0, THETA1))


for i in range(iterations):
    print("Iteration {}".format(i+1))

    X_train, X_test, THETA_train, THETA_test, Y_train, Y_test = train_test_split(X, THETA, Y, test_size=0.3, shuffle=True)

    # Data loaders
    train_dataset = CNNData(X_train, THETA_train, Y_train)
    test_dataset = CNNData(X_test, THETA_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model
    fit_model = CNNClassifier()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(fit_model.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    fit_model = fit_model.to(device=device)

    # Model summary
    # print("*** model summary ***")
    # print("using device : {}".format(device))
    # total_trainable_params = sum(p.numel() for p in fit_model.parameters() if p.requires_grad)
    # print(fit_model)
    # print('total trainable params: {}'.format(total_trainable_params))

    # Validation data
    best_model_weights = train_model(fit_model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

    X0_train1, X0_val = train_test_split(X0, test_size=n_val, shuffle=True)

    Y0_val = np.zeros(n_val).reshape(-1, 1)
    Y1_val = np.ones(n_val).reshape(-1, 1)

    X_val = np.concatenate((X0_val, X1_val))
    Y_val = np.concatenate((Y0_val, Y1_val))

    dataset = FitData(X_val, Y_val)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load the best model weights
    fit_model.load_state_dict(best_model_weights)

    # Define the parameters
    theta_fit_init = [np.random.uniform(0.5, 1.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0]]

    # Create the AddParams2Input layer
    add_params_layer = AddParams2Input(theta_fit_init)

    # Set all weights in fit model to non-trainable
    for param in fit_model.parameters():
        param.requires_grad = False

    # Model summary
    # print("*** model summary ***")
    # print("using device : {}".format(device))
    # total_trainable_params = sum(p.numel() for p in add_params_layer.parameters() if p.requires_grad)
    # print(fit_model)
    # print('total trainable params: {}'.format(total_trainable_params))

    # Define the loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(add_params_layer.parameters(), lr=learning_rate)

    # Transfer models to GPU
    add_params_layer = add_params_layer.to(device)
    fit_model = fit_model.to(device)

    fit_vals = fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn)

    opt_vals["lambda"].append(fit_vals["lambda"][-1])
    opt_vals["mu"].append(fit_vals["mu"][-1])
    opt_vals["nu"].append(fit_vals["nu"][-1])


print("lambda test : {:.3f} lambda opt : {:.3f} +/- {:.3f}".format(LAMBDA_val, np.mean(opt_vals["lambda"]), np.std(opt_vals["lambda"])))
print("mu test : {:.3f} mu opt : {:.3f} +/- {:.3f}".format(MU_val, np.mean(opt_vals["mu"]), np.std(opt_vals["mu"])))
print("nu test : {:.3f} nu opt : {:.3f} +/- {:.3f}".format(NU_val, np.mean(opt_vals["nu"]), np.std(opt_vals["nu"])))