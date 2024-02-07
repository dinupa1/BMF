import numpy as np
import matplotlib.pyplot as plt

import torch

from models import NetFit

import h5py

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)


f = h5py.File("net.hdf5", "r")

X0_train = f["X0_train"]
X1_train = f["X1_train"]
X0_test = f["X0_test"]
X1_test = f["X1_test"]

# particle level train data
X0_train_par = {
    "x":X0_train["X_par"][:],
    "y": X0_train["label"][:].reshape(-1, 1),
    "weight": X0_train["W_par"][:, 0].reshape(-1, 1),
    "theta": X0_train["thetas"][:],
    }

X1_train_par = {
    "x":X1_train["X_par"][:],
    "y": X1_train["label"][:].reshape(-1, 1),
    "weight": X1_train["W_par"][:, 0].reshape(-1, 1),
    "theta": X1_train["thetas"][:],
    }


# detetctor level train data
X0_train_det = {
    "x":X0_train["X_det"][:],
    "y": X0_train["label"][:].reshape(-1, 1),
    "weight": X0_train["W_det"][:, 0].reshape(-1, 1),
    "theta": X0_train["thetas"][:],
    }

X1_train_det = {
    "x":X1_train["X_det"][:],
    "y": X1_train["label"][:].reshape(-1, 1),
    "weight": X1_train["W_det"][:, 0].reshape(-1, 1),
    "theta": X1_train["thetas"][:],
    }

# particle level test data
X0_test_par = {
    "x":X0_test["X_par"][:],
    "y": X0_test["label"][:].reshape(-1, 1),
    "weight": X0_test["W_par"][:, 0].reshape(-1, 1),
    "theta": X0_test["thetas"][:],
    "x_err": X0_test["W_par"][:, 1].reshape(-1, 1),
    }

X1_test_par = {
    "x":X1_test["X_par"][:],
    "y": X1_test["label"][:].reshape(-1, 1),
    "weight": X1_test["W_par"][:, 0].reshape(-1, 1),
    "theta": X1_test["thetas"][:],
    "x_err": X1_test["W_par"][:, 1].reshape(-1, 1),
    }

# detector level test data
X0_test_det = {
    "x":X0_test["X_det"][:],
    "y": X0_test["label"][:].reshape(-1, 1),
    "weight": X0_test["W_det"][:, 0].reshape(-1, 1),
    "theta": X0_test["thetas"][:],
    "x_err": X0_test["W_det"][:, 1].reshape(-1, 1),
    }

X1_test_det = {
    "x":X1_test["X_det"][:],
    "y": X1_test["label"][:].reshape(-1, 1),
    "weight": X1_test["W_det"][:, 0].reshape(-1, 1),
    "theta": X1_test["thetas"][:],
    "x_err": X1_test["W_det"][:, 1].reshape(-1, 1),
    }

hidden_dim = 64
batch_size = 1024
learning_rate = 0.001
num_epochs = 500
step_size = 20
early_stopping_patience = 20
gamma = 0.1
num_runs = 50

# particle level fit
model = NetFit(hidden_dim, learning_rate, step_size, gamma, batch_size)
best_model_weights = model.train(X0_train_par, X1_train_par, num_epochs, device, early_stopping_patience)
#model.scan(best_model_weights, X0_test_tree, X1_test_tree)
#model.reweight(best_model_weights, X0_test_tree, X1_test_tree)
model.fit(best_model_weights, X0_test_par, X1_test_par, num_runs, num_epochs)
