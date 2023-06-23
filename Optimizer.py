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

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to calculate weights analytically
def weight_fn(xx1, xx2, xx3, phi, costh):
    weight = 1. + xx1 * costh * costh + 2. * xx2 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * xx3 * (1. - costh * costh) * np.cos(2. * phi)
    return weight / (1. + costh * costh)


# Define classifier
class BMFClassifier(nn.Module):
    def __init__(self, input_dim: int = 67, out_dim: int = 1, hidden_dim: int = 100):
        super(BMFClassifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# Module used to add parameter for fitting
class AddParams2Input(nn.Module):
    def __init__(self, params):
        super(AddParams2Input, self).__init__()
        self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

    def forward(self, inputs):
        batch_params = torch.ones((inputs.size(0), 1), device=inputs.device) * self.params.to(device=inputs.device)
        concatenated = torch.cat([inputs, batch_params], dim=-1)
        return concatenated


# Training step
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience):
    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # Train step
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)

                running_loss += loss.item() * batch_inputs.size(0)

            validation_loss = running_loss / len(test_loader.dataset)

            print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}".format(epoch + 1, epoch_loss, validation_loss))

            # Check for early stopping
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    return best_model_weights


# Fit the model
def fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn):
    fit_vals = {
        "loss": [],
        "lambda": [],
        "mu": [],
        "nu": []
    }

    for epoch in range(epochs):
        add_params_layer.train()
        running_loss = 0.0
        for batch_inputs, batch_labels in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            param_input = add_params_layer(batch_inputs)
            output = fit_model(param_input)

            # Compute the loss
            loss = loss_fn(output, batch_labels)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        print("epoch : {}, loss = {:.4f}, lambda = {:.4f}, mu = {:.4f}, nu = {:.4f}".format(epoch + 1, epoch_loss,
                                                                                            add_params_layer.params[0].item(),
                                                                                            add_params_layer.params[1].item(),
                                                                                            add_params_layer.params[2].item()))
        fit_vals["loss"].append(epoch_loss)
        fit_vals["lambda"].append(add_params_layer.params[0].item())
        fit_vals["mu"].append(add_params_layer.params[1].item())
        fit_vals["nu"].append(add_params_layer.params[2].item())

    return fit_vals


LAMBDA_val, MU_val, NU_val = 0.82, -0.12, 0.31
LAMBDA0, MU0, NU0 = 1., 0., 0.

data0 = uproot.open("BinMCData.root:X0_train_dic")
data1 = uproot.open("BinMCData.root:X1_train_dic")
data_val = uproot.open("BinMCData.root:X_val_dic")

X0_dic = data0.arrays(data0.keys(), library="np")
X1_dic = data1.arrays(data1.keys(), library="np")
X_val_dic = data_val.arrays(data_val.keys(), library="np")

X0 = np.array(list(X0_dic.values())).T
X1 = np.array(list(X1_dic.values())).T
X_val = np.array(list(X_val_dic.values())).T


# Constants
iterations = 20
epochs = 100
learning_rate = 0.001
train_points = 10**4
hist_events = 10000
val_points = 10**4
batch_size = 1000

hist_bins = 8
hist_range = [[-np.pi, np.pi], [-0.5, 0.5]]

early_stopping_patience = 20
opt_vals = {
    "lambda": [],
    "mu": [],
    "nu": [],
}

for i in range(iterations):
    # Define model
    fit_model = BMFClassifier()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(fit_model.parameters(), lr=learning_rate)

    Y0 = np.zeros(train_points)
    Y1 = np.ones(train_points)

    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1)).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, shuffle=True)

    # Convert to torch tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test).float()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    # Data loders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_model_weights = train_model(fit_model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience)

    X0_val, X1_val = [], []

    # Eval dataset
    for j in range(val_points):
        data0_val, data1_val = train_test_split(X_val, test_size=0.5, shuffle=True)

        data_sample = resample(data0_val, replace=False, n_samples=hist_events)
        weights = weight_fn(LAMBDA0, MU0, NU0, data_sample[:, 2], data_sample[:, 3])
        bc = np.histogram2d(data_sample[:, 0], data_sample[:, 1], bins=hist_bins, range=hist_range, density=True, weights=weights)[0]
        X0_val.append(bc.ravel())

        data_sample = resample(data1_val, replace=False, n_samples=hist_events)
        weights = weight_fn(LAMBDA_val, MU_val, NU_val, data_sample[:, 2], data_sample[:, 3])
        bc = np.histogram2d(data_sample[:, 0], data_sample[:, 1], bins=hist_bins, range=hist_range, density=True, weights=weights)[0]
        X1_val.append(bc.ravel())

        if j % 1000 == 0:
            print("Iteration: [{}/{}]".format(j, val_points))

    X0_val = np.array(X0_val)
    X1_val = np.array(X1_val)

    Y0_val = np.zeros(val_points)
    Y1_val = np.ones(val_points)

    X = np.concatenate((X0_val, X1_val))
    Y = np.concatenate((Y0_val, Y1_val)).reshape(-1, 1)

    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load the best model weights
    fit_model.load_state_dict(best_model_weights)

    # Define the parameters
    mu_fit_init = [np.random.uniform(0.5, 1.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0],
                   np.random.uniform(-0.5, 0.5, 1)[0]]

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

    fit_vals = fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn)

    opt_vals["lambda"].append(fit_vals["lambda"][-1])
    opt_vals["mu"].append(fit_vals["mu"][-1])
    opt_vals["nu"].append(fit_vals["nu"][-1])

    print("Iteration {}".format(i+1))


print("lambda test : {:.3f} lambda opt : {:.3f} +/- {:.3f}".format(LAMBDA_val, np.mean(opt_vals["lambda"]), np.std(opt_vals["lambda"])))
print("mu test : {:.3f} mu opt : {:.3f} +/- {:.3f}".format(MU_val, np.mean(opt_vals["mu"]), np.std(opt_vals["mu"])))
print("nu test : {:.3f} nu opt : {:.3f} +/- {:.3f}".format(NU_val, np.mean(opt_vals["nu"]), np.std(opt_vals["nu"])))