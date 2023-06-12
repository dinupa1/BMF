#
# simple models used for fitting
# dinupa3@gmail.com
# 05-26-2023
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from sklearn.model_selection import train_test_split


# Custom loss function
class BMFLoss(nn.Module):
    def __init__(self):
        super(BMFLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        weighted_targets = targets * weights + (1 - targets) * (1 - weights)
        criterion = nn.BCELoss()
        loss = criterion(outputs, weighted_targets)
        return loss


# classifier used for reweighting

class BMFClassifier(nn.Module):
    def __init__(self, input_dim: int = 5, output_dim: int = 1, hidden_dim: int = 50):
        super(BMFClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc6 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.bn6 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.bn3(self.fc3(x))
        x = self.relu(x)
        x = self.bn4(self.fc4(x))
        x = self.relu(x)
        # x = self.relu(self.fc5(x))
        x = self.bn6(self.fc6(x))
        x = self.sigmoid(x)
        return x


# module used to add parameter for fitting

class AddParams2Input(nn.Module):
    def __init__(self, params):
        super(AddParams2Input, self).__init__()
        self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

    def forward(self, inputs):
        batch_params = torch.ones((inputs.size(0), 1), device=inputs.device) * self.params.to(device=inputs.device)
        concatenated = torch.cat([inputs, batch_params], dim=-1)
        return concatenated


# Function to calculate weights analytically
def weight_fn(xx1, xx2, xx3, phi, costh):
    weight = 1. + xx1 * costh * costh + 2. * xx2 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * xx3 * (1. - costh * costh) * np.cos(2. * phi)
    return weight / (1. + costh * costh)


# Training step

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience):
    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # Train step
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_labels, batch_weights in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels, batch_weights)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_inputs, batch_labels, batch_weights in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_weights = batch_weights.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels, batch_weights)

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


# Reweighting function
def reweight_fn(model, X_val):
    # Move the model to CPU for evaluation
    model = model.to(torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        preds = model(torch.Tensor(X_val)).detach().numpy().ravel()
        weights = preds / (1.0 - preds)

    return weights


# Fit the model
def fit_fn(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn):
    losses = []
    fit_vals = {
        "lambda": [],
        "mu": [],
        "nu": []
    }

    for epoch in range(epochs):
        add_params_layer.train()
        running_loss = 0.0
        for batch_inputs, batch_labels, batch_weights in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)

            # Forward pass
            optimizer.zero_grad()
            param_input = add_params_layer(batch_inputs)
            output = fit_model(param_input)

            # Compute the loss
            loss = loss_fn(output, batch_labels, batch_weights)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        print("epoch : {}, loss = {:.4f}, lambda = {:.4f}, mu = {:.4f}, nu = {:.4f}".format(epoch + 1, epoch_loss,
                                                                                            add_params_layer.params[0].item(),
                                                                                            add_params_layer.params[1].item(),
                                                                                            add_params_layer.params[2].item()))
        losses.append(epoch_loss)
        fit_vals["lambda"].append(add_params_layer.params[0].item())
        fit_vals["mu"].append(add_params_layer.params[1].item())
        fit_vals["nu"].append(add_params_layer.params[2].item())

    return losses, fit_vals


# data loaders
def BMFLoader(lambda_inj, mu_inj, nu_inj, batch_size):
    lambda0, mu0, nu0 = 1., 0., 0.

    data = np.load("BMFData.npy", allow_pickle=True)

    # train test split
    data0 = np.array([(mass, pT, xF, phi, costh, true_phi, true_costh) for mass, pT, xF, phi, costh, true_phi, true_costh in data[["mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh"]][:1000000]])
    data1 = np.array([(mass, pT, xF, phi, costh, true_phi, true_costh) for mass, pT, xF, phi, costh, true_phi, true_costh in data[["mass", "pT", "xF","phi", "costh", "true_phi", "true_costh"]][1000000:2000000]])

    # data0_train, data0_val, data1_train, data1_val = train_test_split(data0, data1, test_size=0.5, shuffle=True)

    # Sample lambda, mu, nu values in the range (0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5)
    lambda_vals = np.random.uniform(0.5, 1.5, data1.shape[0])
    mu_vals = np.random.uniform(-0.5, 0.5, data1.shape[0])
    nu_vals = np.random.uniform(-0.5, 0.5, data1.shape[0])

    X0 = [(mass, pT, xF, phi, costh, theta0, theta1, theta2) for mass, pT, xF, phi, costh, theta0, theta1, theta2 in
          zip(data0[:, 0], data0[:, 1], data0[:, 2], data0[:, 3], data0[:, 4], lambda_vals, mu_vals, nu_vals)]
    X1 = [(mass, pT, xF, phi, costh, theta0, theta1, theta2) for mass, pT, xF, phi, costh, theta0, theta1, theta2 in
          zip(data1[:, 0], data1[:, 1], data1[:, 2], data1[:, 3], data1[:, 4], lambda_vals, mu_vals, nu_vals)]

    Y0 = np.zeros(data0.shape[0])
    Y1 = np.ones(data1.shape[0])

    weight0 = [(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in zip(data0[:, 5], data0[:, 6])]
    weight1 = [(weight_fn(theta0, theta1, theta2, phi, costh)) for theta0, theta1, theta2, phi, costh, in
               zip(lambda_vals, mu_vals, nu_vals, data1[:, 5], data1[:, 6])]

    X = np.concatenate((X0, X1))
    Y = np.concatenate((Y0, Y1)).reshape(-1, 1)
    weights = np.concatenate((weight0, weight1)).reshape(-1, 1)

    X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size=0.3,
                                                                                     shuffle=True)

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

    # Create validation data set
    data0_1, data0_2 = train_test_split(data0, test_size=0.015, shuffle=True)
    # data0_2 = data0
    data1_1 = np.array([(mass, pT, xF, phi, costh, true_phi, true_costh) for mass, pT, xF, phi, costh, true_phi, true_costh in data[["mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh"]][2000000:2015000]])

    X0_val = np.array([(mass, pT, xF, phi, costh) for mass, pT, xF, phi, costh in zip(data0_2[:, 0], data0_2[:, 1], data0_2[:, 2], data0_2[:, 3], data0_2[:, 4])])
    X1_val = np.array([(mass, pT, xF, phi, costh) for mass, pT, xF, phi, costh in zip(data1_1[:, 0], data1_1[:, 1], data1_1[:, 2], data1_1[:, 3], data1_1[:, 4])])

    Y0_val = np.zeros(X0_val.shape[0])
    Y1_val = np.ones(X1_val.shape[0])

    weight0_val = [(weight_fn(lambda0, mu0, nu0, phi, costh)) for phi, costh in zip(data0_2[:, 5], data0_2[:, 6])]
    weight1_val = [(weight_fn(lambda_inj, mu_inj, nu_inj, phi, costh)) for phi, costh in zip(data1_1[:, 5],data1_1[:, 6])]

    X = np.concatenate((X0_val, X1_val))
    Y = np.concatenate((Y0_val, Y1_val)).reshape(-1, 1)
    weights = np.concatenate((weight0_val, weight1_val)).reshape(-1, 1)

    # Create PyTorch datasets and dataloaders
    dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y).float(), torch.Tensor(weights))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loaders = {
        "train": train_loader,
        "test": test_loader,
        "val": data_loader
    }

    return loaders

# Module used to add parameter for fitting

class AddParams2Input2(nn.Module):
    def __init__(self):
        super(AddParams2Input2, self).__init__()
        self.lambda_mean = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.lambda_width = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.mu_mean = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.mu_width = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.nu_mean = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.nu_width = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def forward(self, inputs):
        batch_lambdas = self.lambda_mean + self.lambda_width * torch.randn(inputs.size(0))
        batch_mus = self.mu_mean + self.mu_width * torch.randn(inputs.size(0))
        batch_nus = self.nu_mean + self.nu_width * torch.randn(inputs.size(0))
        concatenated = torch.cat([inputs, batch_lambdas.view(-1, 1), batch_mus.view(-1, 1), batch_nus.view(-1, 1)], dim=-1)
        return concatenated


def fit_fn2(epochs, add_params_layer, fit_model, data_loader, device, optimizer, loss_fn):
    losses = []
    fit_vals = {
        "lambda": [],
        "mu": [],
        "nu": []
    }

    for epoch in range(epochs):
        add_params_layer.train()
        running_loss = 0.0
        for batch_inputs, batch_labels, batch_weights in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)

            # Forward pass
            optimizer.zero_grad()
            param_input = add_params_layer(batch_inputs)
            output = fit_model(param_input)

            # Compute the loss
            loss = loss_fn(output, batch_labels, batch_weights)

            # Backward pass and update weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_inputs.size(0)

        # epoch_loss = running_loss / len(data_loader.dataset)
        # print("epoch : {}, loss = {:.4f}, lambda = {:.4f}, mu = {:.4f}, nu = {:.4f}".format(epoch + 1, epoch_loss,
        #                                                                                     add_params_layer.params[0].item(),
        #                                                                                     add_params_layer.params[1].item(),
        #                                                                                     add_params_layer.params[2].item()))
        # losses.append(epoch_loss)
        # fit_vals["lambda"].append(add_params_layer.params[0].item())
        # fit_vals["mu"].append(add_params_layer.params[1].item())
        # fit_vals["nu"].append(add_params_layer.params[2].item())

        epoch_loss = running_loss / len(data_loader.dataset)
        print("epoch : {}, loss = {:.4f}, lambda = {:.4f}, mu = {:.4f}, nu = {:.4f}".format(epoch + 1, epoch_loss,
                                                                                            add_params_layer.lambda_mean.item(),
                                                                                            add_params_layer.mu_mean.item(),
                                                                                            add_params_layer.nu_mean.item()))


    print("**** fit values with method 2 ****")
    print("fit lambda = {} +/- {}".format(add_params_layer.lambda_mean.item(), 2.* add_params_layer.lambda_width.item()))
    print("fit mu = {} +/- {}".format(add_params_layer.mu_mean.item(), 2.* add_params_layer.mu_width.item()))
    print("fit nu = {} +/- {}".format(add_params_layer.nu_mean.item(), 2.* add_params_layer.nu_width.item()))

    # return losses, fit_vals