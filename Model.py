import numpy as np

import torch
import torch.nn as nn


# Function to calculate weights analytically
def weight_fn(theta0, theta1, theta2, phi, costh):
    weight = 1. + theta0 * costh * costh + 2. * theta1 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * theta2 * (1. - costh * costh) * np.cos(2. * phi)
    return weight / (1. + costh * costh)


# Define classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 10, 2)
        self.fc1 = nn.Linear(43, 64, bias=True) # 3* 2* 2 + 3 = 11 input dim
        self.fc2 = nn.Linear(64, 64, bias=True)
        # self.fc3 = nn.Linear(64, 64, bias=True)
        self.fc4 = nn.Linear(64, 1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, theta):
        x = self.pool(self.relu(self.conv1(image)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.cat((x, theta), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Module used to add parameter for fitting
class AddParams2Input(nn.Module):
    def __init__(self, params):
        super(AddParams2Input, self).__init__()
        self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

    def forward(self, batch_inputs):
        batch_params = torch.ones((batch_inputs.size(0), 1), device=batch_inputs.device) * self.params.to(device=batch_inputs.device)
        return batch_inputs, batch_params


# Training step
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience):

    train_params = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_accuracy = 0
    best_model_weights = None
    patience_counter = 0

    # Train step
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_thetas, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_thetas = batch_thetas.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs, batch_thetas)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_params["train_loss"].append(epoch_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for batch_inputs, batch_thetas, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_thetas = batch_thetas.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs, batch_thetas)
                loss = criterion(outputs, batch_labels)

                running_loss += loss.item() * batch_inputs.size(0)

                predicted_labels = torch.round(outputs)
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted_labels == batch_labels).sum().item()

            validation_loss = running_loss / len(test_loader.dataset)
            validation_accuracy = correct_predictions / total_predictions
            train_params["val_loss"].append(validation_loss)
            train_params["val_acc"].append(validation_accuracy)

            print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}, Test Accuracy = {:.4f}".format(epoch + 1,
                                                                                                     epoch_loss,
                                                                                                     validation_loss,
                                                                                                     validation_accuracy))

            # Check for early stopping
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch))
                break

        if patience_counter >= early_stopping_patience:
            break

    return best_model_weights, train_params


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
            batch_inputs, param_input = add_params_layer(batch_inputs)
            output = fit_model(batch_inputs, param_input)

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


class CNNData(torch.utils.data.Dataset):
    def __init__(self, images, thetas, labels):
        self.images = torch.tensor(images).unsqueeze(1).float()
        self.thetas = torch.tensor(thetas).float()
        self.labels = torch.tensor(labels).float()

    def __getitem__(self, index):
        return self.images[index], self.thetas[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class FitData(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images).unsqueeze(1).float()
        self.labels = torch.tensor(labels).float()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
