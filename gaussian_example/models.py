import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


# model used for likelihood learning
class GaussClassifier(nn.Module):
    def __init__(self, input_dim: int=1, output_dim: int=1, hidden_dim: int=10):
        super(GaussClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
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


# Custom loss function
class GaussLoss(nn.Module):
    def __init__(self):
        super(GaussLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        criterion = nn.BCELoss(reduction="none")
        loss = criterion(outputs, targets)
        weighted_loss = loss* weights
        return weighted_loss.mean()


# model used for fitting
class GaussFit():
    def __init__(self, hidden_dim: int=10, learning_rate: int=0.001, step_size: int=100, gamma: int=0.1):
        super(GaussFit, self)

        self.classifier = GaussClassifier(hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def dataloaders(self, X0_train_tree, X1_train_tree, batch_size: int=1024):

        X0 = X0_train_tree["x"].reshape(-1, 1)
        Y0 = X0_train_tree["y"].reshape(-1, 1)
        W0 = X0_train_tree["weight"].reshape(-1, 1)
        theta0 = X0_train_tree["theta"].reshape(-1, 1)

        X1 = X1_train_tree["x"].reshape(-1, 1)
        Y1 = X1_train_tree["y"].reshape(-1, 1)
        W1 = X1_train_tree["weight"].reshape(-1, 1)
        theta1 = X1_train_tree["theta"].reshape(-1, 1)

        X01 = torch.cat((X0, W0), dim=1)
        X11 = torch.cat((X1, W1), dim=1)
        X = torch.cat((X01, X11))
        Y = torch.cat((Y0, Y1))
        theta = torch.cat((theta0, theta1))
        weight = torch.cat((W0, W1))

        X_train, X_val, Y_train, Y_val, weight_train, weight_val = train_test_split(X, Y, weight)

        train_dataset = TensorDataset(X_train, Y_train, weight_train)
        val_dataset = TensorDataset(X_val, Y_val, weight_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self, )
