import numpy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class ReweightModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ReweightModel, self).__init__()

        self.fc_input = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.ReLU(),
            )

        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            )

        self.fc_output = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=True),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.fc_input(x)
        x = self.fc_hidden(x)
        x = self.fc_output(x)
        return x


class ReweightLoss(nn.Module):
    def __init__(self):
        super(ReweightLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        criterion = nn.BCELoss(reduction="none")
        loss = criterion(outputs, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()