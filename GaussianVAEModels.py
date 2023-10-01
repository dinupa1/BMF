import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


class GaussianVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super(GaussianVAE, self).__init__()

        self.fc_encoder = nn.Sequential(
            nn.Linear(2, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(32, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(32, latent_dim, bias=True)

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim+1, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True),
        )

    def encode(self, x):
        h = self.fc_encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        r = self.fc_decoder(z)
        return r

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat((z, x[:, 0].reshape(-1, 1)), dim=-1)
        r = self.decode(z)
        return r, mu, logvar

    def fit(self, X_data,  batch_size, criterion, optimizer, num_epochs, device, plotson=False):

        self.device = device

        X_train, X_val = train_test_split(X_data, test_size=0.3, shuffle=True)

        X_train_tesnsor = torch.from_numpy(X_train).float().to(device)
        X_val_tensor = torch.from_numpy(X_val).float().to(device)

        train_dataset = TensorDataset(X_train_tesnsor)
        val_dataset = TensorDataset(X_val_tensor)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        train_loss, val_loss = [], []

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.
            for inputs in train_dataloader:
                inputs = inputs[0]

                reco, mu, logvar = self.forward(inputs)
                loss = criterion(reco, mu, logvar, inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            train_loss.append(np.nanmean(running_loss))

            self.eval()
            running_val = 0.
            for inputs in val_dataloader:
                inputs = inputs[0]

                reco, mu, logvar = self.forward(inputs)
                loss = criterion(reco, mu, logvar, inputs)
                running_val += loss.item()
            val_loss.append(np.nanmean(running_val))

            print("===> Epoch {}/{} train loss = {:.3f} val. loss = {:.3f}".format(epoch, num_epochs, np.nanmean(running_loss), np.nanmean(running_val)))

        if plotson==True:
            plt.plot(train_loss, label="Training loss")
            plt.plot(val_loss, label="val. loss")
            plt.xlabel("epoch")
            plt.ylabel("loss [a.u.]")
            plt.legend(frameon=False)
            plt.show()

        return self.state_dict()


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def reconstruction(self, src, tgt):
        loss_fn = nn.MSELoss(reduction="sum")
        return loss_fn(src, tgt[:, 0].reshape(-1, 1))

    def kl_divergence(self, mu, logvar):
        return -0.5* torch.sum(1.+ logvar- mu.pow(2)- logvar.exp())

    def forward(self, src, mu, logvar, tgt):
        return (self.reconstruction(src, tgt) + self.kl_divergence(mu, logvar))/tgt.size(0)


class AddTheta2Inputs(nn.Module):
    def __init__(self, thetas):
        super(AddTheta2Inputs, self).__init__()
        self.params = nn.Parameter(torch.Tensor(thetas), requires_grad=True)

    def forward(self, inputs):
        batch_params = torch.ones((inputs.size(0), 1), device=inputs.device) * self.params.to(device=inputs.device)
        concatenated = torch.cat([inputs, batch_params], dim=-1)
        return concatenated