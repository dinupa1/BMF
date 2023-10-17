import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

plt.rc("font", size=14)


class DenoisingAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super(DenoisingAE, self).__init__()

        self.fc_encoder = nn.Sequential(
            nn.Linear(12* 12, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            # nn.Linear(64, 32, bias=True),
            # nn.ReLU(),
            nn.Linear(64, latent_dim, bias=True),
            nn.ReLU(),
        )

        # self.fc_mu = nn.Linear(32, latent_dim, bias=True)
        # self.fc_logvar = nn.Linear(32, latent_dim, bias=True)

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64, bias=True),
            nn.ReLU(),
            # nn.Linear(32, 64, bias=True),
            # nn.ReLU(),
            nn.Linear(64, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 12* 12, bias=True),
            nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.fc_encoder(x)
        # mu = self.fc_mu(h)
        # logvar = self.fc_logvar(h)
        return z

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def decode(self, z):
        r = self.fc_decoder(z)
        return r

    def forward(self, x):
        z = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        r = self.decode(z)
        return r


class DenoisingLoss(nn.Module):
    def __init__(self):
        super(DenoisingLoss, self).__init__()

    def reconstruction(self, src, tgt):
        # loss_fn = nn.BCELoss(reduction="sum")
        loss_fn = nn.MSELoss(reduction="sum")
        return loss_fn(src, tgt)

    def kl_divergence(self, mu, logvar):
        return -0.5* torch.sum(1.+ logvar- mu.pow(2)- logvar.exp())

    def forward(self, src, mu, logvar, tgt):
        return (self.reconstruction(src, tgt) + self.kl_divergence(mu, logvar))/tgt.size(0)


def fit_denoising_ae(train_tree, batch_size, model, criterion, optimizer, num_epochs, device):

    # train test split
    true_hist = train_tree["true_hist"].array().to_numpy()
    reco_hist = train_tree["reco_hist"].array().to_numpy()


    X_train, X_val, Y_train, Y_val = train_test_split(true_hist, reco_hist, test_size=0.3, shuffle=True)

    # convert to tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    Y_val_tensor = torch.from_numpy(Y_val).float()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    training_loss, val_loss = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = []
        for true_hist, reco_hist in train_dataloader:
            true_hist = true_hist.to(device)
            reco_hist = reco_hist.to(device)

            optimizer.zero_grad()

            reco = model(reco_hist)
            loss = criterion(reco, true_hist)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        epoch_loss = np.nanmean(running_loss)
        training_loss.append(epoch_loss)

        model.eval()
        running_acc = []
        for true_hist, reco_hist in val_dataloader:
            true_hist = true_hist.to(device)
            reco_hist = reco_hist.to(device)

            reco = model(reco_hist)
            loss = criterion(reco, true_hist)

            running_acc.append(loss.item())

        epoch_val = np.nanmean(running_acc)
        val_loss.append(epoch_val)

        print("===> Epoch = {}/{} train loss = {:.3f} val. loss = {:.3f}".format(epoch, num_epochs, epoch_loss, epoch_val))

    plt.plot(training_loss, label="train")
    plt.plot(val_loss, label="val.")
    plt.xlabel("epoch")
    plt.ylabel("loss [a.u.]")
    plt.yscale("log")
    plt.legend(frameon=False)
    plt.savefig("imgs/fit_loss.png")
    plt.close("all")


def denoise_reco_hist(model, X_val):

    X_val_tensor = torch.from_numpy(X_val["reco_hist"].array().to_numpy()).float()

    model.eval()
    outputs = model(X_val_tensor)

    tree = {
        "true_hist": X_val["true_hist"].array().to_numpy(),
        "true_error": X_val["true_error"].array().to_numpy(),
        "reco_hist": X_val["reco_hist"].array().to_numpy(),
        "reco_error": X_val["reco_error"].array().to_numpy(),
        "pred_hist": outputs.detach().numpy(),
        "lambda": X_val["lambda"].array().to_numpy(),
        "mu": X_val["mu"].array().to_numpy(),
        "nu": X_val["nu"].array().to_numpy(),
    }

    outfile = uproot.recreate("results.root", compression=uproot.ZLIB(4))
    outfile["tree"] = tree
    outfile.close()
