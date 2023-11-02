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


class DenoisingUNet(nn.Module):
    def __init__(self):
        super(DenoisingUNet, self).__init__()

        kernel = 3
        pad = 1
        channel = 20

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(2, channel, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            )

        self.encoder_pool1 = nn.MaxPool2d(2)

        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel* 2, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel* 2),
            nn.ReLU(),
            nn.Conv2d(channel* 2, channel* 2, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            )

        self.encoder_pool2 = nn.MaxPool2d(2)

        self.bottlenck = nn.Sequential(
            nn.Conv2d(channel* 2, channel* 4, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel* 4),
            nn.ReLU(),
            nn.Conv2d(channel* 4, channel* 4, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            )

        self.decoder_decon1 = nn.ConvTranspose2d(channel* 4, channel* 2, kernel_size=2, stride=2)

        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(channel* 4, channel* 2, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel* 2),
            nn.ReLU(),
            nn.Conv2d(channel* 2, channel* 2, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            )

        self.decoder_decon2 = nn.ConvTranspose2d(channel* 2, channel, kernel_size=2, stride=2)

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(channel* 2, channel, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            )

        self.outputs = nn.Conv2d(channel, 2, kernel_size=1, padding=0)

        # self.outputs = nn.Sequential(
        #     nn.Conv2d(channel, 1, kernel_size=1, padding=0),
        #     nn.Sigmoid(),
        #     )

    def forward(self, x):
        # encoder
        enco_o1 = self.encoder_conv1(x)
        enco_p1 = self.encoder_pool1(enco_o1)

        enco_o2 = self.encoder_conv2(enco_p1)
        enco_p2 = self.encoder_pool2(enco_o2)

        # bottlenck
        bn_out = self.bottlenck(enco_p2)

        # decoder
        decon_o1 = self.decoder_decon1(bn_out)
        deco_in1 = torch.cat((decon_o1, enco_o2), dim=1)
        deco_o1 = self.decoder_conv1(deco_in1)

        decon_o2 = self.decoder_decon2(deco_o1)
        deco_in2 = torch.cat((decon_o2, enco_o1), dim=1)
        deco_o2 = self.decoder_conv2(deco_in2)

        # outputs
        outputs = self.outputs(deco_o2)
        return outputs



class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, true_hist, reco_hist):
        self.true_hist = torch.Tensor(true_hist)
        self.reco_hist = torch.Tensor(reco_hist)

    def __getitem__(self, index):
        return self.true_hist[index], self.reco_hist[index]

    def __len__(self):
        return len(self.reco_hist)



def fit_denoising_unet(train_tree, val_tree, batch_size, model, criterion, optimizer, num_epochs, device):

    # train test split
    X_train = train_tree["reco_hist"].array().to_numpy()
    Y_train = train_tree["true_hist"].array().to_numpy()

    X_val = val_tree["reco_hist"].array().to_numpy()
    Y_val = val_tree["true_hist"].array().to_numpy()

    # convert to tensor
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    Y_val_tensor = torch.from_numpy(Y_val).float()


    train_dataset = UNetDataset(X_train_tensor, Y_train_tensor)
    val_dataset = UNetDataset(X_val_tensor, Y_val_tensor)

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


def denoise_reco_hist(model, X_val, device):

    X_val_tensor = torch.from_numpy(X_val["reco_hist"].array().to_numpy()).float()

    model = model.to(device)
    X_val_tensor = X_val_tensor.to(device)

    model.eval()
    outputs = model(X_val_tensor)

    tree = {
        "true_hist": X_val["true_hist"].array().to_numpy(),
        "reco_hist": X_val["reco_hist"].array().to_numpy(),
        "pred_hist": outputs.cpu().detach().numpy(),
        "lambda": X_val["lambda"].array().to_numpy(),
        "mu": X_val["mu"].array().to_numpy(),
        "nu": X_val["nu"].array().to_numpy(),
    }

    outfile = uproot.recreate("results.root", compression=uproot.ZLIB(4))
    outfile["tree"] = tree
    outfile.close()
