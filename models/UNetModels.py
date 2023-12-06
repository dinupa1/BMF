import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rc("font", size=14)

class ConvBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(ConvBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU()

	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		return x


class EncoderBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(EncoderBlock, self).__init__()

		self.conv = ConvBlock(in_channel, out_channel)
		self.pool = nn.MaxPool2d((2, 2))

	def forward(self, inputs):
		x = self.conv(inputs)
		p = self.pool(x)

		return x, p


class DecoderBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(DecoderBlock, self).__init__()

		self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
		self.conv = ConvBlock(out_channel+out_channel, out_channel)

	def forward(self, inputs, skip):
		x = self.up(inputs)
		x = torch.cat([x, skip], axis=1)
		x = self.conv(x)

		return x

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()

		# Encoder
		self.enc1 = EncoderBlock(3, 8)
		self.enc2 = EncoderBlock(8, 16)

		# Latent space
		self.latent = ConvBlock(16, 32)

		# Decoder
		self.dec1 = DecoderBlock(32, 16)
		self.dec2 = DecoderBlock(16, 8)

		# Regression
		self.reg = nn.Conv2d(8, 6, kernel_size=4, padding=0, stride=3)

	def forward(self, inputs):
		# Encoder
		s1, p1 = self.enc1(inputs)
		s2, p2 = self.enc2(p1)

		# Latent space
		z = self.latent(p2)

		# Decoder
		d1 = self.dec1(z, s2)
		d2 = self.dec2(d1, s1)

		# Regression
		outputs = self.reg(d2)

		return outputs


class UNetDataset(torch.utils.data.Dataset):
	def __init__(self, inputs, targets):
		self.inputs = inputs
		self.targets = targets

	def __getitem__(self, index):
		return self.inputs[index], self.targets[index]

	def __len__(self):
		return len(self.targets)


def fit_unet(train_tree, val_tree, batch_size, model, criterion, optimizer, scheduler, num_epochs, device):

    train_dataset = TensorDataset(train_tree["X_det"], train_tree["X_par"])
    val_dataset = TensorDataset(val_tree["X_det"], val_tree["X_par"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    training_loss, val_loss = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = []
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        epoch_loss = np.nanmean(running_loss)
        training_loss.append(epoch_loss)

        model.eval()
        running_acc = []
        with torh.no_grad():
        	for inputs, targets in val_dataloader:
        		inputs = inputs.to(device)
        		targets = targets.to(device)

        		outputs = model(inputs)
        		loss = criterion(outputs, targets)

        		running_acc.append(loss.item())
        	epoch_val = np.nanmean(running_acc)
        	val_loss.append(epoch_val)

        scheduler.step()
        print("---> Epoch = {}/{} train loss = {:.3f} val. loss = {:.3f}".format(epoch, num_epochs, epoch_loss, epoch_val))

    plt.plot(training_loss, label="train")
    plt.plot(val_loss, label="val.")
    plt.xlabel("epoch")
    plt.ylabel("loss [a.u.]")
    plt.yscale("log")
    plt.legend(frameon=False)
    plt.savefig("imgs/fit_loss.png")
    plt.close("all")


def unet_prediction(model, test_tree, device):

    model = model.to(device)
    X_test_tensor = test_tree["X_det"].to(device)

    model.eval()
    with torh.no_grad():
    	outputs = model(X_test_tensor)

    tree = {
    	"X_par": test_tree["X_par"],
    	"X_preds": outputs.cpu().detach(),
    }

    torch.save(tree, "results.pt")