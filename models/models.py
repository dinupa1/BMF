import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

plt.rc("font", size=14)


class Autoencoder(nn.Module):
	def __init__(self, latent_size=64):
		super().__init__()

		self.act = nn.ReLU()

		self.enc1 = nn.Linear(4* 10* 10, 256, bias=True)
		self.enc2 = nn.Linear(256, 128, bias=True)
		self.enc3 = nn.Linear(128, latent_size, bias=True)

		self.dec1 = nn.Linear(latent_size, 128, bias=True)
		self.dec2 = nn.Linear(128, 256, bias=True)
		self.dec3 = nn.Linear(256, 4* 3, bias=True)

	def encoder(self, x):
		x = self.enc1(x)
		x = self.act(x)
		x = self.enc2(x)
		x = self.act(x)
		z = self.enc3(x)
		return z

	def decoder(self, x):
		x = self.dec1(x)
		x = self.act(x)
		x = self.dec2(x)
		x = self.act(x)
		x = self.dec3(x)
		return x

	def forward(self, x):
		z = self.encoder(x)
		x = self.decoder(z)
		return x


class ParamExtractor():
	def __init__(self, latent_size=64, learning_rate=0.001, step_size=100, gamma=0.1):
		super().__init__()

		self.network = Autoencoder(latent_size)
		self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
		self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

	def train(self, train_tree, val_tree, batch_size, num_epochs, device):

		self.network = self.network.to(device)

		print("===> using decvice {}".format(device))
		total_trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
		print(self.network)
		print("total trainable params: {}".format(total_trainable_params))

		criterion = nn.MSELoss()

		train_dataset = TensorDataset(train_tree["X_det"], train_tree["X_par"])
		val_dataset = TensorDataset(val_tree["X_det"], val_tree["X_par"])

		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


		training_loss, val_loss = [], []

		for epoch in range(num_epochs):
			self.network.train()
			runnig_loss = []
			for inputs, targets in train_dataloader:
				inputs = inputs.view(inputs.size(0), -1)
				targets = targets.view(inputs.size(0), -1)

				inputs = inputs.to(device)
				targets = targets.to(device)

				self.optimizer.zero_grad()

				outputs = self.network(inputs)

				loss = criterion(outputs, targets)

				loss.backward()
				self.optimizer.step()

				runnig_loss.append(loss.item())

			epoch_loss = np.nanmean(runnig_loss)
			training_loss.append(epoch_loss)

			self.network.eval()
			running_acc = []
			with torch.no_grad():
				for inputs, targets in val_dataloader:
					inputs = inputs.view(inputs.size(0), -1)
					targets = targets.view(inputs.size(0), -1)

					inputs = inputs.to(device)
					targets = targets.to(device)

					outputs = self.network(inputs)
					loss = criterion(outputs, targets)

					running_acc.append(loss.item())
				epoch_val = np.nanmean(running_acc)
				val_loss.append(epoch_val)

			self.scheduler.step()
			print("---> Epoch = {}/{} train loss = {:.3f} val. loss = {:.3f}".format(epoch, num_epochs, epoch_loss, epoch_val))

		plt.plot(training_loss, label="train")
		plt.plot(val_loss, label="val.")
		plt.xlabel("epoch")
		plt.ylabel("loss [a.u.]")
		plt.yscale("log")
		plt.legend(frameon=False)
		plt.savefig("imgs/fit_loss.png")
		plt.close("all")


	def prediction(self, test_tree, batch_size):

		test_dataset = TensorDataset(test_tree["X_det"], test_tree["X_par"])
		test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		tree = {
		"X_par": [],
		"X_pred": [],
		}

		self.network.eval()
		with torch.no_grad():
			for inputs, targets in test_dataloader:
				inputs = inputs.view(inputs.size(0), -1)

				outputs = self.network(inputs)

				# y = outputs.view(inputs.size(0), 4, 3)
				# print("prediction shape : ", y.shape)
				# print("target shape : ", targets.shape)

				tree["X_pred"].append(outputs.view(inputs.size(0), 4, 3).detach())
				tree["X_par"].append(targets)

		torch.save(tree, "results.pt")