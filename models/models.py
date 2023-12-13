import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

plt.rc("font", size=14)


class Autoencoder(nn.Module):
	def __init__(self, latent_size=16):
		super().__init__()

		self.encoder = nn.Sequential(
			nn.Linear(4* 10* 10, 128, bias=True),
			nn.ReLU(),
			nn.Linear(128, 64, bias=True),
			nn.ReLU(),
			nn.Linear(64, 32, bias=True),
			nn.ReLU(),
			nn.Linear(32, latent_size, bias=True),
			)

		self.decoder = nn.Sequential(
			nn.Linear(latent_size, 32, bias=True),
			nn.ReLU(),
			nn.Linear(32, 64, bias=True),
			nn.ReLU(),
			nn.Linear(64, 128, bias=True),
			nn.ReLU(),
			nn.Linear(128, 4* 3, bias=True),
			)

	def encode(self, x):
		z = self.encoder(x)
		return z

	def decode(self, x):
		x = self.decoder(x)
		return x

	def forward(self, x):
		z = self.encode(x)
		x = self.decode(z)
		return x


class ParamCNN(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1), # 10, 10
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 5, 5
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((2, 2)), # 2, 2
			)

		self.fc = nn.Sequential(
			nn.Linear(32* 2* 2, 32, bias=True),
			nn.ReLU(),
			nn.Linear(32, 16, bias=True),
			nn.ReLU(),
			nn.Linear(16, 4*3)
			)



	def forward(self, x):
		x = self.conv(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		x = x.view(x.size(0), 4, 3)
		return x


class ParamExtractor():
	def __init__(self, latent_size, learning_rate=0.001, step_size=100, gamma=0.1):
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
				targets = targets.view(targets.size(0), -1)

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
					targets = targets.view(targets.size(0), -1)

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

		X_par, X_pred = [], []

		self.network = self.network.to("cpu")

		self.network.eval()
		with torch.no_grad():
			for inputs, targets in test_dataloader:

				outputs = self.network(inputs)

				# y = outputs.view(inputs.size(0), 4, 3)
				# print("prediction shape : ", y.shape)
				# print("target shape : ", targets.shape)

				X_pred.append(outputs.view(outputs.size(0), 4, 3).detach())
				X_par.append(targets)

		tree = {
		"X_par": torch.cat(X_par, axis=0),
		"X_pred": torch.cat(X_pred, axis=0),
		}

		torch.save(tree, "results.pt")