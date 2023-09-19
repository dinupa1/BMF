#
# dinupa3@gmail.com
# 09-18-2023
#

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

plt.rc('font', size=14)

#
# Loss function
#
class VAELoss(nn.Module):
	def __init__(self):
		super(VAELoss, self).__init__()

	def forward(self, reco, mu, logvar, x):
		loss_fn = nn.BCELoss(reduction="sum")
		reconstruction = loss_fn(reco, x)
		kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return (reconstruction + kl_divergence)/reco.size(0)


#
# VAE model
#
class VAEModel(nn.Module):
	def __init__(self, input_dim: int = 1, hidden_dim: int = 64, param_dim: int = 3, latent_dim: int = 8):
		super(VAEModel, self).__init__()

		self.fc_encoder = nn.Sequential(
				nn.Linear(input_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
			)

		self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True)
		self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True)

		self.fc_decoder = nn.Sequential(
				nn.Linear(latent_dim + param_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, hidden_dim, bias=True),
				nn.BatchNorm1d(hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, input_dim, bias=True),
				# nn.BatchNorm1d(hidden_dim),
				# nn.ReLU(),
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

	def decode(self, z, thetas):
		z = torch.cat((z, thetas), -1)
		reco = self.fc_decoder(z)
		return reco

	def forward(self, x, thetas):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		reco = self.decode(z, thetas)
		return reco, mu, logvar


def fit_vae(model, X_data, thetas_data, batch_size, learning_rate, num_epochs, early_stopping_patience, device):
	#
	# train, test split
	#
	X_train, X_val, theta_train, theta_val = train_test_split(X_data, thetas_data, test_size=0.3, shuffle=True)

	X_train_tensor = torch.from_numpy(X_train).float()
	X_val_tensor = torch.from_numpy(X_val).float()
	theta_train_tensor = torch.from_numpy(theta_train).float()
	theta_val_tensor = torch.from_numpy(theta_val).float()

	train_dataset = TensorDataset(X_train_tensor, theta_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, theta_val_tensor)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	criterion = VAELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	model = model.to(device)

	train_loss, val_loss, val_acc = [], [], []

	best_loss = float('inf')
	best_model_weights = None
	patience_counter = 0

	for epoch in range(num_epochs):
		model.train()
		running_train, running_val, running_acc = 0.0, 0.0, 0.0
		n_train, n_val = 0., 0.
		for inputs, thetas in train_dataloader:
			inputs = inputs.to(device)
			thetas = thetas.to(device)

			optimizer.zero_grad()

			reco, mu, logvar = model(inputs, thetas)

			loss = criterion(reco, mu, logvar, inputs)

			loss.backward()
			optimizer.step()

			running_train += loss.item()
			n_train += 1.

		train_loss.append(running_train/n_train)

		model.eval()
		for inputs, thetas in val_dataloader:
			inputs = inputs.to(device)
			thetas = thetas.to(device)

			reco, mu, logvar = model(inputs, thetas)

			loss = criterion(reco, mu, logvar, inputs)

			running_val += loss.item()
			n_val += 1.
		val_loss.append(running_val/n_val)

		if epoch%10 == 0:
			print("===> Epoch {}/{} Training loss {:.3f}, Val. loss {:.3f} ".format(
				epoch, num_epochs, running_train/n_train, running_val/n_val))

		# Check for early stopping
		if running_val/n_val < best_loss:
			best_loss = running_val/n_val
			best_model_weights = model.state_dict()
			patience_counter = 0
		else:
			patience_counter += 1

		if patience_counter >= early_stopping_patience:
			print("Early stopping at epoch {}".format(epoch))
			break

	plt.plot(train_loss, label="train loss")
	plt.plot(val_loss, label="val. loss")
	plt.legend(frameon=True)
	plt.xlabel("epoch")
	plt.ylabel("loss")
	plt.show()

	return best_model_weights