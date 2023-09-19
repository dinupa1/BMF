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
from sklearn.metrics import accuracy_score

plt.rc('font', size=14)

#
# VAE loss module
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
				nn.Sigmoid(),
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

#
# train VAE model
#
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

#
# classifier to paramatrize the latent space
#
class Classifier(nn.Module):
	def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 1):
		super(Classifier, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim, bias=True),
			nn.Sigmoid(),
			)

	def forward(self, x):
		x = self.fc(x)
		return x


#
# classifier train function
#
def train_classifier(model, X_data, label, batch_size, learning_rate, num_epochs, early_stopping_patience, device):


	X_train, X_val, Y_train, Y_val = train_test_split(X_data, label, test_size=0.3, shuffle=True)

	X_train_tensor = torch.from_numpy(X_train).float()
	X_val_tensor = torch.from_numpy(X_val).float()
	Y_train_tensor = torch.from_numpy(Y_train).float()
	Y_val_tensor = torch.from_numpy(Y_val).float()

	train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
	val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=Flase)

	model = model.to(device)

	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	train_loss, val_loss, val_acc = [], [], []

	best_loss = float('inf')
	best_model_weights = None
	patience_counter = 0

	for epoch in range(num_epochs):
		model.train()
		running_train, running_val, running_acc = 0.0, 0.0, 0.0
		n_train, n_val = 0., 0.
		for inputs, targets in train_dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()

			outputs = model(inputs)

			loss = criterion(outputs, targets)

			loss.backward()
			optimizer.step()

			running_train += loss.item()
			n_train += 1.

		train_loss.append(running_train/n_train)

		model.eval()
		for inputs, targets in val_dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			outputs = model(inputs)

			loss = criterion(outputs, targets)

			# accuracy accuracy
			running_acc += accuracy_score(targets.ravel(), outputs.ravel())

			running_val += loss.item()
			n_val += 1.
		val_loss.append(running_val/n_val)
		val_acc.append(running_val/n_val)

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

	fig, ax1 = plt.subplots()
	ax1.plot(train_loss, label="train loss")
	ax1.plot(val_loss, label="val. loss")
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("BCE Loss [a.u.]")
	ax1.tick_params(axis="y", labelcolor="red")

	ax2 = ax1.twinx()
	ax2.plot(val_acc, label="val. accuracy", color="green")
	ax2.set_ylabel("Accuracy", color="green")
	ax2.tick_params(axis="y", labelcolor="green")
	ax1.legend(frameon=False)
	ax2.legend(frameon=False)
	# plt.savefig("notes/09-07-2023/imgs/train_vae.png")
	# plt.close("all")
	plt.show()

	return best_model_weights


#
# module used to add parameter for fitting
#
class AddParams2Input(nn.Module):
	def __init__(self, params):
		super(AddParams2Input, self).__init__()
		self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

	def forward(self, inputs):
		batch_params = torch.ones((inputs.size(0), 1), device=inputs.device)* self.params.to(device=inputs.device)
		concatenated = torch.cat([inputs, batch_params], dim=-1)
		return concatenated

#
# optimize parameters
#
def optimize_theta(classifier, add_params, dataloader, learning_rate, num_epochs, device):

	classifier = classifier.to(device)
	add_params = add_params.to(device)

	criterion = nn.BCELoss()
	optimizer = optim.Adam(add_params.parameters(), lr=learning_rate)

	opt_loss, opt_theta = [], []

	for epoch in range(num_epochs):
		add_params.train()
		running_loss, events = 0., 0.
		for inputs, targets in dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			concat_inputs = add_params(inputs)
			outputs = classifier(concat_inputs)

			loss = criterion(outputs, targets)

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			events += 1.

		opt_loss.append(running_loss/events)
		opt_theta.append(add_params.params.item())

	plt.plot(opt_loss)
	plt.xlabel("epoch")
	plt.ylabel("loss [a.u.]")
	# plt.savefig("notes/09-07-2023/imgs/opt_loss.png")
	# plt.close("all")
	plt.show()

	plt.plot(opt_theta)
	plt.hlines(0.2, 0, len(opt_theta), colors="red", label=r"$\nu = 0.2$ true value")
	plt.xlabel("epoch")
	plt.ylabel(r"fitted $\mu$ values")
	plt.legend(frameon=False)
	# plt.savefig("notes/09-07-2023/imgs/opt_params.png")
	# plt.close("all")
	plt.show()

def scan_theta(classifier, dataloader, device):

	theta_scan = np.linspace(-0.5, 0.5, 21)

	scan_loss = []

	criterion = nn.BCELoss()

	for theta in theta_scan:
		add_params = AddParams2Input([theta])
		add_params = add_params.to(device)
		running_loss, events = 0., 0.
		for inputs, targets in dataloader:
			inputs = inputs.to(device)
			targets = targets.to(device)

			outputs = classifier(inputs)

			loss = criterion(outputs, targets)

			running_loss += loss.item()
			events += 1.

		scan_loss.append(running_loss/events)

	plt.plot(theta_scan, scan_loss)
	plt.xlabel(r"$\nu [a.u.]$")
	plt.ylabel("BCE Loss [a.u.]")
	plt.title(r"Injected $\nu = 0.2$")
	plt.show()


#
# Extract theta values
#
