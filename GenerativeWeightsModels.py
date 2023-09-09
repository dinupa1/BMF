#
# dinupa3@gmail.com
# 09-06-2024
#

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


class GaussianVAE(nn.Module):
    def __init__(self, input_dim: int = 1, latent_dim: int = 5, hidden_dim: int = 20):
        super(GaussianVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=True),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reco_x = self.decode(z)
        return reco_x, mu, logvar


class GaussianVAELoss(nn.Module):
    def __init__(self):
        super(GaussianVAELoss, self).__init__()

    def forward(self, reco_x, x, mu, logvar):
        reco_fn = nn.MSELoss(reduction="sum")
        reconstruction = reco_fn(reco_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (reconstruction + kl_divergence)/reco_x.size(0)


class Classifier(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 20, output_dim: int = 1):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# Loss function
def train_vae(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, early_stopping_patience):
    model = model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_labels in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            reco_x, mu, logvar = model(batch_x)
            loss = criterion(reco_x, batch_x, mu, logvar)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_x, batch_labels in test_loader:
                batch_x = batch_x.to(device)

                reco_x, mu, logvar = model(batch_x)
                loss = criterion(reco_x, batch_x, mu, logvar)

                running_loss += loss.item() * batch_x.size(0)

            validation_loss = running_loss / len(test_loader.dataset)

            print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}".format(epoch + 1, epoch_loss, validation_loss))

            # Check for early stopping
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    return best_model_weights



# Train the classifier model
def train_classifier(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience):
    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)

                running_loss += loss.item() * batch_inputs.size(0)

            validation_loss = running_loss / len(test_loader.dataset)

            print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}".format(epoch + 1, epoch_loss, validation_loss))

            # Check for early stopping
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print("Early stopping at epoch {}".format(epoch))
                break

    return best_model_weights