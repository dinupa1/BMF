#
# dinupa3@gmail.com
# 09-06-2024
#

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


plt.rc('font', size=14)


class ParamVAE(nn.Module):
    def __init__(self, input_dim: int = 1, param_dim: int = 1, latent_dim: int = 10, hidden_dim: int = 20):
        super(ParamVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim+param_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim+param_dim, hidden_dim, bias=True),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=True),
        )

    def encode(self, x, params):
        x = torch.cat((x, params), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = torch.cat((z, params), -1)
        return self.decoder(z)

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        reco_x = self.decode(z, params)
        return reco_x, mu, logvar


class ParamVAELoss(nn.Module):
    def __init__(self):
        super(ParamVAELoss, self).__init__()

    def forward(self, reco_x, x, mu, logvar):
        reco_fn = nn.MSELoss(reduction="sum")
        reconstruction = reco_fn(reco_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (reconstruction + kl_divergence)/reco_x.size(0)


# Loss function
def train_param_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs, early_stopping_patience, scheduler):
    model = model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    train_loss = []
    test_loss = []
    val_score = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_params in train_loader:
            batch_x = batch_x.to(device)
            batch_params = batch_params.to(device)

            optimizer.zero_grad()
            reco_x, mu, logvar = model(batch_x, batch_params)
            loss = criterion(reco_x, batch_x, mu, logvar)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()* batch_x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for batch_x, batch_params in test_loader:
                batch_x = batch_x.to(device)
                batch_params = batch_params.to(device)

                reco_x, mu, logvar = model(batch_x, batch_params)
                loss = criterion(reco_x, batch_x, mu, logvar)

                score = r2_score(batch_x.detach().numpy(), reco_x.detach().numpy())

                running_loss += loss.item()* batch_x.size(0)

            validation_loss = running_loss / len(train_loader.dataset)

            print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}, R2 Score = {:.4f}".format(epoch + 1, epoch_loss, validation_loss, score))

            # scheduler.step()

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

        train_loss.append(epoch_loss)
        test_loss.append(validation_loss)
        val_score.append(score)

    fig, ax1 = plt.subplots()
    ax1.plot(train_loss, label="Train loss")
    ax1.plot(test_loss, label="Validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss [a.u.]")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(val_score, label="Validation R2 score", color="green")
    ax2.set_ylabel("R2 Score", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.savefig("notes/09-07-2023/imgs/train_vae.png")
    plt.close("all")
    # plt.show()

    return best_model_weights


# module used to add parameter for fitting
class ParamOptimizer(nn.Module):
    def __init__(self, params):
        super(ParamOptimizer, self).__init__()
        self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

    def forward(self, batch_size, device, latent_dim: int=10):
        batch_params = torch.ones((batch_size, 1), device=device) * self.params.to(device)
        batch_z = torch.randn(batch_size, latent_dim)
        return batch_params.float(), batch_z.float()


def train_optimizer(param_optimizer, param_vae, dataloader, criterion, optimizer, device, num_epochs, scheduler, latent_dim: int = 10):
    param_optimizer.to(device)
    param_vae.to(device)

    opt_loss = []
    opt_params = []

    for epoch in range(num_epochs):
        param_optimizer.train()
        running_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            batch_size = batch_x.size(0)

            batch_params, batch_z = param_optimizer(batch_size, device, latent_dim)

            reco_x = param_vae.decode(batch_z, batch_params)

            loss = criterion(reco_x, batch_x)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()* batch_size

        epoch_loss = running_loss / len(dataloader.dataset)

        scheduler.step()

        print("===> Epoch {} Loss {:.4f}".format(epoch, epoch_loss))

        opt_loss.append(epoch_loss)
        opt_params.append(param_optimizer.params.item())

    plt.plot(opt_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss [a.u.]")
    plt.savefig("notes/09-07-2023/imgs/opt_loss.png")
    plt.close("all")
    # plt.show()

    plt.plot(opt_params)
    plt.hlines(1.5, 0, len(opt_params), colors="red", label=r"$\mu = 1.5$ true value")
    plt.xlabel("epoch")
    plt.ylabel(r"fitted $\mu$ values")
    plt.legend(frameon=False)
    plt.savefig("notes/09-07-2023/imgs/opt_params.png")
    plt.close("all")
    # plt.show()


def scan_fn(param_vae, dataloader, criterion, device, latent_dim: int = 5):

    mu_scan = np.linspace(-0.5, 3.5, 31)

    scan_loss = []

    for mu in mu_scan:
        param_optimizer = ParamOptimizer([mu])
        param_optimizer.to(device)
        param_vae.to(device)
        running_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x[0].to(device)
            batch_size = batch_x.size(0)

            batch_params, batch_z = param_optimizer(batch_size, device, latent_dim)

            reco_x = param_vae.decode(batch_z, batch_params)

            loss = criterion(reco_x, batch_x)

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / len(dataloader.dataset)
        scan_loss.append(epoch_loss)

    plt.plot(mu_scan, scan_loss)
    plt.vlines(x=1.5, ymin=1., ymax=4., colors="teal", label=r"$\mu$ = 1.5 true value")
    plt.xlabel(r"$\mu$ values [a.u.]")
    plt.ylabel("loss [a.u.]")
    plt.legend(frameon=False)
    plt.savefig("notes/09-07-2023/imgs/scan_opt.png")
    plt.close("all")
    # plt.show()