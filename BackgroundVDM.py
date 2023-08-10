#
# dinupa3@gmail.com
# 08-08-2023
#

import torch
import torch.nn as nn

# A simple variational diffusion model use for background substraction


class BackgroundVDM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BackgroundVDM, self).__init__()

        # Encoder layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True)


        # Decoder layers
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim, bias=True),
            )

    def encode(self, x):
        h = self.fc_encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5* logvar)
        eps = torch.randn_like(std)
        return mu + eps* std

    def decode(self, z):
        h = self.fc_decoder(z)
        return h

    def forward(self, x, timesteps):

        # Encode
        mu, logvar = self.encode(x)

        # Reparameterize and sample from the latent space
        z = self.reparameterize(mu, logvar)

        # Forward diffusion process
        for t in range(timesteps):
            diffusion_noise = torch.randn_like(z)
            alpha = torch.tensor(0.1).float()
            z_new = z + torch.sqrt(alpha)* diffusion_noise # Diffusion step
            z = z_new

        # Backward diffusion process
        for t in range(timesteps):
            z_new = self.decode(z)
            z = z_new

        # Final output
        reco_x = self.decode(z)

        return reco_x, mu, logvar


class VDMLoss(nn.Module):
    def __init__(self):
        super(VDMLoss, self).__init__()

    def forward(self, reco_x, x, batch_weight, mu, logvar):
        mse_loss = nn.MSELoss(reduction="sum")
        MSE = mse_loss(reco_x, x)
        MSE_weight = torch.sum(batch_weight*MSE)
        KLD = torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
        return (MSE_weight + KLD)/x.size(0)