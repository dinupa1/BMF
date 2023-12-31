import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

plt.rc("font", size=14)

class ParamEstimatorVAE(nn.Module):
    def __init__(self, latent_size):
        super(ParamEstimatorVAE, self).__init__()

        # Encoder layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(100, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
        )

        self.fc_mu = nn.Linear(64, latent_size, bias=True)
        self.fc_logvar = nn.Linear(64, latent_size, bias=True)

        self.fc_theta_mu = nn.Linear(64, 3, bias=True)
        self.fc_theta_logvar = nn.Linear(64, 3, bias=True)

        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_size+3, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 100, bias=True),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        x = self.fc_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        theta_mu = self.fc_theta_mu(x)
        theta_logvar = self.fc_theta_logvar(x)
        return mu, logvar, theta_mu, theta_logvar
        
    def decoder(self, z):
        x = self.fc_decoder(z)
        return x
        
    def sampling(self, mu, logvar):
        std = torch.exp(0.5* logvar)
        eps = torch.randn_like(std)
        return mu + std* eps
        
    def forward(self, x):
        mu, logvar, theta_mu, theta_logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        theta = self.sampling(theta_mu, theta_logvar)
        z_cat = torch.cat((z, theta), dim=1)
        outputs = self.decoder(z_cat)
        return outputs, mu, logvar, theta_mu, theta_logvar, theta



class ParamEstimatorLoss(nn.Module):
    def __init__(self):
        super(ParamEstimatorLoss, self).__init__()
        
        self.reco_loss = nn.BCELoss(reduction="sum")
        self.theta_loss = nn.MSELoss(reduction="sum")

    def forward(self, outputs, x, mu, logvar, theta, target):
        reconstruction = self.reco_loss(outputs, x)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        regression = self.theta_loss(theta, target)
        return (reconstruction + kl_divergence + regression)/outputs.size(0)



class ParamEstimator():
    def __init__(self, latent_size, learning_rate=0.001, step_size=100, gamma=0.1):
        super(ParamEstimator, self).__init__()
        
        self.network = ParamEstimatorVAE(latent_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
    def train(self, train_tree, val_tree, batch_size, num_epochs, device):
        
        self.network = self.network.to(device)
        
        print("===> using decvice {}".format(device))
        total_trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(self.network)
        print("total trainable params: {}".format(total_trainable_params))
        
        criterion = ParamEstimatorLoss()
        
        train_dataset = TensorDataset(train_tree["X_det"], train_tree["X_par"])
        val_dataset = TensorDataset(val_tree["X_det"], val_tree["X_par"])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        training_loss, val_loss = [], []
        
        for epoch in range(num_epochs):
            self.network.train()
            runnig_loss = []
            for inputs, targets in train_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                self.optimizer.zero_grad()
                
                outputs, mu, logvar, theta_mu, theta_logvar, theta = self.network(inputs)
                loss = criterion(outputs, inputs, mu, logvar, theta, targets)
                
                loss.backward()
                self.optimizer.step()
                
                runnig_loss.append(loss.item())
                
            epoch_loss = np.nanmean(runnig_loss)
            training_loss.append(epoch_loss)
            
            self.network.eval()
            running_acc = []
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs, mu, logvar, theta_mu, theta_logvar, theta = self.network(inputs)
                    loss = criterion(outputs, inputs, mu, logvar, theta, targets)
                    
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
        
        
    def prediction(self, test_tree, batch_size, device):
        
        test_dataset = TensorDataset(test_tree["X_det"], test_tree["X_par"])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        X_par, X_pred_mu, X_pred_std = [], [], []
        
        self.network = self.network.to(device)
        
        self.network.eval()
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs, mu, logvar, theta_mu, theta_logvar, theta = self.network(inputs)
                
                X_pred_mu.append(theta_mu.cpu().detach())
                X_pred_std.append(torch.exp(0.5* theta_logvar))
                X_par.append(targets.cpu().detach())
                
        tree = {
            "X_par": torch.cat(X_par, axis=0),
            "X_pred_mu": torch.cat(X_pred_mu, axis=0),
            "X_pred_std": torch.cat(X_pred_std, axis=0),
        }
        
        torch.save(tree, "results.pt")