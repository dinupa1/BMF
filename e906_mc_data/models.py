import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import h5py


# save outputs
save = h5py.File("net_fit.hdf5", "w")


# model used for likelihood learning
class NetClassifier(nn.Module):
    def __init__(self, input_dim: int=5, output_dim: int=1, hidden_dim: int=32):
        super(NetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc5 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x


# module used to add parameter for fitting
class AddParams2Input(nn.Module):
    def __init__(self, params):
        super(AddParams2Input, self).__init__()
        self.params = nn.Parameter(torch.Tensor(params), requires_grad=True)

    def forward(self, inputs):
        batch_params = torch.ones((inputs.size(0), 1), device=inputs.device) * self.params.to(device=inputs.device)
        concatenated = torch.cat([inputs, batch_params], dim=-1)
        return concatenated


# Custom loss function
class NetLoss(nn.Module):
    def __init__(self):
        super(NetLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        criterion = nn.BCELoss(reduction="none")
        loss = criterion(outputs, targets)
        weighted_loss = loss* weights
        return weighted_loss.mean()


# model used for fitting
class NetFit():
    def __init__(self, hidden_dim: int=10, learning_rate: int=0.001, step_size: int=100, gamma: int=0.1, batch_size: int=1024):
        super(NetFit, self)

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.batch_size = batch_size

    def dataloaders(self, X0_train_tree, X1_train_tree):

        X0 = X0_train_tree["x"]
        Y0 = X0_train_tree["y"]
        W0 = X0_train_tree["weight"]
        theta0 = X0_train_tree["theta"]

        X1 = X1_train_tree["x"]
        Y1 = X1_train_tree["y"]
        W1 = X1_train_tree["weight"]
        theta1 = X1_train_tree["theta"]

        X01 = np.concatenate((X0, theta0), axis=1)
        X11 = np.concatenate((X1, theta1), axis=1)

        X = np.concatenate((X01, X11))
        Y = np.concatenate((Y0, Y1))
        weight = np.concatenate((W0, W1))

        X_train, X_val, Y_train, Y_val, weight_train, weight_val = train_test_split(X, Y, weight, test_size=0.4, shuffle=True)

        print("---> train shape : {} , {}, {}".format(X_train.shape, Y_train.shape, weight_train.shape))
        print("---> test shape : {} , {}, {}".format(X_val.shape, Y_val.shape, weight_val.shape))

        train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(), torch.from_numpy(weight_train).float())
        val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float(), torch.from_numpy(weight_val).float())

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self, X0_train_tree, X1_train_tree, num_epochs, device, early_stopping_patience):

        classifier = NetClassifier(hidden_dim=self.hidden_dim)

        classifier = classifier.to(device)

        print("---> using decvice {}".format(device))
        total_trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        print(classifier)
        print("---> total trainable params: {}".format(total_trainable_params))

        criterion = NetLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        train_dataloader, val_dataloader = self.dataloaders(X0_train_tree, X1_train_tree)

        best_loss = float('inf')
        best_model_weights = None
        patience_counter = 0

        training_loss, val_loss = [], []

        for epoch in range(num_epochs):
            classifier.train()
            running_loss = 0.0
            for inputs, targets, weights in train_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                weights = weights.to(device)

                optimizer.zero_grad()

                outputs = classifier(inputs)
                loss = criterion(outputs, targets, weights)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()*inputs.size(0)

            epoch_train_loss = running_loss/len(train_dataloader.dataset)
            training_loss.append(epoch_train_loss)


            classifier.eval()
            with torch.no_grad():
                running_loss = 0.0
                for inputs, targets, weights in val_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    weights = weights.to(device)

                    outputs = classifier(inputs)
                    loss = criterion(outputs, targets, weights)

                    running_loss += loss.item()*inputs.size(0)

                epoch_val_loss = running_loss/len(val_dataloader.dataset)
                val_loss.append(epoch_val_loss)

                print("Epoch {}: Train Loss = {:.4f}, Test Loss = {:.4f}".format(epoch + 1, epoch_train_loss, epoch_val_loss))

                # Check for early stopping
                if epoch_val_loss < best_loss:
                    best_loss = epoch_val_loss
                    best_model_weights = classifier.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print("Early stopping at epoch {}".format(epoch))
                    break
            scheduler.step()


        #save.create_dataset("loss/train", data=np.array(training_loss))
        #save.create_dataset("loss/val", data=np.array(val_loss))

        plt.plot(training_loss, label="train")
        plt.plot(val_loss, label="val.")
        plt.xlabel("epoch [a.u.]")
        plt.ylabel("BCE loss [a.u.]")
        plt.yscale("log")
        plt.legend(frameon=False)
        plt.savefig("training_loss.png")
        plt.close("all")

        return best_model_weights

    #def scan(self, best_model_weights, X0_test_tree, X1_test_tree):

        #classifier = ParamsClassifier(hidden_dim=self.hidden_dim)
        #classifier.load_state_dict(best_model_weights)

        ## Set all weights in fit model to non-trainable
        #for param in classifier.parameters():
            #param.requires_grad = False

        #criterion = ParamsLoss()

        #mu_values = np.linspace(-3., 1., 50)

        #X0 = X0_test_tree["x"].reshape(-1, 1)
        #Y0 = X0_test_tree["y"].reshape(-1, 1)
        #W0 = X0_test_tree["weight"].reshape(-1, 1)

        #X1 = X1_test_tree["x"].reshape(-1, 1)
        #Y1 = X1_test_tree["y"].reshape(-1, 1)
        #W1 = X1_test_tree["weight"].reshape(-1, 1)

        #X = torch.cat((X0, X1)).float()
        #target = torch.cat((Y0, Y1)).float()
        #weight = torch.cat((W0, W1)).float()

        #scan_loss = []

        #classifier.eval()
        #for mu in mu_values:
            #theta = torch.full(X.shape, mu).float()
            #X_theta = torch.cat((X, theta), dim=1)
            #outputs = classifier(X_theta)
            #loss = criterion(outputs, target, weight)
            #scan_loss.append(loss.item())

        #plt.plot(mu_values, scan_loss)
        #plt.xlabel("mu [a.u.]")
        #plt.ylabel("BCE loss [a.u.]")
        #plt.savefig("imgs/scan_loss.png")
        #plt.close("all")

    def reweight(self, best_model_weights, X0_test_tree, X1_test_tree):

        classifier = ParamsClassifier(hidden_dim=self.hidden_dim)
        classifier.load_state_dict(best_model_weights)

        X0 = X0_test_tree["x"].reshape(-1, 1)
        theta1 = X1_test_tree["theta"].reshape(-1, 1)

        X = torch.cat((X0, theta1), dim=1).float()

        classifier.eval()
        with torch.no_grad():
            preds = classifier(X).detach().ravel()
            weights = preds/(1.0 - preds)


        save.create_dataset("weights/x0", data=X0_test_tree["x"])
        save.create_dataset("weights/weight0", data=X0_test_tree["weight"])
        save.create_dataset("weights/x0_err", data=X0_test_tree["x_err"])
        save.create_dataset("weights/reweight", weights)
        save.create_dataset("weights/x1", data=X1_test_tree["x"])
        save.create_dataset("weights/weight1", data=X1_test_tree["weight"])
        save.create_dataset("weights/x1_err", data=X1_test_tree["x_err"])

    def fit(self, best_model_weights, X0_test_tree, X1_test_tree, num_runs, num_epochs):

        fit_model = NetClassifier(hidden_dim=self.hidden_dim)
        fit_model.load_state_dict(best_model_weights)

        # Set all weights in fit model to non-trainable
        for param in fit_model.parameters():
            param.requires_grad = False

        X0 = X0_test_tree["x"]
        Y0 = X0_test_tree["y"]
        W0 = X0_test_tree["weight"]
        theta0 = X0_test_tree["theta"]

        X0_bt, W0_bt = resample(X0, W0)

        X1 = X1_test_tree["x"]
        Y1 = X1_test_tree["y"]
        W1 = X1_test_tree["weight"]
        theta1 = X1_test_tree["theta"]

        X = np.concatenate((X0_bt, X1))
        targets = np.concatenate((Y0, Y1))
        weights = np.concatenate((W0_bt, W1))

        X_tensor = torch.from_numpy(X).float()
        W_tensor = torch.from_numpy(weights).float()
        Y_tensor = torch.from_numpy(targets).float()

        lambda_fits = []
        lambda_init = []
        mu_fits = []
        mu_init = []
        nu_fits = []
        nu_init = []

        for i in range(num_runs):

            theta_fit_init = [np.random.uniform(-1.0, 1.0, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0], np.random.uniform(-0.5, 0.5, 1)[0]]

            lambda_init.append(theta_fit_init[0])
            mu_init.append(theta_fit_init[1])
            nu_init.append(theta_fit_init[2])

            add_params_layer = AddParams2Input(theta_fit_init)

            criterion = NetLoss()
            optimizer = optim.Adam(add_params_layer.parameters(), lr=0.01)
            scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

            losses = []
            fit_lambda_vals = []
            fit_mu_vals = []
            fit_nu_vals = []

            for epoch in range(num_epochs):
                add_params_layer.train()
                optimizer.zero_grad()

                X_conct = add_params_layer(X_tensor)
                outputs = fit_model(X_conct)

                loss = criterion(outputs, Y_tensor, W_tensor)

                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss.item())
                fit_lambda_vals.append(add_params_layer.params[0].item())
                fit_mu_vals.append(add_params_layer.params[1].item())
                fit_nu_vals.append(add_params_layer.params[2].item())


            #save.create_dataset("fit/loss", data=np.array(losses))
            #save.create_dataset("fit/lambda_fit", data=np.array(fit_lambda_vals))
            #save.create_dataset("fit/mu_fit", data=np.array(fit_mu_vals))
            #save.create_dataset("fit/nu_fit", data=np.array(fit_nu_vals))


            plt.plot(fit_nu_vals, label='Fit', color='r')
            plt.hlines(0.2, 0, len(fit_nu_vals), label='Truth')
            plt.xlabel("Epochs")
            plt.ylabel(r'$\mu_{fit}$')
            plt.legend(frameon=False)
            plt.savefig("mu_fits.png")
            plt.close("all")

            lambda_fits.append(fit_lambda_vals[-1])
            mu_fits.append(fit_mu_vals[-1])
            nu_fits.append(fit_nu_vals[-1])

        save.create_dataset("fit_final/lambda_fits", data=np.array(lambda_fits))
        save.create_dataset("fit_final/mu_fits", data=np.array(mu_fits))
        save.create_dataset("fit_final/nu_fits", data=np.array(nu_fits))
        save.create_dataset("fit_final/lambda_inits", data=np.array(lambda_init))
        save.create_dataset("fit_final/mu_inits", data=np.array(mu_init))
        save.create_dataset("fit_final/nu_inits", data=np.array(nu_init))
        save.close()
