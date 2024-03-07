import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import os

plt.rc("font", size=14)

# Train
def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs, early_stopping_patience):
    best_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # Model summary
    model = model.to(device=device)
    print("using device : {}".format(device))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('total trainable params: {}'.format(total_trainable_params))

    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = []
        for batch_inputs, batch_labels, batch_weights in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels, batch_weights)

            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        epoch_loss = np.nanmean(running_loss)
        train_loss.append(epoch_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            running_loss = []
            for batch_inputs, batch_labels, batch_weights in test_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                batch_weights = batch_weights.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels, batch_weights)

                running_loss.append(loss.item())

            validation_loss = np.nanmean(running_loss)
            val_loss.append(validation_loss)

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

    fig, axs = plt.subplots(figsize=(8, 8))
    axs.plot(train_loss, label="train")
    axs.plot(val_loss, label="val.")
    axs.set_xlabel("epoch")
    axs.set_ylabel("BCE loss")
    # axs.set_yscale("log")
    axs.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("imgs/training_loss.png")
    plt.close("all")

    return best_model_weights


def reweight_fn(model, X_val):
    # Move the model to CPU for evaluation
    model = model.to(torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        preds = model(torch.Tensor(X_val)).detach().numpy().ravel()
        weights = preds / (1.0 - preds)
    return weights


def data_loaders(fname, batch_size=1024):

    infile = h5py.File(fname, "r")

    branches = ["mass", "pT", "xT", "xB", "xF", "weight"]

    train_dic = {}
    train_dic_mc = {}
    test_dic_mc = {}

    for branch in branches:
        train_dic[branch] = np.array(infile["train_tree"][branch])
        train_dic_mc[branch] = np.array(infile["train_tree_mc"][branch])
        test_dic_mc[branch] = np.array(infile["test_tree_mc"][branch])

    mass = np.concatenate((train_dic["mass"], train_dic_mc["mass"])).reshape(-1, 1)
    pT = np.concatenate((train_dic["pT"], train_dic_mc["pT"])).reshape(-1, 1)
    xT = np.concatenate((train_dic["xT"], train_dic_mc["xT"])).reshape(-1, 1)
    xB = np.concatenate((train_dic["xB"], train_dic_mc["xB"])).reshape(-1, 1)
    xF = np.concatenate((train_dic["xF"], train_dic_mc["xF"])).reshape(-1, 1)
    weight = np.concatenate((train_dic["weight"], train_dic_mc["weight"])).reshape(-1, 1)
    label = np.concatenate((1. * np.ones(len(train_dic["mass"])), 0. * np.ones(len(train_dic_mc["mass"])))).reshape(-1, 1)

    X_data = np.concatenate((mass, pT, xT, xB, xF), axis=-1)

    X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X_data, weight, label, test_size=0.4, shuffle=True)

    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).float()
    W_train_tensor = torch.from_numpy(W_train).float()

    X_val_tensor = torch.from_numpy(X_val).float()
    Y_val_tensor = torch.from_numpy(Y_val).float()
    W_val_tensor = torch.from_numpy(W_val).float()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, W_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor, W_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("---> train shape {}".format(X_train_tensor.shape))
    print("---> val shape {}".format(X_val_tensor.shape))

    test_mass = test_dic_mc["mass"].reshape(-1, 1)
    test_pT = test_dic_mc["pT"].reshape(-1, 1)
    test_xT = test_dic_mc["xT"].reshape(-1, 1)
    test_xB = test_dic_mc["xB"].reshape(-1, 1)
    test_xF = test_dic_mc["xF"].reshape(-1, 1)
    test_weight = test_dic_mc["weight"].reshape(-1, 1)

    X_test = np.concatenate((test_mass, test_pT, test_xT, test_xB, test_xF), axis=-1)

    X_test_tensor = torch.from_numpy(X_test).float()

    print("---> test shape {}".format(X_test_tensor.shape))

    infile.close()

    return train_loader, val_loader, X_test_tensor


def update_data(fname, weights):

    infile = h5py.File(fname, "r")

    branches = ["mass", "pT", "xB", "xT", "xF", "weight"]

    outputs = h5py.File("../e906-LH2-data/output.hdf5", "w")

    for branch in branches:
        outputs.create_dataset("train_tree/{}".format(branch), data=np.array(infile["train_tree"][branch]))
        outputs.create_dataset("test_tree_mc/{}".format(branch), data=np.array(infile["test_tree_mc"][branch]))

    outputs.create_dataset("test_tree_mc/weights2", data=weights)

    infile.close()
    outputs.close()
