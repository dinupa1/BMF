import os

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import torch

print("---> create histograms")
os.system("root -b -q CreateTree.cc")


data = uproot.open("gauss_data.root")
X0_train_tree = data["X0_train_tree"]
X1_train_tree = data["X1_train_tree"]
X0_test_tree = data["X0_test_tree"]
X1_test_tree = data["X1_test_tree"]

tensor_tree = {
    "X0_train_tree": {
        "x": torch.from_numpy(X0_train_tree["x"].array().to_numpy()).float(),
        "y": torch.from_numpy(X0_train_tree["y"].array().to_numpy()).float(),
        "theta": torch.from_numpy(X0_train_tree["theta"].array().to_numpy()).float(),
        "weight": torch.from_numpy(X0_train_tree["weight"].array().to_numpy()).float(),
        "x_err": torch.from_numpy(X0_train_tree["x_err"].array().to_numpy()).float(),
        },
    "X1_train_tree": {
        "x": torch.from_numpy(X1_train_tree["x"].array().to_numpy()).float(),
        "y": torch.from_numpy(X1_train_tree["y"].array().to_numpy()).float(),
        "theta": torch.from_numpy(X1_train_tree["theta"].array().to_numpy()).float(),
        "weight": torch.from_numpy(X1_train_tree["weight"].array().to_numpy()).float(),
        "x_err": torch.from_numpy(X1_train_tree["x_err"].array().to_numpy()).float(),
        },
    "X0_test_tree": {
        "x": torch.from_numpy(X0_test_tree["x"].array().to_numpy()).float(),
        "y": torch.from_numpy(X0_test_tree["y"].array().to_numpy()).float(),
        "theta": torch.from_numpy(X0_test_tree["theta"].array().to_numpy()).float(),
        "weight": torch.from_numpy(X0_test_tree["weight"].array().to_numpy()).float(),
        "x_err": torch.from_numpy(X0_test_tree["x_err"].array().to_numpy()).float(),
        },
    "X1_test_tree": {
        "x": torch.from_numpy(X1_test_tree["x"].array().to_numpy()).float(),
        "y": torch.from_numpy(X1_test_tree["y"].array().to_numpy()).float(),
        "theta": torch.from_numpy(X1_test_tree["theta"].array().to_numpy()).float(),
        "weight": torch.from_numpy(X1_test_tree["weight"].array().to_numpy()).float(),
        "x_err": torch.from_numpy(X1_test_tree["x_err"].array().to_numpy()).float(),
        },
    }

print("---> save to tensor")
torch.save(tensor_tree, "gauss_tensor.pt")
