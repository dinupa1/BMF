import numpy as np

import uproot
import awkward as ak

import torch

train_tree = uproot.open("unet.root:train_tree")
train_true_hist = train_tree["true_hist"].array().to_numpy()
train_reco_hist = train_tree["reco_hist"].array().to_numpy()
train_lambda = train_tree["lambda"].array().to_numpy()
train_mu = train_tree["mu"].array().to_numpy()
train_nu = train_tree["nu"].array().to_numpy()

val_tree = uproot.open("unet.root:val_tree")
val_true_hist = val_tree["true_hist"].array().to_numpy()
val_reco_hist = val_tree["reco_hist"].array().to_numpy()
val_lambda = val_tree["lambda"].array().to_numpy()
val_mu = val_tree["mu"].array().to_numpy()
val_nu = val_tree["nu"].array().to_numpy()

test_tree = uproot.open("unet.root:test_tree")
test_true_hist = test_tree["true_hist"].array().to_numpy()
test_reco_hist = test_tree["reco_hist"].array().to_numpy()
test_lambda = test_tree["lambda"].array().to_numpy()
test_mu = test_tree["mu"].array().to_numpy()
test_nu = test_tree["nu"].array().to_numpy()

print("===> save to pytorch tensor <===")

bmf_tensor ={
    "train_tensor": {
        "true_hist": torch.from_numpy(train_true_hist).float(),
        "reco_hist": torch.from_numpy(train_reco_hist).float(),
        "lambda": torch.from_numpy(train_lambda).float(),
        "mu": torch.from_numpy(train_mu).float(),
        "nu": torch.from_numpy(train_nu).float(),
        },
    "val_tensor": {
        "true_hist": torch.from_numpy(val_true_hist).float(),
        "reco_hist": torch.from_numpy(val_reco_hist).float(),
        "lambda": torch.from_numpy(val_lambda).float(),
        "mu": torch.from_numpy(val_mu).float(),
        "nu": torch.from_numpy(val_nu).float(),
        },
    "test_tensor": {
        "true_hist": torch.from_numpy(test_true_hist).float(),
        "reco_hist": torch.from_numpy(test_reco_hist).float(),
        "lambda": torch.from_numpy(test_lambda).float(),
        "mu": torch.from_numpy(test_mu).float(),
        "nu": torch.from_numpy(test_nu).float(),
        }
    }


torch.save(bmf_tensor, "bmf-tensor.pt")

# X_array = torch.load("bmf-tensor.pt")
#
#
# print(X_array["train_tensor"]["true_hist"].shape)
