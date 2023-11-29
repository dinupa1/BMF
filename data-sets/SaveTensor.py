import numpy as np

import uproot
import awkward as ak

import torch


train_tree = uproot.open("unet.root:train_tree")
val_tree = uproot.open("unet.root:val_tree")
test_tree = uproot.open("unet.root:test_tree")

tensor_tree = {
	"train_tree": {
		"X_par": torch.from_numpy(train_tree["X_par"].array().to_numpy()).float(),
		"X_det": torch.from_numpy(train_tree["X_det"].array().to_numpy()).float(),
	},
	"val_tree": {
		"X_par": torch.from_numpy(val_tree["X_par"].array().to_numpy()).float(),
		"X_det": torch.from_numpy(val_tree["X_det"].array().to_numpy()).float(),
	},
	"test_tree": {
		"X_par": torch.from_numpy(test_tree["X_par"].array().to_numpy()).float(),
		"X_det": torch.from_numpy(test_tree["X_det"].array().to_numpy()).float(),
	}
}

torch.save(tensor_tree, "unet-tensor.pt")