#
# dinupa3@gmail.com
#
#
import numpy as np
import matplotlib.pyplot as plt

import h5py

import uproot
import awkward as ak

X0_train = uproot.open("net.root:X0_train_tree")
X1_train = uproot.open("net.root:X1_train_tree")
X0_test = uproot.open("net.root:X0_test_tree")
X1_test = uproot.open("net.root:X1_test_tree")

# print(X0_train["thetas"].array().to_numpy().shape)

outputs = h5py.File("net.hdf5", "w")

outputs.create_dataset("X0_train/thetas", data=X0_train["thetas"].array().to_numpy())
outputs.create_dataset("X0_train/X_par", data=X0_train["X_par"].array().to_numpy())
outputs.create_dataset("X0_train/X_det", data=X0_train["X_det"].array().to_numpy())
outputs.create_dataset("X0_train/W_par", data=X0_train["W_par"].array().to_numpy())
outputs.create_dataset("X0_train/W_det", data=X0_train["W_det"].array().to_numpy())

outputs.create_dataset("X1_train/thetas", data=X1_train["thetas"].array().to_numpy())
outputs.create_dataset("X1_train/X_par", data=X1_train["X_par"].array().to_numpy())
outputs.create_dataset("X1_train/X_det", data=X1_train["X_det"].array().to_numpy())
outputs.create_dataset("X1_train/W_par", data=X1_train["W_par"].array().to_numpy())
outputs.create_dataset("X1_train/W_det", data=X1_train["W_det"].array().to_numpy())

outputs.create_dataset("X0_test/thetas", data=X0_test["thetas"].array().to_numpy())
outputs.create_dataset("X0_test/X_par", data=X0_test["X_par"].array().to_numpy())
outputs.create_dataset("X0_test/X_det", data=X0_test["X_det"].array().to_numpy())
outputs.create_dataset("X0_test/W_par", data=X0_test["W_par"].array().to_numpy())
outputs.create_dataset("X0_test/W_det", data=X0_test["W_det"].array().to_numpy())

outputs.create_dataset("X1_test/thetas", data=X1_test["thetas"].array().to_numpy())
outputs.create_dataset("X1_test/X_par", data=X1_test["X_par"].array().to_numpy())
outputs.create_dataset("X1_test/X_det", data=X1_test["X_det"].array().to_numpy())
outputs.create_dataset("X1_test/W_par", data=X1_test["W_par"].array().to_numpy())
outputs.create_dataset("X1_test/W_det", data=X1_test["W_det"].array().to_numpy())

outputs.close()

# print(outputs["X0_train/thetas"][:])
