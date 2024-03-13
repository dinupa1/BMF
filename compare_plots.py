import numpy as np
import matplotlib.pyplot as plt


import uproot
import awkward as ak

import h5py

hdf="../e906-LH2-data/output.hdf5"

infile = h5py.File(hdf, "r")

branches = ["mass", "pT", "xT", "xB", "xF", "weight"]

tree = {}
tree_mc = {}

for branch in branches:

    tree[branch] = np.array(infile["train_tree"][branch], dtype=float)
    tree_mc[branch] = np.array(infile["test_tree_mc"][branch], dtype=float)


tree_mc["weight2"] = np.array(infile["test_tree_mc"]["weights2"], dtype=float)

outputs = uproot.recreate("../e906-LH2-data/output.root", compression=uproot.ZLIB(4))
outputs["tree"] = tree
outputs["tree_mc"] = tree_mc
outputs.close()