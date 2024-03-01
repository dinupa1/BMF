#
#
#


import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import h5py

from sklearn.model_selection import train_test_split

tree_mc = uproot.open("data.root:save")
events_mc1 = tree_mc.arrays(["mass", "pT", "xF", "x1", "x2", "costh"]).to_numpy()

events_mc = events_mc1[(events_mc1["mass"] > 4.5) & (events_mc1["xF"] > -0.1) & (events_mc1["xF"] < 0.95) & (-0.5 < events_mc1["costh"]) & (events_mc1["costh"] < 0.5)]

print("---> # of MC events {}".format(events_mc["mass"].shape[0]))

tree = uproot.open("simple_tree.root:tree")
events = tree.arrays(["mass", "pT", "xF", "xT", "xB", "pot_p00", "liveP"]).to_numpy()

tree_mix = uproot.open("simple_tree.root:tree_mix")
events_mix = tree_mix.arrays(["mass", "pT", "xF", "xT", "xB"]).to_numpy()

tree_flask = uproot.open("simple_tree.root:tree_flask")
events_flask = tree_flask.arrays(["mass", "pT", "xF", "xT", "xB", "pot_p00", "liveP"]).to_numpy()

counts1 = np.sum(events["liveP"])
counts2 = np.sum(events_flask["liveP"])

counts = counts1/counts2

print("---> shapes {}, {}, {}".format(events.shape[0], events_mix.shape[0], events_flask.shape[0]))
print("---> POT ratio {}/{} = {}".format(counts1, counts2, counts))

train_tree_mc, test_tree_mc = train_test_split(events_mc, test_size=0.5, shuffle=True)

num_events = events["mass"].shape[0] + events_mix["mass"].shape[0] + events_flask["mass"].shape[0]

train_tree_mc1, train_tree_mc2 = train_test_split(train_tree_mc, train_size=num_events, shuffle=True)

train_weight_mc = np.ones(num_events)
tree_weight = np.ones(events["mass"].shape[0])
mix_weight = -1. * np.ones(events_mix["mass"].shape[0])
flask_weight = -counts * np.ones(events_flask["mass"].shape[0])


train_label_mc = np.zeros(num_events)
tree_label = np.ones(events["mass"].shape[0])
mix_label = np.ones(events_mix["mass"].shape[0])
flask_label = np.ones(events_flask["mass"].shape[0])


train_mc = {
    "mass": np.double(train_tree_mc1["mass"]),
    "pT": np.double(train_tree_mc1["pT"]),
    "xF": np.double(train_tree_mc1["xF"]),
    "x1": np.double(train_tree_mc1["x1"]),
    "x2": np.double(train_tree_mc1["x2"]),
    "weight": np.double(train_weight_mc),
    "label": np.double(train_label_mc),
}

train_tree = {
    "mass": np.double(np.concatenate((events["mass"], events_mix["mass"], events_flask["mass"]))),
    "pT": np.double(np.concatenate((events["pT"], events_mix["pT"], events_flask["pT"]))),
    "xF": np.double(np.concatenate((events["xF"], events_mix["xF"], events_flask["xF"]))),
    "x1": np.double(np.concatenate((events["xB"], events_mix["xB"], events_flask["xB"]))),
    "x2": np.double(np.concatenate((events["xT"], events_mix["xT"], events_flask["xT"]))),
    "weight": np.double(np.concatenate((tree_weight, mix_weight, flask_weight))),
    "label": np.double(np.concatenate((tree_label, mix_label, flask_label))),
}

test_tree_mc1, test_tree_mc2 = train_test_split(test_tree_mc, train_size=num_events, shuffle=True)

test_mc = {
    "mass": np.double(test_tree_mc1["mass"]),
    "pT": np.double(test_tree_mc1["pT"]),
    "xF": np.double(test_tree_mc1["xF"]),
    "x1": np.double(test_tree_mc1["x1"]),
    "x2": np.double(test_tree_mc1["x2"]),
}

print("---> creating root file for reweighting")

outfile = uproot.recreate("reweight_data.root", compression=uproot.ZLIB(4))
outfile["train_mc"] = train_mc
outfile["train_tree"] = train_tree
outfile["test_mc"] = test_mc
outfile.close()

print("---> creating HDF5 file for reweighting")

outputs = h5py.File("reweight_data.hdf5", "w")

outputs.create_dataset("train_mc/mass", data=train_tree_mc1["mass"])
outputs.create_dataset("train_mc/pT", data=train_tree_mc1["pT"])
outputs.create_dataset("train_mc/xF", data=train_tree_mc1["xF"])
outputs.create_dataset("train_mc/x1", data=train_tree_mc1["x1"])
outputs.create_dataset("train_mc/x2", data=train_tree_mc1["x2"])
outputs.create_dataset("train_mc/weight", data=train_weight_mc)
outputs.create_dataset("train_mc/label", data=train_label_mc)

outputs.create_dataset("train_tree/mass", data=np.concatenate((events["mass"], events_mix["mass"], events_flask["mass"])))
outputs.create_dataset("train_tree/pT", data=np.concatenate((events["pT"], events_mix["pT"], events_flask["pT"])))
outputs.create_dataset("train_tree/xF", data=np.concatenate((events["xF"], events_mix["xF"], events_flask["xF"])))
outputs.create_dataset("train_tree/x1", data=np.concatenate((events["xT"], events_mix["xT"], events_flask["xT"])))
outputs.create_dataset("train_tree/x2", data=np.concatenate((events["xB"], events_mix["xB"], events_flask["xB"])))
outputs.create_dataset("train_tree/weight", data=np.concatenate((train_weight_mc, tree_weight, mix_weight, flask_weight)))
outputs.create_dataset("train_tree/label", data=np.concatenate((train_label_mc, tree_label, mix_label, flask_label)))

outputs.create_dataset("test_mc/mass", data=test_tree_mc1["mass"])
outputs.create_dataset("test_mc/pT", data=test_tree_mc1["pT"])
outputs.create_dataset("test_mc/xF", data=test_tree_mc1["xF"])
outputs.create_dataset("test_mc/x1", data=test_tree_mc1["x1"])
outputs.create_dataset("test_mc/x2", data=test_tree_mc1["x2"])

outputs.close()
