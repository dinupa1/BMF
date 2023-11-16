#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

data = uproot.open("simple.root:tree")
events = data.arrays(["fpga1", "mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh"]).to_numpy()

print("===> split the data to train, val and test <===")

train_val_events, test_events = train_test_split(events, test_size=0.3, shuffle=True)
train_events, val_events = train_test_split(train_val_events, test_size=0.3, shuffle=True)

train_data = {
    "fpga1": train_events["fpga1"],
    "mass": train_events["mass"],
    "pT": train_events["pT"],
    "xF": train_events["xF"],
    "phi": train_events["phi"],
    "costh": train_events["costh"],
    "true_mass": train_events["true_mass"],
    "true_pT": train_events["true_pT"],
    "true_xF": train_events["true_xF"],
    "true_phi": train_events["true_phi"],
    "true_costh": train_events["true_costh"],
}

val_data = {
    "fpga1": val_events["fpga1"],
    "mass": val_events["mass"],
    "pT": val_events["pT"],
    "xF": val_events["xF"],
    "phi": val_events["phi"],
    "costh": val_events["costh"],
    "true_mass": val_events["true_mass"],
    "true_pT": val_events["true_pT"],
    "true_xF": val_events["true_xF"],
    "true_phi": val_events["true_phi"],
    "true_costh": val_events["true_costh"],
}

test_data = {
    "fpga1": test_events["fpga1"],
    "mass": test_events["mass"],
    "pT": test_events["pT"],
    "xF": test_events["xF"],
    "phi": test_events["phi"],
    "costh": test_events["costh"],
    "true_mass": test_events["true_mass"],
    "true_pT": test_events["true_pT"],
    "true_xF": test_events["true_xF"],
    "true_phi": test_events["true_phi"],
    "true_costh": test_events["true_costh"],
}


outfile = uproot.recreate("split.root", compression=uproot.ZLIB(4))
outfile["train_data"] = train_data
outfile["val_data"] = val_data
outfile["test_data"] = test_data
outfile.close()
