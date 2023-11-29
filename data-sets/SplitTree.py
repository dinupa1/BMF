#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

tree = uproot.open("simple.root:tree")
events = tree.arrays(["true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "phi", "costh"]).to_numpy()

print("===> split the data to train, val & test <===")

train_val_events, test_events = train_test_split(events, test_size=0.3, shuffle=True)
train_events, val_events = train_test_split(train_val_events, test_size=0.4, shuffle=True)

train_data = {
	"true_mass": train_events["true_mass"],
	"true_pT": train_events["true_pT"],
	"true_xF": train_events["true_xF"],
	"true_phi": train_events["true_phi"],
	"true_costh": train_events["true_costh"],
	"phi": train_events["phi"],
	"costh": train_events["costh"],
}

val_data = {
	"true_mass": val_events["true_mass"],
	"true_pT": val_events["true_pT"],
	"true_xF": val_events["true_xF"],
	"true_phi": val_events["true_phi"],
	"true_costh": val_events["true_costh"],
	"phi": val_events["phi"],
	"costh": val_events["costh"],
}

test_data = {
	"true_mass": test_events["true_mass"],
	"true_pT": test_events["true_pT"],
	"true_xF": test_events["true_xF"],
	"true_phi": test_events["true_phi"],
	"true_costh": test_events["true_costh"],
	"phi": test_events["phi"],
	"costh": test_events["costh"],
}


outfile = uproot.recreate("split.root", compression=uproot.ZLIB(4))
outfile["train_data"] = train_data
outfile["val_data"] = val_data
outfile["test_data"] = test_data
outfile.close()