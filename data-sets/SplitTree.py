#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

inFile = uproot.open("simple.root")
tree = inFile["tree"]
events = tree.arrays(["mass", "pT", "xF", "true_phi", "true_costh", "phi", "costh"]).to_numpy()

print("===> split the data to train, val & test <===")

train_val_events, test_events = train_test_split(events, test_size=0.3, shuffle=True)
train_events, val_events = train_test_split(train_val_events, test_size=0.4, shuffle=True)

train_data = {
	"mass": train_events["mass"],
	"pT": train_events["pT"],
	"xF": train_events["xF"],
	"phi": train_events["phi"],
	"costh": train_events["costh"],
	"true_phi": train_events["true_phi"],
	"true_costh": train_events["true_costh"],
}

val_data = {
	"mass": val_events["mass"],
	"pT": val_events["pT"],
	"xF": val_events["xF"],
	"phi": val_events["phi"],
	"costh": val_events["costh"],
	"true_phi": val_events["true_phi"],
	"true_costh": val_events["true_costh"],
}

test_data = {
	"mass": test_events["mass"],
	"pT": test_events["pT"],
	"xF": test_events["xF"],
	"phi": test_events["phi"],
	"costh": test_events["costh"],
	"true_phi": test_events["true_phi"],
	"true_costh": test_events["true_costh"],
}


outfile = uproot.recreate("split.root", compression=uproot.ZLIB(4))
outfile["train_data"] = train_data
outfile["val_data"] = val_data
outfile["test_data"] = test_data
outfile.close()