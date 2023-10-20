import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

import os

from sklearn.model_selection import train_test_split

data = uproot.open("GMC_lh2_DY_RUN3_All.root:result_mc")


events = data.arrays(["fpga1", "mass", "phi", "costh", "true_phi", "true_costh"]).to_numpy()

print("===> split data for unet training")

train_events, test_events = train_test_split(events, test_size=0.5, shuffle=True)

train_data = {
    "fpga1": train_events["fpga1"],
    "mass": train_events["mass"],
    "phi": train_events["phi"],
    "costh": train_events["costh"],
    "true_phi": train_events["true_phi"],
    "true_costh": train_events["true_costh"],
}

test_data = {
    "fpga1": test_events["fpga1"],
    "mass": test_events["mass"],
    "phi": test_events["phi"],
    "costh": test_events["costh"],
    "true_phi": test_events["true_phi"],
    "true_costh": test_events["true_costh"],
}


outfile = uproot.recreate("simple.root", compression=uproot.ZLIB(4))
outfile["train_data"] = train_data
outfile["test_data"] = test_data
outfile.close()

print("===> make trees for unet training")

os.system("root -b -q MakeUNetData.cc")