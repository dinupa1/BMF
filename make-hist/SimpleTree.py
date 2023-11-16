#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

DIR="/seaquest/users/dinupa/bmf-data/"

data = uproot.open(DIR+"GMC_lh2_DY_RUN3_All.root:result_mc")
events = data.arrays(["fpga1", "mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh"])

print("===> applying simple cuts to events <===")

tree = {
    "fpga1": events.fpga1[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "mass": events.mass[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "pT": events.pT[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "xF": events.xF[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "phi": events.phi[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "costh": events.costh[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "true_mass": events.true_mass[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "true_pT": events.true_pT[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "true_xF": events.true_xF[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "true_phi": events.true_phi[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
    "true_costh": events.true_costh[(events.true_mass > 4.0) & (-0.6 < events.true_costh) & (events.true_costh < 0.6)],
}

outfile = uproot.recreate("simple.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()
