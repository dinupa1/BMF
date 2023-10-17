import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

data = uproot.open("GMC_lh2_DY_RUN3_All.root:result_mc")


events = data.arrays(["fpga1", "mass", "phi", "costh", "true_phi", "true_costh"])

tree = {
    "fpga1": events.fpga1.to_numpy(),
    "mass": events.mass.to_numpy(),
    "phi": events.phi.to_numpy(),
    "costh": events.costh.to_numpy(),
    "true_phi": events.true_phi.to_numpy(),
    "true_costh": events.true_costh.to_numpy(),
}

outfile = uproot.recreate("simple.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()