import numpy as np
import uproot
import matplotlib.pyplot as plt
import hist
from hist import Hist

pi = np.pi

tree = uproot.open("clean_LH2.root:tree")
tree_mix = uproot.open("clean_mix_LH2.root:tree")

# for now, we skip the flask data
event = tree.arrays(["mass", "pT", "xF", "phi", "costh"])
event_mix = tree_mix.arrays(["mass", "pT", "xF", "phi", "costh"])

# histogram to store the clean data
hist_raw = Hist(
    hist.axis.Regular(2, 4.5, 6.5, name="mass"),
    hist.axis.Regular(3, 0.0, 1.5, name="pT"),
    hist.axis.Regular(4, 0.0, 0.8, name="xF"),
    hist.axis.Regular(20, -pi, pi, name="phi"),
    hist.axis.Regular(20, -0.5, 0.5, name="costh")
)

hist_mix = Hist(
    hist.axis.Regular(2, 4.5, 6.5, name="mass"),
    hist.axis.Regular(3, 0.0, 1.5, name="pT"),
    hist.axis.Regular(4, 0.0, 0.8, name="xF"),
    hist.axis.Regular(20, -pi, pi, name="phi"),
    hist.axis.Regular(20, -0.5, 0.5, name="costh")
)

hist_raw.fill(event.mass, event.pT, event.xF, event.phi, event.costh)
hist_mix.fill(event_mix.mass, event_mix.pT, event_mix.xF, event_mix.phi, event_mix.costh)

hist_clean = hist_raw - hist_mix


# save data for prediction -> save only the bin content of the 2D histogram
# axis -> mass, pT, xF, phi, costh
output = uproot.recreate("clean_plot.root", compression=uproot.ZLIB(4))

output["mass0"] = hist_clean[0, :, :, :, :].project("phi", "costh")
output["mass1"] = hist_clean[1, :, :, :, :].project("phi", "costh")
output["pT0"] = hist_clean[:, 0, :, :, :].project("phi", "costh")
output["pT1"] = hist_clean[:, 1, :, :, :].project("phi", "costh")
output["pT2"] = hist_clean[:, 2, :, :, :].project("phi", "costh")
output["xF0"] = hist_clean[:, :, 0, :, :].project("phi", "costh")
output["xF1"] = hist_clean[:, :, 1, :, :].project("phi", "costh")
output["xF2"] = hist_clean[:, :, 2, :, :].project("phi", "costh")
output["xF3"] = hist_clean[:, :, 3, :, :].project("phi", "costh")

output["total"] = hist_clean[:, :, :, :, :].project("phi", "costh")

output.close()

output = uproot.recreate("clean_real.root", compression=uproot.ZLIB(4))

X_mass0, X_mass1, X_pT0, X_pT1, X_pT2, X_xF0, X_xF1, X_xF2, X_xF3 = [], [], [], [], [], [], [], [], []

for i in range(50):
    bins, edge1, edge2 = hist_clean[0, :, :, :, :].project("phi", "costh").to_numpy()
    X_mass0.append(bins)

    bins, edge1, edge2 = hist_clean[1, :, :, :, :].project("phi", "costh").to_numpy()
    X_mass1.append(bins)

    bins, edge1, edge2 = hist_clean[:, 0, :, :, :].project("phi", "costh").to_numpy()
    X_pT0.append(bins)

    bins, edge1, edge2 = hist_clean[:, 1, :, :, :].project("phi", "costh").to_numpy()
    X_pT1.append(bins)

    bins, edge1, edge2 = hist_clean[:, 2, :, :, :].project("phi", "costh").to_numpy()
    X_pT2.append(bins)

    bins, edge1, edge2 = hist_clean[:, :, 0, :, :].project("phi", "costh").to_numpy()
    X_xF0.append(bins)

    bins, edge1, edge2 = hist_clean[:, :, 1, :, :].project("phi", "costh").to_numpy()
    X_xF1.append(bins)

    bins, edge1, edge2 = hist_clean[:, :, 2, :, :].project("phi", "costh").to_numpy()
    X_xF2.append(bins)

    bins, edge1, edge2 = hist_clean[:, :, 3, :, :].project("phi", "costh").to_numpy()
    X_xF3.append(bins)

output["mass0"] = {
    "hist": X_mass0
}

output["mass1"] = {
    "hist": X_mass1
}

output["pT0"] = {
    "hist": X_pT0
}

output["pT1"] = {
    "hist": X_pT1
}

output["pT2"] = {
    "hist": X_pT2
}

output["xF0"] = {
    "hist": X_xF0
}

output["xF1"] = {
    "hist": X_xF1
}

output["xF2"] = {
    "hist": X_xF2
}

output["xF3"] = {
    "hist": X_xF3
}

output.close()