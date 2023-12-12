#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

save = uproot.open("../data/data.root:save")
events = save.arrays(["true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "mass", "pT", "xF", "phi", "costh"])

print("===> apply simple cuts to events <===")

tree = {
	"true_phi": events.true_phi[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"true_costh": events.true_costh[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"mass": events.mass[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"pT": events.pT[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"xF": events.xF[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"phi": events.phi[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"costh": events.costh[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
}

# pT bins
bins = np.array([0.0, 0.4, 0.8, 1.2, 2.5])
plt.hist(tree["pT"], bins=bins, density=True, alpha=0.5)
plt.xlabel("pT [GeV]")
plt.ylabel("counts [a.u.]")
plt.savefig("imgs/pT.png")
plt.close("all")

# xF bins
bins = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
plt.hist(tree["xF"], bins=bins, density=True, alpha=0.5)
plt.xlabel("xF")
plt.ylabel("counts [a.u.]")
plt.savefig("imgs/xF.png")
plt.close("all")

# mass bins
bins = np.array([5., 5.5, 6.2, 9.])
plt.hist(tree["mass"], bins=bins, density=True, alpha=0.5)
plt.xlabel("mass [GeV]")
plt.ylabel("counts [a.u.]")
plt.savefig("imgs/mass.png")
plt.close("all")

outfile = uproot.recreate("simple.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()