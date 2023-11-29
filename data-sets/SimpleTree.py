#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

save = uproot.open("../data/data.root:save")
events = save.arrays(["true_mass", "true_pT", "true_xF", "true_phi", "true_costh", "mass", "pT", "xF", "phi", "costh"])

# mass > 5.0 && xF > 0.
# bins = np.array([4., 5.5, 6.5, 9.0])
# plt.hist2d(events.true_mass[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), events.mass[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), bins=bins)
# plt.xlabel("particle level")
# plt.ylabel("detector level")
# plt.show()

# bins = np.array([0., 0.5, 1., 2.5])
# plt.hist2d(events.true_pT[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), events.pT[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), bins=bins)
# plt.xlabel("particle level")
# plt.ylabel("detector level")
# plt.show()

# bins = np.array([-0.1, 0.3, 0.5, 1.])
# plt.hist2d(events.true_xF[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), events.xF[(events.mass > 5.) & (events.xF > 0.)].to_numpy(), bins=bins)
# plt.xlabel("particle level")
# plt.ylabel("detector level")
# plt.show()

print("===> apply simple cuts to events <===")

tree = {
	"true_mass": events.true_mass[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"true_pT": events.true_pT[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"true_xF": events.true_xF[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"true_phi": events.true_phi[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"true_costh": events.true_costh[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"phi": events.phi[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
	"costh": events.costh[(events.mass > 5.) & (events.xF > 0.)].to_numpy(),
}

outfile = uproot.recreate("simple.root", compression=uproot.ZLIB(4))
outfile["tree"] = tree
outfile.close()