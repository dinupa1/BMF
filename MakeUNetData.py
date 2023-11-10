#
# dinupa3@gmail.com
#

import numpy as np
import matplotlib.pyplot as plt


import uproot
import awkward as ak

import hist
from hist import Hist

import numba

from sklearn.utils import resample

@numba.njit
def weight_fn(theta0, theta1, theta2, phi, costh):
    weight = 1. + theta0* costh* costh + 2.* theta1* costh* np.sqrt(1. - costh* costh) *np.cos(phi) + 0.5* theta2* (1. - costh* costh)* np.cos(2.* phi)
    return weight/(1. + costh* costh)


# @numba.jit(nopython=True)
def make_tree(tree_name, num_events):

    print("===> reading from "+tree_name)

    data = uproot.open("split.root:"+tree_name)
    events = data.arrays(["fpga1", "mass", "pT", "xF", "phi", "costh", "true_mass", "true_pT", "true_xF", "true_phi", "true_costh"]).to_numpy()

    tree = {
        "true_hist": [],
        "reco_hist": [],
        "lambda": [],
        "mu": [],
        "nu": [],
        }

    theta0 = np.random.uniform(-1., 1., num_events)
    theta1 = np.random.uniform(-0.5, 0.5, num_events)
    theta2 = np.random.uniform(-0.5, 0.5, num_events)

    for i in range(num_events):
        X_data = resample(events, replace=False, n_samples=1000000)

        event_weight = weight_fn(theta0[i], theta1[i], theta2[i], X_data["true_phi"], X_data["true_costh"])

        hist_true = Hist(
            hist.axis.Regular(12, -np.pi, np.pi, name="phi"),
            hist.axis.Regular(12, -0.6, 0.6, name="costh"),
            storage=hist.storage.Weight()
            )

        hist_reco = Hist(
            hist.axis.Regular(12, -np.pi, np.pi, name="phi"),
            hist.axis.Regular(12, -0.6, 0.6, name="costh"),
            storage=hist.storage.Weight()
            )

        reco_phi = X_data[(X_data["fpga1"]==1) & (X_data["mass"] > 0.)]["phi"]
        reco_costh = X_data[(X_data["fpga1"]==1) & (X_data["mass"] > 0.)]["costh"]
        reco_weight = event_weight[(X_data["fpga1"]==1) & (X_data["mass"] > 0.)]

        hist_true.fill(X_data["true_phi"], X_data["true_costh"], weight=event_weight)
        hist_reco.fill(reco_phi, reco_costh, weight=reco_weight)

        tree["true_hist"].append(np.stack((hist_true.values(), hist_true.variances()), axis=0))
        tree["reco_hist"].append(np.stack((hist_reco.values(), hist_reco.variances()), axis=0))
        tree["lambda"].append(theta0[i])
        tree["mu"].append(theta1[i])
        tree["nu"].append(theta2[i])

        if (i+1)%1000==0:
            print("epoch {} lambda = {:.3f} mu = {:.3f} nu = {:.3f}".format(i+1, theta0[i], theta1[i], theta2[i]))

    return tree

outfile = uproot.recreate("unet.root", compression=uproot.ZLIB(8))
outfile["train_tree"] = make_tree("train_data", 70000)
outfile["val_tree"] = make_tree("val_data", 30000)
outfile["test_tree"] = make_tree("test_data", 40000)
outfile.close()
