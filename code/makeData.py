import uproot
import numpy as np
import awkward as ak

import hist
from hist import Hist

from numba import njit

@njit(parallel=True)
def xsec(lam, mu, nu, phi, costh):
    weight = 1. + lam* costh* costh + 2.* mu* costh* np.sqrt(1. - costh* costh)* np.cos(phi) + nu* (1. - costh* costh)* np.cos(2. * phi)/2./(1. + costh* costh)
    return weight

pi = np.pi

tree = uproot.open("../data.root:save")
events1 = tree.arrays(["occuD1", "true_phi", "true_costh", "mass", "pT", "x1", "xF", "phi", "costh"])

events = events1[events1.occuD1 < 200.]

# print(events.to_numpy().shape)
# print(events1.to_numpy().shape)

hist_event = 100000
ntrain = 15
ntest = 30

train_lam, train_mu, train_nu, train_hist = [], [], [], []

for h in range(ntrain):
    for i in range(21):
        for j in range(21):
            for k in range(21):
                df = events[hist_event* h: hist_event*(h + 1)]
                lam = -1. + 0.1* i
                mu = -1. + 0.1* j
                nu = -1. + 0.1* k
                train_lam.append(lam)
                train_mu.append(mu)
                train_nu.append(nu)
                hist2d = Hist(
                    hist.axis.Regular(20, -pi, pi, name="phi"),
                    hist.axis.Regular(20, -0.6, 0.6, name="costh"),
                ).fill(
                    df.phi, df.costh,
                    weight=xsec(lam, mu, nu, df.true_phi.to_numpy(), df.true_costh.to_numpy())
                )

                bins, edge1, edge2 = hist2d.to_numpy()
                train_hist.append(bins)

# we make test histograms
test_lam, test_mu, test_nu = [], [], []
hist_pT1, hist_pT2, hist_pT3, hist_pT4 = [], [], [], []
hist_xF1, hist_xF2, hist_xF3, hist_xF4 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam.append(0.2)
    test_mu.append(0.2)
    test_nu.append(0.2)
    hist4d = Hist(
        hist.axis.Regular(2, 4., 6., name="mass"),
        hist.axis.Regular(4, 0., 1.6, name="pT"),
        hist.axis.Regular(4, 0.4, 0.8, name="x1"),
        hist.axis.Regular(4, 0.1, 0.9, name="xF"),
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh")
    ).fill(
        df.mass, df.pT, df.x1, df.xF, df.phi, df.costh,
        weight=xsec(test_lam[i], test_mu[i], test_nu[i], df.true_phi.to_numpy(), df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist4d[:, 0, :, :, :, :].project("phi", "costh").to_numpy()
    hist_pT1.append(bins)

    bins, edge1, edge2 = hist4d[:, 1, :, :, :, :].project("phi", "costh").to_numpy()
    hist_pT2.append(bins)

    bins, edge1, edge2 = hist4d[:, 2, :, :, :, :].project("phi", "costh").to_numpy()
    hist_pT3.append(bins)

    bins, edge1, edge2 = hist4d[:, 3, :, :, :, :].project("phi", "costh").to_numpy()
    hist_pT4.append(bins)

    bins, edge1, edge2 = hist4d[:, :, :, 0, :, :].project("phi", "costh").to_numpy()
    hist_xF1.append(bins)

    bins, edge1, edge2 = hist4d[:, :, :, 1, :, :].project("phi", "costh").to_numpy()
    hist_xF2.append(bins)

    bins, edge1, edge2 = hist4d[:, :, :, 2, :, :].project("phi", "costh").to_numpy()
    hist_xF3.append(bins)

    bins, edge1, edge2 = hist4d[:, :, :, 3, :, :].project("phi", "costh").to_numpy()
    hist_xF4.append(bins)

output = uproot.recreate("hist.root", compression=uproot.ZLIB(4))
output["train"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_hist
}

output["test_pT1"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_pT1
}

output["test_pT2"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_pT2
}

output["test_pT3"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_pT3
}

output["test_pT4"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_pT4
}

output["test_xF1"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_xF1
}

output["test_xF2"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_xF2
}

output["test_xF3"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_xF3
}

output["test_xF4"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": hist_xF4
}

output.close()