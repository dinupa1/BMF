import uproot
import numpy as np
import awkward as ak

import hist
from hist import Hist

from numba import njit

from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

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

hist_event = 50000
ntrain = 41
ntest = 50

train_lam, train_mu, train_nu, train_hist = [], [], [], []
train_mass1, train_mass2 = [], []
train_pT1, train_pT2, train_pT3, train_pT4 = [], [], [], []
train_xF1, train_xF2, train_xF3, train_xF4 = [], [], [], []
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
                hist4d = Hist(
                    hist.axis.Regular(2, 4., 6., name="mass"),
                    hist.axis.Regular(4, 0., 1.6, name="pT"),
                    # hist.axis.Regular(4, 0.4, 0.8, name="x1"),
                    hist.axis.Regular(4, 0.1, 0.9, name="xF"),
                    hist.axis.Regular(20, -pi, pi, name="phi"),
                    hist.axis.Regular(20, -0.6, 0.6, name="costh")
                ).fill(
                    df.mass, df.pT, df.xF, df.phi, df.costh,
                    weight=xsec(lam, mu, nu, df.true_phi.to_numpy(), df.true_costh.to_numpy())
                )

                bins, edge1, edge2 = hist4d[0, :, :, :, :].project("phi", "costh").to_numpy()
                train_mass1.append(bins)

                bins, edge1, edge2 = hist4d[1, :, :, :, :].project("phi", "costh").to_numpy()
                train_mass2.append(bins)

                bins, edge1, edge2 = hist4d[:, 0, :, :, :].project("phi", "costh").to_numpy()
                train_pT1.append(bins)

                bins, edge1, edge2 = hist4d[:, 1, :, :, :].project("phi", "costh").to_numpy()
                train_pT2.append(bins)

                bins, edge1, edge2 = hist4d[:, 2, :, :, :].project("phi", "costh").to_numpy()
                train_pT3.append(bins)

                bins, edge1, edge2 = hist4d[:, 3, :, :, :].project("phi", "costh").to_numpy()
                train_pT4.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 0, :, :].project("phi", "costh").to_numpy()
                train_xF1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 1, :, :].project("phi", "costh").to_numpy()
                train_xF2.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 2, :, :].project("phi", "costh").to_numpy()
                train_xF3.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 3, :, :].project("phi", "costh").to_numpy()
                train_xF4.append(bins)

# we make test histograms
test_lam, test_mu, test_nu = [], [], []
test_mass1, test_mass2 = [], []
test_pT1, test_pT2, test_pT3, test_pT4 = [], [], [], []
test_xF1, test_xF2, test_xF3, test_xF4 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam.append(0.4)
    test_mu.append(0.4)
    test_nu.append(0.4)
    hist4d = Hist(
        hist.axis.Regular(2, 4., 6., name="mass"),
        hist.axis.Regular(4, 0., 1.6, name="pT"),
        # hist.axis.Regular(4, 0.4, 0.8, name="x1"),
        hist.axis.Regular(4, 0.1, 0.9, name="xF"),
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh")
    ).fill(
        df.mass, df.pT, df.xF, df.phi, df.costh,
        weight=xsec(test_lam[i], test_mu[i], test_nu[i], df.true_phi.to_numpy(), df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist4d[0, :, :, :, :].project("phi", "costh").to_numpy()
    test_mass1.append(bins)

    bins, edge1, edge2 = hist4d[1, :, :, :, :].project("phi", "costh").to_numpy()
    test_mass2.append(bins)

    bins, edge1, edge2 = hist4d[:, 0, :, :, :].project("phi", "costh").to_numpy()
    test_pT1.append(bins)

    bins, edge1, edge2 = hist4d[:, 1, :, :, :].project("phi", "costh").to_numpy()
    test_pT2.append(bins)

    bins, edge1, edge2 = hist4d[:, 2, :, :, :].project("phi", "costh").to_numpy()
    test_pT3.append(bins)

    bins, edge1, edge2 = hist4d[:, 3, :, :, :].project("phi", "costh").to_numpy()
    test_pT4.append(bins)

    bins, edge1, edge2 = hist4d[:, :, 0, :, :].project("phi", "costh").to_numpy()
    test_xF1.append(bins)

    bins, edge1, edge2 = hist4d[:, :, 1, :, :].project("phi", "costh").to_numpy()
    test_xF2.append(bins)

    bins, edge1, edge2 = hist4d[:, :, 2, :, :].project("phi", "costh").to_numpy()
    test_xF3.append(bins)

    bins, edge1, edge2 = hist4d[:, :, 3, :, :].project("phi", "costh").to_numpy()
    test_xF4.append(bins)

output = uproot.recreate("hist.root", compression=uproot.ZLIB(4))
output["train_mass1"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_mass1
}

output["train_mass2"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_mass2
}

output["train_pT1"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_pT1
}

output["train_pT2"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_pT2
}

output["train_pT3"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_pT3
}

output["train_pT4"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_pT4
}

output["train_xF1"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_xF1
}

output["train_xF2"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_xF2
}

output["train_xF3"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_xF3
}

output["train_xF4"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_xF4
}

output["test_mass1"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_mass1
}

output["test_mass2"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_mass2
}

output["test_pT1"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_pT1
}

output["test_pT2"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_pT2
}

output["test_pT3"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_pT3
}

output["test_pT4"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_pT4
}

output["test_xF1"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_xF1
}

output["test_xF2"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_xF2
}

output["test_xF3"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_xF3
}

output["test_xF4"] = {
    "lambda": test_lam,
    "mu": test_mu,
    "nu": test_nu,
    "hist": test_xF4
}

output.close()