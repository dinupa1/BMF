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
events1 = tree.arrays(["occuD1", "true_phi", "true_costh", "phi", "costh"])

events = events1[events1.occuD1 < 200.]

# print(events.to_numpy().shape)
# print(events1.to_numpy().shape)

hist_event = 50000
ntrain = 41
ntest = 50

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
test_lam_1, test_mu_1, test_nu_1, test_hist_1 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam_1.append(0.25)
    test_mu_1.append(-0.15)
    test_nu_1.append(0.15)
    hist2d = Hist(
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh"),
    ).fill(
        df.phi, df.costh, weight=xsec(test_lam_1[i], test_mu_1[i], test_nu_1[i], df.true_phi.to_numpy(),
                                      df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist2d.to_numpy()

    test_hist_1.append(bins)

# we make test histograms
test_lam_2, test_mu_2, test_nu_2, test_hist_2 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam_2.append(-0.25)
    test_mu_2.append(0.35)
    test_nu_2.append(0.45)
    hist2d = Hist(
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh"),
    ).fill(
        df.phi, df.costh, weight=xsec(test_lam_2[i], test_mu_2[i], test_nu_2[i], df.true_phi.to_numpy(),
                                      df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist2d.to_numpy()

    test_hist_2.append(bins)

# we make test histograms
test_lam_3, test_mu_3, test_nu_3, test_hist_3 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam_3.append(0.45)
    test_mu_3.append(0.15)
    test_nu_3.append(-0.35)
    hist2d = Hist(
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh"),
    ).fill(
        df.phi, df.costh, weight=xsec(test_lam_3[i], test_mu_3[i], test_nu_3[i], df.true_phi.to_numpy(),
                                      df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist2d.to_numpy()

    test_hist_3.append(bins)

# we make test histograms
test_lam_4, test_mu_4, test_nu_4, test_hist_4 = [], [], [], []
for i in range(ntest):
    df = events[hist_event* (ntrain + i): hist_event* (ntrain+ 1+ i)]
    test_lam_4.append(-0.45)
    test_mu_4.append(0.15)
    test_nu_4.append(0.25)
    hist2d = Hist(
        hist.axis.Regular(20, -pi, pi, name="phi"),
        hist.axis.Regular(20, -0.6, 0.6, name="costh"),
    ).fill(
        df.phi, df.costh, weight=xsec(test_lam_4[i], test_mu_4[i], test_nu_4[i], df.true_phi.to_numpy(),
                                      df.true_costh.to_numpy())
    )

    bins, edge1, edge2 = hist2d.to_numpy()

    test_hist_4.append(bins)

output = uproot.recreate("hist.root", compression=uproot.ZLIB(4))
output["train"] = {
    "lambda": train_lam,
    "mu": train_mu,
    "nu": train_nu,
    "hist": train_hist
}

output["test1"] = {
    "lambda": test_lam_1,
    "mu": test_mu_1,
    "nu": test_nu_1,
    "hist": test_hist_1
}

output["test2"] = {
    "lambda": test_lam_2,
    "mu": test_mu_2,
    "nu": test_nu_2,
    "hist": test_hist_2
}

output["test3"] = {
    "lambda": test_lam_3,
    "mu": test_mu_3,
    "nu": test_nu_3,
    "hist": test_hist_3
}

output["test4"] = {
    "lambda": test_lam_4,
    "mu": test_mu_4,
    "nu": test_nu_4,
    "hist": test_hist_4
}

output.close()