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

tree = uproot.open("data.root:save")
events0 = tree.arrays(["occuD1", "true_phi", "true_costh", "mass", "pT", "xF", "phi", "costh"])

events = events0[(events0.mass > 4.5) & (events0.mass < 8.8) & (events0.occuD1 < 200.)]


# fill the histogram as clean data
hist_event = 12662
ntrain = 50

X_lambda, X_mu, X_nu = [], [], []
X_mass0, X_mass1 = [], []
X_pT0, X_pT1, X_pT2 = [], [], []
X_xF0, X_xF1, X_xF2, X_xF3 = [], [], [], []
for h in range(ntrain):
    for i in range(21):
        for j in range(21):
            for k in range(21):
                df = events[hist_event* h: hist_event*(h + 1)]
                lam = -1. + 0.1* i
                mu = -1. + 0.1* j
                nu = -1. + 0.1* k
                X_lambda.append(lam)
                X_mu.append(mu)
                X_nu.append(nu)
                hist4d = Hist(
                    hist.axis.Regular(2, 4.5, 6.5, name="mass"),
                    hist.axis.Regular(3, 0., 1.5, name="pT"),
                    hist.axis.Regular(4, 0.0, 0.8, name="xF"),
                    hist.axis.Regular(20, -pi, pi, name="phi"),
                    hist.axis.Regular(20, -0.5, 0.5, name="costh")
                ).fill(
                    df.mass, df.pT, df.xF, df.phi, df.costh,
                    weight=xsec(lam, mu, nu, df.true_phi.to_numpy(), df.true_costh.to_numpy())
                )

                bins, edge1, edge2 = hist4d[0, :, :, :, :].project("phi", "costh").to_numpy()
                X_mass0.append(bins)

                bins, edge1, edge2 = hist4d[1, :, :, :, :].project("phi", "costh").to_numpy()
                X_mass1.append(bins)

                bins, edge1, edge2 = hist4d[:, 0, :, :, :].project("phi", "costh").to_numpy()
                X_pT0.append(bins)

                bins, edge1, edge2 = hist4d[:, 1, :, :, :].project("phi", "costh").to_numpy()
                X_pT1.append(bins)

                bins, edge1, edge2 = hist4d[:, 2, :, :, :].project("phi", "costh").to_numpy()
                X_pT2.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 0, :, :].project("phi", "costh").to_numpy()
                X_xF0.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 1, :, :].project("phi", "costh").to_numpy()
                X_xF1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 2, :, :].project("phi", "costh").to_numpy()
                X_xF2.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 3, :, :].project("phi", "costh").to_numpy()
                X_xF3.append(bins)

output = uproot.recreate("clean_mc.root", compression=uproot.ZLIB(4))
output["mass0"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_mass0
}

output["mass1"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_mass1
}

output["pT0"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_pT0
}

output["pT1"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_pT1
}

output["pT2"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_pT2
}

output["xF0"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_xF0
}

output["xF1"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_xF1
}

output["xF2"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_xF2
}

output["xF3"] = {
    "lambda": X_lambda,
    "mu": X_mu,
    "nu": X_nu,
    "hist": X_xF3
}


output.close()