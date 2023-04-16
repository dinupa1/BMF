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
ntrain = 40
ntest = 10

#
# train histogram
#

Train_lambda, Train_mu, Train_nu = [], [], []
Train_mass0, Train_mass1, Train_mass2 = [], [], []
Train_pT0, Train_pT1 = [], []
Train_xF0, Train_xF1, Train_xF2, Train_xF3 = [], [], [], []
for h in range(ntrain):
    for i in range(21):
        for j in range(21):
            for k in range(21):
                df = events[hist_event* h: hist_event*(h + 1)]
                lam = -1. + 0.1* i
                mu = -1. + 0.1* j
                nu = -1. + 0.1* k
                Train_lambda.append(lam)
                Train_mu.append(mu)
                Train_nu.append(nu)
                hist4d = Hist(
                    hist.axis.Regular(3, 4.5, 8.8, name="mass"),
                    hist.axis.Regular(2, 0., 2., name="pT"),
                    hist.axis.Regular(4, 0.0, 0.8, name="xF"),
                    hist.axis.Regular(20, -pi, pi, name="phi"),
                    hist.axis.Regular(20, -0.5, 0.5, name="costh")
                ).fill(
                    df.mass, df.pT, df.xF, df.phi, df.costh,
                    weight=xsec(lam, mu, nu, df.true_phi.to_numpy(), df.true_costh.to_numpy())
                )

                bins, edge1, edge2 = hist4d[0, :, :, :, :].project("phi", "costh").to_numpy()
                Train_mass0.append(bins)

                bins, edge1, edge2 = hist4d[1, :, :, :, :].project("phi", "costh").to_numpy()
                Train_mass1.append(bins)

                bins, edge1, edge2 = hist4d[2, :, :, :, :].project("phi", "costh").to_numpy()
                Train_mass2.append(bins)

                bins, edge1, edge2 = hist4d[:, 0, :, :, :].project("phi", "costh").to_numpy()
                Train_pT0.append(bins)

                bins, edge1, edge2 = hist4d[:, 1, :, :, :].project("phi", "costh").to_numpy()
                Train_pT1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 0, :, :].project("phi", "costh").to_numpy()
                Train_xF0.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 1, :, :].project("phi", "costh").to_numpy()
                Train_xF1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 2, :, :].project("phi", "costh").to_numpy()
                Train_xF2.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 3, :, :].project("phi", "costh").to_numpy()
                Train_xF3.append(bins)

#
# test histogram
#

Test_lambda, Test_mu, Test_nu = [], [], []
Test_mass0, Test_mass1, Test_mass2 = [], [], []
Test_pT0, Test_pT1 = [], []
Test_xF0, Test_xF1, Test_xF2, Test_xF3 = [], [], [], []
for h in range(ntest):
    for i in range(11):
        for j in range(11):
            for k in range(11):
                df = events[hist_event* (h+ntrain): hist_event*(h+ ntrain + 1)]
                lam = -1. + 0.2* i
                mu = -1. + 0.2* j
                nu = -1. + 0.2* k
                Test_lambda.append(lam)
                Test_mu.append(mu)
                Test_nu.append(nu)
                hist4d = Hist(
                    hist.axis.Regular(3, 4.5, 8.8, name="mass"),
                    hist.axis.Regular(2, 0., 2., name="pT"),
                    hist.axis.Regular(4, 0.0, 0.8, name="xF"),
                    hist.axis.Regular(20, -pi, pi, name="phi"),
                    hist.axis.Regular(20, -0.5, 0.5, name="costh")
                ).fill(
                    df.mass, df.pT, df.xF, df.phi, df.costh,
                    weight=xsec(lam, mu, nu, df.true_phi.to_numpy(), df.true_costh.to_numpy())
                )

                bins, edge1, edge2 = hist4d[0, :, :, :, :].project("phi", "costh").to_numpy()
                Test_mass0.append(bins)

                bins, edge1, edge2 = hist4d[1, :, :, :, :].project("phi", "costh").to_numpy()
                Test_mass1.append(bins)

                bins, edge1, edge2 = hist4d[2, :, :, :, :].project("phi", "costh").to_numpy()
                Test_mass2.append(bins)

                bins, edge1, edge2 = hist4d[:, 0, :, :, :].project("phi", "costh").to_numpy()
                Test_pT0.append(bins)

                bins, edge1, edge2 = hist4d[:, 1, :, :, :].project("phi", "costh").to_numpy()
                Test_pT1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 0, :, :].project("phi", "costh").to_numpy()
                Test_xF0.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 1, :, :].project("phi", "costh").to_numpy()
                Test_xF1.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 2, :, :].project("phi", "costh").to_numpy()
                Test_xF2.append(bins)

                bins, edge1, edge2 = hist4d[:, :, 3, :, :].project("phi", "costh").to_numpy()
                Test_xF3.append(bins)

output = uproot.recreate("R2_mc.root", compression=uproot.ZLIB(4))

#
# train hist
#
output["Train_mass0"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_mass0
}

output["Train_mass1"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_mass1
}

output["Train_mass2"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_mass2
}

output["Train_pT0"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_pT0
}

output["Train_pT1"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_pT1
}

output["Train_xF0"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_xF0
}

output["Train_xF1"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_xF1
}

output["Train_xF2"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_xF2
}

output["Train_xF3"] = {
    "lambda": Train_lambda,
    "mu": Train_mu,
    "nu": Train_nu,
    "hist": Train_xF3
}

#
# test histogram
#
output["Test_mass0"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_mass0
}

output["Test_mass1"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_mass1
}

output["Test_mass2"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_mass2
}

output["Test_pT0"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_pT0
}

output["Test_pT1"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_pT1
}

output["Test_xF0"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_xF0
}

output["Test_xF1"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_xF1
}

output["Test_xF2"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_xF2
}

output["Test_xF3"] = {
    "lambda": Test_lambda,
    "mu": Test_mu,
    "nu": Test_nu,
    "hist": Test_xF3
}

output.close()