import numpy as np
import torch
import uproot

tree = torch.load("weights.pt")

save = {
    "x0": tree["x0"].numpy(),
    "weight0": tree["weight0"].numpy(),
    "x0_err": tree["x0_err"].numpy(),
    "reweight": tree["reweight"].numpy(),
    "x1": tree["x1"],
    "weight1": tree["weight1"],
    "x1_err": tree["x1_err"],
}

tree2 = torch.load("mu_vals.pt")

mu_vals = {
    "mu_fits": tree2["mu_fits"].numpy(),
    "mu_inits": tree2["mu_inits"].numpy(),
    }

outfile = uproot.recreate("results.root", compression=uproot.ZLIB(4))
outfile["save"] = save
outfile["mu_vals"] = mu_vals
outfile.close()
