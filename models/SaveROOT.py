import numpy as np
import torch
import uproot

tree = torch.load("results.pt")

save = {
	"X_par": tree["X_par"].numpy(),
	"X_pred": tree["X_pred"].numpy(),
}

outfile = uproot.recreate("../plots/results.root", compression=uproot.ZLIB(4))
outfile["save"] = save
outfile.close()