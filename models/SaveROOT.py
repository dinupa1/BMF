import numpy as np
import torch
import uproot

tree = torch.load("results.pt")

save = {
	"X_par": tree["X_par"],
	"X_preds": tree["X_preds"],
}

outfile = uproot.recreate("../plots/results.root", compression=uproot.ZLIB(4))
outfile["save"] = save
outfile.close()