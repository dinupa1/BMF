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

tree2 = torch.load("examples.pt")

save2 = {
	"X_par_mean": tree2["X_par_mean"].numpy(),
	"X_par_std": tree2["X_par_std"].numpy(),
	"X_pred_mean": tree2["X_pred_mean"].numpy(),
	"X_pred_std": tree2["X_pred_std"].numpy(),
}


outfile2 = uproot.recreate("../plots/examples.root", compression=uproot.ZLIB(4))
outfile2["save2"] = save2
outfile2.close()