#
# dinupa3@gmail.com
#
#
import numpy as np
import matplotlib.pyplot as plt

import uproot
import awkward as ak

from sklearn.model_selection import train_test_split

save = uproot.open("data.root:save")
events = save.arrays(["true_phi", "true_costh", "phi", "costh", "mass", "xF"])

cuts = (events.mass > 5.) & (events.xF > 0.)

tree = np.stack((events.true_phi[cuts].to_numpy(), events.true_costh[cuts].to_numpy(),  events.phi[cuts].to_numpy(), events.costh[cuts].to_numpy()), axis=-1)


X_train, X_test = train_test_split(tree, test_size=0.5, shuffle=True)
X0_train, X1_train = train_test_split(X_train, test_size=0.5, shuffle=True)
X0_test, X1_test = train_test_split(X_test, test_size=0.5, shuffle=True)

X0_train_data = {
    "true_phi": X0_train[:, 0],
    "true_costh": X0_train[:, 1],
    "phi": X0_train[:, 2],
    "costh": X0_train[:, 3],
    }

X1_train_data = {
    "true_phi": X1_train[:, 0],
    "true_costh": X1_train[:, 1],
    "phi": X1_train[:, 2],
    "costh": X1_train[:, 3],
    }


X0_test_data = {
    "true_phi": X0_test[:, 0],
    "true_costh": X0_test[:, 1],
    "phi": X0_test[:, 2],
    "costh": X0_test[:, 3],
    }

X1_test_data = {
    "true_phi": X1_test[:, 0],
    "true_costh": X1_test[:, 1],
    "phi": X1_test[:, 2],
    "costh": X1_test[:, 3],
    }


outfile = uproot.recreate("split.root", compression=uproot.ZLIB(4))
outfile["X0_train"] = X0_train_data
outfile["X1_train"] = X1_train_data
outfile["X0_test"] = X0_test_data
outfile["X1_test"] = X1_test_data
outfile.close()
