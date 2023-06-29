import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

# Bootstrapping Resampling

tree = uproot.open("BMFData.root:save")
data = [(true_phi, true_costh, phi, costh) for true_phi, true_costh, phi, costh in zip(tree["true_phi"].array(library="np"), tree["true_costh"].array(library="np"), tree["phi"].array(library="np"), tree["costh"].array(library="np"))]
data = np.array(data)


# train test split
n_test = 20000
X_train, X_test = train_test_split(data, test_size=n_test, shuffle=True)


X_resample = resample(X_train, n_samples=n_test, replace=False)

bins = np.linspace(-np.pi, np.pi, 31)
plt.hist(X_test[:, 0], bins=bins, alpha=0.4)
plt.hist(X_resample[:, 0], bins=bins, label="resampled", histtype='step', color='k')
plt.xlabel(r"$\phi_{True}$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.savefig("notes/06-30-2023/rs_true_phi.png")
plt.close("all")

bins = np.linspace(-0.5, 0.5, 31)
plt.hist(X_test[:, 1], bins=bins, alpha=0.4)
plt.hist(X_resample[:, 1], bins=bins, label="resampled", histtype='step', color='k')
plt.xlabel(r"$\cos\theta_{True}$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.savefig("notes/06-30-2023/rs_true_costh.png")
plt.close("all")

bins = np.linspace(-np.pi, np.pi, 31)
plt.hist(X_test[:, 2], bins=bins, alpha=0.4)
plt.hist(X_resample[:, 2], bins=bins, label="resampled", histtype='step', color='k')
plt.xlabel(r"$\phi_{Reco.}$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.savefig("notes/06-30-2023/rs_phi.png")
plt.close("all")

bins = np.linspace(-0.5, 0.5, 31)
plt.hist(X_test[:, 3], bins=bins, alpha=0.4)
plt.hist(X_resample[:, 3], bins=bins, label="resampled", histtype='step', color='k')
plt.xlabel(r"$\cos\theta_{Reco.}$")
plt.ylabel("counts")
plt.legend(frameon=False)
plt.savefig("notes/06-30-2023/rs_costh.png")
plt.close("all")

print("true_phi {}, resample {}".format(np.mean(X_test[:, 0]), np.mean(X_resample[:, 0])))
print("true_phi {}, resample {}".format(np.std(X_test[:, 0]), np.std(X_resample[:, 0])))

print("true_costh {}, resample {}".format(np.mean(X_test[:, 1]), np.mean(X_resample[:, 1])))
print("true_costh {}, resample {}".format(np.std(X_test[:, 1]), np.std(X_resample[:, 1])))

print("phi {}, resample {}".format(np.mean(X_test[:, 2]), np.mean(X_resample[:, 2])))
print("phi {}, resample {}".format(np.std(X_test[:, 2]), np.std(X_resample[:, 2])))

print("costh {}, resample {}".format(np.mean(X_test[:, 3]), np.mean(X_resample[:, 3])))
print("costh {}, resample {}".format(np.std(X_test[:, 3]), np.std(X_resample[:, 3])))