import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

# Function to calculate weights analytically
def weight_fn(xx1, xx2, xx3, phi, costh):
    weight = 1. + xx1 * costh * costh + 2. * xx2 * costh * np.sqrt(1. - costh * costh) * np.cos(phi) + 0.5 * xx3 * (1. - costh * costh) * np.cos(2. * phi)
    return weight / (1. + costh * costh)

# Import data
tree = uproot.open("BMFData.root:save")
events = tree.arrays(["mass", "pT", "xF", "phi", "costh", "true_phi", "true_costh"])

data = np.array([(phi, costh, true_phi, true_costh) for phi, costh, true_phi, true_costh in zip( events.phi, events.costh, events.true_phi, events.true_costh)])
data_train, data_val = train_test_split(data, test_size=0.3, shuffle=True)
data0_train, data1_train = train_test_split(data_train, test_size=0.5, shuffle=True)

# Number of samples
n_samples = 10**6
n_events = 10000


LAMBDA0, MU0, NU0 = 1.0, 0.0, 0.0


# Sample lambda, mu, nu values in the range (0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5)
LAMBDA1 = np.random.uniform(0.5, 1.5, n_samples)
MU1 = np.random.uniform(-0.5, 0.5, n_samples)
NU1 = np.random.uniform(-0.5, 0.5, n_samples)

X0, X1 = [], []

hist_bins = 8
hist_range = [[-np.pi, np.pi], [-0.5, 0.5]]

for i in range(n_samples):
    data_sample = resample(data0_train, replace=False, n_samples=n_events)
    weights = weight_fn(LAMBDA0, MU0, NU0, data_sample[:, 2], data_sample[:, 3])
    bc = np.histogram2d(data_sample[:, 0], data_sample[:, 1], bins=hist_bins, range=hist_range, density=True, weights=weights)[0]
    x0 = np.concatenate((bc.ravel(), np.array([LAMBDA1[i], MU1[i], NU1[i]])))
    X0.append(x0)

    data_sample = resample(data1_train, replace=False, n_samples=n_events)
    weights = weight_fn(LAMBDA1[i], MU1[i], NU1[i], data_sample[:, 2], data_sample[:, 3])
    bc = np.histogram2d(data_sample[:, 0], data_sample[:, 1], bins=hist_bins, range=hist_range, density=True, weights=weights)[0]
    x1 = np.concatenate((bc.ravel(), np.array([LAMBDA1[i], MU1[i], NU1[i]])))
    X1.append(x1)

    if i % 10000 == 0:
        print("Iteration: [{}/{}]".format(i, n_samples))

X0 = np.array(X0)
X1 = np.array(X1)

X0_dic, X1_dic = {}, {}

for i in range(67):
    X0_dic["branch_{}".format(i)] = X0[:, i]
    X1_dic["branch_{}".format(i)] = X0[:, i]

X_val_dic = {
    "phi": data_val[:, 0],
    "costh": data_val[:, 1],
    "true_phi": data_val[:, 2],
    "true_costh": data_val[:, 3]
}

output = uproot.recreate("BinMCData.root", compression=uproot.ZLIB(4))

output["X0_train_dic"] = X0_dic
output["X1_train_dic"] = X1_dic
output["X_val_dic"] = X_val_dic

output.close()