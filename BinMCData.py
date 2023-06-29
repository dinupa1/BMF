import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import uproot
import awkward as ak

import ROOT

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
n_samples = 1000000
n_events = 10000
n_val = 20000


LAMBDA0, MU0, NU0 = 1.0, 0.0, 0.0


# Sample lambda, mu, nu values in the range (0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5)
LAMBDA1 = np.random.uniform(0.5, 1.5, n_samples)
MU1 = np.random.uniform(-0.5, 0.5, n_samples)
NU1 = np.random.uniform(-0.5, 0.5, n_samples)

LAMBDA_val, MU_val, NU_val = 0.92, -0.12, 0.34

X0, X1, X1_val = [], [], []
THETA0, THETA1, THETA1_val = [], [], []

hist_bins = 12
PI = ROOT.TMath.Pi()


def hist_fn(data_array, thetas):
    hist = ROOT.TH2D("hist", "hist", hist_bins, -PI, PI, hist_bins, -0.5, 0.5)
    [hist.Fill(phi, costh, weight_fn(thetas[0], thetas[1], thetas[2], true_phi, true_costh)) for phi, costh, true_phi, true_costh in data_array]

    hist.Scale(1/hist.Integral())

    bin_cont = np.zeros((hist_bins, hist_bins))

    for xbin in range(hist_bins):
        for ybin in range(hist_bins):
            bin_cont[xbin][ybin] = hist.GetBinContent(xbin+1, ybin+1)

    return bin_cont


for i in range(n_samples):
    thetas0 = [LAMBDA0, MU0, NU0]
    thetas1 = [LAMBDA1[i], MU1[i], NU1[i]]
    data_sample = resample(data0_train, replace=False, n_samples=n_events)
    X0.append(hist_fn(data_sample, thetas0))
    THETA0.append(thetas1)

    data_sample = resample(data1_train, replace=False, n_samples=n_events)
    X1.append(hist_fn(data_sample, thetas1))
    THETA1.append(thetas1)

    if i % 10000 == 0:
        print("Training samples: [{}/{}]".format(i, n_samples))

for i in range(n_val):
    data_sample = resample(data_val, replace=False, n_samples=n_events)
    thetas = [LAMBDA_val, MU_val, NU_val]
    X1_val.append(hist_fn(data_sample, thetas))
    THETA1_val.append(thetas)

    if i % 5000 == 0:
        print("Validation samples: [{}/{}]".format(i, n_val))

X0 = np.array(X0)
X1 = np.array(X1)
X1_val = np.array(X1_val)

print("X0 train shape : {}".format(X0.shape))
print("X1 train shape : {}".format(X1.shape))
print("X1 val shape : {}".format(X1_val.shape))

X0_dic = {
    "hist": X0,
    "theta": THETA0
}

X1_dic = {
    "hist": X1,
    "theta": THETA1
}

X1_val_dic = {
    "hist": X1_val,
    "theta": THETA1_val
}


output = uproot.recreate("BinMCData.root", compression=uproot.ZLIB(4))

output["X0_train"] = X0_dic
output["X1_train"] = X1_dic
output["X1_val"] = X1_val_dic

output.close()