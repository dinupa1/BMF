#
# try https://arxiv.org/pdf/2007.11586.pdf
# dinupa3@gmail.com
# 05-05-2023
#

import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import hist
from hist import Hist

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from numba import njit


@njit(parallel=True)
def xsec(xx1, xx2, xx3, true_phi, true_costh):
    weight = 1. + xx1* true_costh* true_costh + 2.* xx2* true_costh* np.sqrt(1. - true_costh* true_costh)* np.cos(true_phi) + 0.5* xx3* (1. - true_costh* true_costh)* np.cos(2.* true_phi)
    return weight/(1. + true_costh* true_costh)

# real data
tree0 = uproot.open("../real_DCTR.root:tree")
theta0 = tree0.arrays(["mass", "pT", "xF", "D1"], library="pd").to_numpy()
weight0 = tree0["weight"].array().to_numpy()
theta0 = torch.Tensor(theta0)
weight0 = torch.Tensor(weight0).reshape(-1, 1)
label0 = torch.ones(theta0.shape[0]).reshape(-1, 1)

# mc data
tree1 = uproot.open("../data/data.root:save")
theta1 = tree1.arrays(["mass", "pT", "xF", "occuD1"], library="pd").to_numpy() # number of events = real - flask
selection = 4.5 < theta1[:, 0]
theta1 = theta1[selection][: 13176]
theta1 = torch.Tensor(theta1)
true_phi = tree1["true_phi"].array().to_numpy()
true_phi = true_phi[selection][: 13176]
true_costh = tree1["true_costh"].array().to_numpy()
true_costh = true_costh[selection][: 13176]
weight1 = xsec(-0.2, 0.3, -0.1, true_phi, true_costh) # calculate weight with lambda mu, nu
# print(np.min(weight1), np.max(weight1))
# weight1 = torch.ones(13176).reshape(-1, 1)
weight1 = torch.Tensor(weight1).reshape(-1, 1)
label1 = torch.zeros(13176).reshape(-1, 1)

X = torch.cat((theta0, theta1))
W = torch.cat((weight0, weight1))
Y = torch.cat((label0, label1))

batch_size = 1024

dataset = TensorDataset(X, W, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super (Net, self).__init__()

        self.fc1 = nn.Linear(4, 32, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(32, 32, bias=True)
        # self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1, bias=True)
        self.act4 = nn.Sigmoid()


    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        # x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x

net = Net()
# print(net)


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 1500

loss = []


for epoch in range(epochs):
    net.train()
    run_loss, m = 0., 0.
    for inputs, weights, targets in dataloader:
        optimizer.zero_grad()

        outputs = net(inputs)

        criterion.weight = weights

        loss_batch = criterion(outputs, targets)

        run_loss += loss_batch.item()
        m += 1.0

        loss_batch.backward()
        optimizer.step()

    loss.append(run_loss/m)
    print("epoch : {} loss : {}".format(epoch+1, run_loss/m))


plt.plot(loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("../imgs/loss.png")
plt.close("all")

net.eval()
outputs = net(theta1).detach().numpy()
weights = outputs/(1. - outputs)

theta0.detach().numpy()
weight0.detach().numpy()
theta1.detach().numpy()

output = uproot.recreate("result.root", compression=uproot.ZLIB(4))
output["result"] = {
    "mass": theta0[:, 0],
    "pT": theta0[:, 1],
    "xF": theta0[:, 2],
    "D1": theta0[:, 3],
    "weight": weight0
}


output["result_reweight"] = {
    "mass": theta1[:, 0],
    "pT": theta1[:, 1],
    "xF": theta1[:, 2],
    "D1": theta1[:, 3],
    "weight": weights
}

output.close()

# hist1 = Hist(hist.axis.Regular(20, 4.5, 8.8, name="mass")).fill(theta0[:, 0], weight=weight0)
# hist2 = Hist(hist.axis.Regular(20, 4.5, 8.8, name="mass")).fill(theta1[:, 0], weight=weights)
# hist1.plot_ratio(hist2, rp_ylabel=r"Ratio", rp_num_label="Real Data", rp_denom_label="reweighted MC")
# plt.savefig("../imgs/rp_mass.png")
# plt.close("all")
#
# hist1 = Hist(hist.axis.Regular(20, 0.0, 2.5, name="pT")).fill(theta0[:, 1], weight=weight0)
# hist2 = Hist(hist.axis.Regular(20, 0.0, 2.5, name="pT")).fill(theta1[:, 1], weight=weights)
# hist1.plot_ratio(hist2, rp_ylabel=r"Ratio", rp_num_label="Real Data", rp_denom_label="reweighted MC")
# plt.savefig("../imgs/rp_pT.png")
# plt.close("all")
#
# hist1 = Hist(hist.axis.Regular(20, 0.0, 0.9, name="xF")).fill(theta0[:, 2], weight=weight0)
# hist2 = Hist(hist.axis.Regular(20, 0.0, 0.9, name="xF")).fill(theta1[:, 2], weight=weights)
# hist1.plot_ratio(hist2, rp_ylabel=r"Ratio", rp_num_label="Real Data", rp_denom_label="reweighted MC")
# plt.savefig("../imgs/rp_xF.png")
# plt.close("all")
#
# hist1 = Hist(hist.axis.Regular(20, 0.0, 400, name="D1")).fill(theta0[:, 3], weight=weight0)
# hist2 = Hist(hist.axis.Regular(20, 0.0, 400, name="D1")).fill(theta1[:, 3], weight=weights)
# hist1.plot_ratio(hist2, rp_ylabel=r"Ratio", rp_num_label="Real Data", rp_denom_label="reweighted MC")
# plt.savefig("../imgs/rp_D1.png")
# plt.close("all")