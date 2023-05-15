import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize, Bounds
from sklearn.model_selection import train_test_split


def cal_weight(xx1, xx2, xx3, phi, costh):
    weight = 1. + xx1* costh* costh + 2.* xx2* costh* torch.sqrt(1. - costh* costh)* torch.cos(phi) + 0.5* xx3* (1. - costh* costh)* torch.cos(2.* phi)
    return weight/(1. + costh* costh)


class Net(nn.Module):
    def __init__(self, in_features=5, out_features=1):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(in_features, 32, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.act2 = nn.ReLU()
        # self.fc3 = nn.Linear(32, 32, bias=True)
        # self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(32, out_features, bias=True)
        self.act4 = nn.Sigmoid()
        # self.act4 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        # x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return x


net = Net()
# opt_net = torch.compile(net)
total_trainable_params = sum(p.numel() for p in net.parameters())
print('total trainable params:', total_trainable_params)

data = uproot.open("../data/data.root:save")
true_mass = data["true_mass"].array(library="np").reshape(-1, 1)
true_pT = data["true_pT"].array(library="np").reshape(-1, 1)
true_xF = data["true_xF"].array(library="np").reshape(-1, 1)
true_phi = data["true_phi"].array(library="np").reshape(-1, 1)
true_costh = data["true_costh"].array(library="np").reshape(-1, 1)
mass = data["mass"].array(library="np").reshape(-1, 1)
pT = data["pT"].array(library="np").reshape(-1, 1)
xF = data["xF"].array(library="np").reshape(-1, 1)
phi = data["phi"].array(library="np").reshape(-1, 1)
costh = data["costh"].array(library="np").reshape(-1, 1)

events = np.concatenate((true_mass, true_pT, true_xF, true_phi, true_costh, mass, pT, xF, phi, costh), axis=1)

# filter only the events with 4.5 < mass
# number of events in the test sample is close to the real events
filter_events = events[4.5 < events[:, 5]][:25000]

# split data
theta0, theta1 = train_test_split(filter_events, test_size=0.5, shuffle=True)

theta0 = torch.tensor(theta0, dtype=torch.float32)
theta1 = torch.tensor(theta1, dtype=torch.float32)

# set defaults
lambda0, mu0, nu0 = 0.4, 0.0, 0.2


epochs = 25
batch_size = 512

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def run_classifier(x1, x2, x3, fit_type='generator'):

    if fit_type == 'generator':
        theta0_G = theta0[:, :5]
        weight0_G = cal_weight(lambda0, mu0, nu0, theta0[:, 3], theta0[:, 4]).reshape(-1, 1)
        label0_G = torch.zeros(theta0_G.shape[0], dtype=torch.float32).reshape(-1, 1)

        theta1_G = theta1[:, :5]
        weight1_G = cal_weight(x1, x2, x3, theta1[:, 3], theta1[:, 4]).reshape(-1, 1)
        label1_G = torch.ones(theta1_G.shape[0], dtype=torch.float32).reshape(-1, 1)

        theta = torch.cat((theta0_G, theta1_G), 0)
        weight = torch.cat((weight0_G, weight1_G), 0)
        label = torch.cat((label0_G, label1_G), 0)

    elif fit_type == 'detector':
        theta0_S = theta0[:, 5:]
        weight0_S = cal_weight(lambda0, mu0, nu0, theta0[:, 3], theta0[:, 4]).reshape(-1, 1)
        label0_S = torch.zeros(theta0_S.shape[0], dtype=torch.float32).reshape(-1, 1)

        theta1_S = theta1[:, 5:]
        weight1_S = cal_weight(x1, x2, x3, theta1[:, 3], theta1[:, 4]).reshape(-1, 1)
        label1_S = torch.ones(theta1_S.shape[0], dtype=torch.float32).reshape(-1, 1)

        theta = torch.cat((theta0_S, theta1_S), 0)
        weight = torch.cat((weight0_S, weight1_S), 0)
        label = torch.cat((label0_S, label1_S), 0)
    else:
        raise ValueError("fit type must be set to 'generator' or 'detector'")

    dataset = TensorDataset(theta, weight, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()

    loss = []

    for epoch in range(epochs):
        net.train()
        run_loss, m = 0., 0.
        for inputs, weights, targets in dataloader:
            optimizer.zero_grad()

            outputs = net(inputs)

            criterion.weight = weights
            loss_batch = criterion(outputs, targets)

            loss_batch.backward()
            optimizer.step()

            run_loss += loss_batch.item()
            m += 1.0

        loss.append(run_loss/m)
        # print("epoch : {} loss : {}".format(epoch + 1, run_loss / m))

    # plt.figure(figsize=(6, 5))
    # plt.plot(loss)
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.savefig("../imgs/loss.png")
    # plt.close("all")

    net.eval()

    outputs = net(theta).detach().numpy()
    auc = roc_auc_score(label, outputs, sample_weight=weight)
    return auc


thetas = np.linspace(-1., 1., 11)

AUC_G = []
AUC_S = []


def scan_theta():
    for theta in thetas:
        print("*** generator level ***")
        print("Testing theta = {:.1f}".format(theta))

        auc = run_classifier(lambda0, mu0, theta, "generator")
        AUC_G.append(auc)

        print("AUC: {:.2f}".format(auc))
        print("\n")
        pass

        print("*** detector level ***")
        print("Testing theta = {:.1f}".format(theta))

        auc = run_classifier(lambda0, mu0, theta, "detector")
        AUC_S.append(auc)

        print("AUC: {:.2f}".format(auc))
        print("\n")
        pass


def get_AUC_G(x):
    # print("=== theta_test = {:.2f} ===".format(x[0]))
    AUC_G = run_classifier(lambda0, mu0, x[0], "generator")
    return AUC_G


def get_AUC_S(x):
    # print("=== theta_test = {:.2f} ===".format(x[0]))
    AUC_S = run_classifier(lambda0, mu0, x[0], "detector")
    return AUC_S


# restrict optimization to bounds of parameterization of the reweighting function
mybounds = Bounds(np.array([-1.]), np.array([1.]))

THETA_G = []
FUNC_G = []
THETA_S = []
FUNC_S = []


def run_minimizer():
    for i in range(30):
        theta_prime = np.random.uniform(-1., 1., 1)
        print("iteration : {} theta prime = {:.2f}".format(i+1, theta_prime[0]))
        res = minimize(get_AUC_G, theta_prime, method="Powell", bounds=mybounds, options={
                                                                            'maxiter': 50,
                                                                            'disp': True,
                                                                            'return_all': True
                                                                            })

        THETA_G.append(res["x"][0])
        FUNC_G.append(res["fun"])

        print("best theta : ({:.2f})".format(res["x"][0]))
        print("\n")

        res = minimize(get_AUC_S, theta_prime, method="Powell", bounds=mybounds, options={
                                                                            'maxiter': 50,
                                                                            'disp': True,
                                                                            'return_all': True
                                                                            })

        THETA_S.append(res["x"][0])
        FUNC_S.append(res["fun"])

        print("best theta : ({:.2f})".format(res["x"][0]))
        print("\n")


opt_scan = torch.compile(scan_theta, mode="reduce-overhead")
opt_minimizer = torch.compile(run_minimizer, mode="reduce-overhead")


# run compiled functions
opt_scan()
opt_minimizer()


print("best fit generator level = {:.2f} +/- {:.2f}".format(np.mean(THETA_G), np.std(THETA_G)))
print("best fit detector level = {:.2f} +/- {:.2f}".format(np.mean(THETA_S), np.std(THETA_S)))

plt.figure(figsize=(6, 5))
plt.plot(thetas, AUC_G, label='Generator-level')
plt.plot(thetas, AUC_S, label='Detector-level')
plt.errorbar(np.mean(THETA_G), 0.51, xerr=np.std(THETA_G), fmt="o", label=r'$\theta_G = {:.2f} \pm {:.2f}$'.format(np.mean(THETA_G), np.std(THETA_G)))
plt.errorbar(np.mean(THETA_S), 0.51, xerr=np.std(THETA_S), fmt="o", label=r'$\theta_S = {:.2f} \pm {:.2f}$'.format(np.mean(THETA_S), np.std(THETA_S)))
plt.xlabel(r'$\nu$')
plt.ylabel('AUC')
plt.legend(loc='lower left', frameon=False)
plt.savefig("../imgs/NNFit1D.png")
plt.close("close")
