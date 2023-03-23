import uproot
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from Model import Generator, Discriminator, data_set

netG = Generator()
netD = Discriminator()

criterionG = nn.MSELoss()
criterionD = nn.BCELoss()

batch_size = 32

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.001)
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.001)

train_tree = uproot.open("hist.root:train")

train_label = train_tree.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
train_hist = train_tree["hist"].array().to_numpy()

train_dataset = data_set(torch.Tensor(train_hist).unsqueeze(1), torch.Tensor(train_label))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

real_label = 1.
fake_label = 0.

num_epochs = 20

real_losses = []
fake_losses = []
gen_losses = []
gen_real = []

for epoch in range(num_epochs):
    netD.train()
    netG.train()
    fakeD = 0.
    realD = 0.
    lossG = 0.
    realG = 0.
    m = 0.
    for inputs, targets in train_dataloader:
        # -------------------------
        # (1) train discriminator
        # -------------------------

        # train real batch
        optimizerD.zero_grad()
        b_size = inputs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float)
        output = netD(targets).view(-1)
        errD_real = criterionD(output, label)
        errD_real.backward()

        # train fake batch
        fake = netG(inputs)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterionD(output, label)
        errD_fake.backward()

        optimizerD.step()

        # --------------------
        # (2) train generator
        # --------------------
        optimizerG.zero_grad()
        output = netG(inputs)
        errG_loss = criterionG(output, targets)
        errG_loss.backward()

        label.fill_(real_label)
        output = netD(fake).view(-1)

        errG_real = criterionD(output, label)
        errG_real.backward()

        optimizerG.step()
        # optimizerD.step()


        fakeD += errD_fake.item()
        realD += errD_real.item()
        lossG += errG_loss.item()
        realG += errG_real.item()
        m += 1

    real_losses.append(realD/m)
    fake_losses.append(fakeD/m)
    gen_losses.append(lossG/m)
    gen_real.append(realG/m)

    print("epoch: {} real loss: {} fake loss: {} generator loss: {}".format(epoch+1, realD/m, fakeD/m, realG/m))


plt.plot(real_losses, label="real loss")
plt.plot(fake_losses, label="fake loss")
plt.plot(gen_losses, label="reg. loss")
plt.plot(gen_real, label="gen. real")
plt.xlabel("epoch")
plt.ylabel("error")
plt.legend()
plt.savefig("imgs/error.png")
plt.close("all")

import ROOT
from ROOT import TH1D, TCanvas

# test histograms
test_tree = uproot.open("hist.root:test1")

test_label_1 = test_tree.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
test_hist_1 = test_tree["hist"].array().to_numpy()

output = netG(torch.Tensor(test_hist_1).unsqueeze(1)).detach().numpy()


hist1 = TH1D("hist1", "; lambda; count [a.u.]", 10, -1., 1.)
hist2 = TH1D("hist2", "; mu; count [a.u.]", 10, -0.4, 0.2)
hist3 = TH1D("hist3", "; nu; count [a.u.]", 10, -0.2, 0.4)

[hist1.Fill(m) for m in output[:, 0]]
[hist2.Fill(m) for m in output[:, 1]]
[hist3.Fill(m) for m in output[:, 2]]

can = TCanvas()

hist1.Draw()
can.SaveAs("imgs/lambda1.png")

hist2.Draw()
can.SaveAs("imgs/mu1.png")

hist3.Draw()
can.SaveAs("imgs/nu1.png")

test_tree2 = uproot.open("hist.root:test2")

test_label_2 = test_tree2.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
test_hist_2 = test_tree2["hist"].array().to_numpy()

output = netG(torch.Tensor(test_hist_2).unsqueeze(1)).detach().numpy()


hist4 = TH1D("hist4", "; lambda; count [a.u.]", 10, -1., 1.)
hist5 = TH1D("hist5", "; mu; count [a.u.]", 10, -1., 1.)
hist6 = TH1D("hist6", "; nu; count [a.u.]", 10, -1., 1.)

[hist4.Fill(m) for m in output[:, 0]]
[hist5.Fill(m) for m in output[:, 1]]
[hist6.Fill(m) for m in output[:, 2]]


hist4.Draw()
can.SaveAs("imgs/lambda2.png")

hist5.Draw()
can.SaveAs("imgs/mu2.png")

hist6.Draw()
can.SaveAs("imgs/nu2.png")

test_tree3 = uproot.open("hist.root:test3")

test_label_3 = test_tree3.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
test_hist_3 = test_tree3["hist"].array().to_numpy()

output = netG(torch.Tensor(test_hist_3).unsqueeze(1)).detach().numpy()

hist7 = TH1D("hist7", "; lambda; count [a.u.]", 10, -1., 1.)
hist8 = TH1D("hist8", "; mu; count [a.u.]", 10, -1., 1.)
hist9 = TH1D("hist9", "; nu; count [a.u.]", 10, -1., 1.)

[hist7.Fill(m) for m in output[:, 0]]
[hist8.Fill(m) for m in output[:, 1]]
[hist9.Fill(m) for m in output[:, 2]]


hist7.Draw()
can.SaveAs("imgs/lambda3.png")

hist8.Draw()
can.SaveAs("imgs/mu3.png")

hist9.Draw()
can.SaveAs("imgs/nu3.png")

test_tree4 = uproot.open("hist.root:test4")

test_label_4 = test_tree4.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
test_hist_4 = test_tree4["hist"].array().to_numpy()


output = netG(torch.Tensor(test_hist_4).unsqueeze(1)).detach().numpy()

hist10 = TH1D("hist10", "; lambda; count [a.u.]", 10, -1., 1.)
hist11 = TH1D("hist11", "; mu; count [a.u.]", 10, -1., 1.)
hist12 = TH1D("hist12", "; nu; count [a.u.]", 10, -1., 1.)

[hist10.Fill(m) for m in output[:, 0]]
[hist11.Fill(m) for m in output[:, 1]]
[hist12.Fill(m) for m in output[:, 2]]


hist10.Draw()
can.SaveAs("imgs/lambda4.png")

hist11.Draw()
can.SaveAs("imgs/mu4.png")

hist12.Draw()
can.SaveAs("imgs/nu4.png")