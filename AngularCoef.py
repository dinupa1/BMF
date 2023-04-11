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

batch_size = 64

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.001)
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.001)

train_tree = uproot.open("hist.root:train_xF3")

train_label = train_tree.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
train_hist = train_tree["hist"].array().to_numpy()

train_dataset = data_set(torch.Tensor(train_hist).unsqueeze(1), torch.Tensor(train_label))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

real_label = 1.
fake_label = 0.

num_epochs = 10

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


netG.eval()

# test histograms
test_tree = uproot.open("hist.root:test_xF3")

test_label_1 = test_tree.arrays(["lambda", "mu", "nu"], library="pd").to_numpy()
test_hist_1 = test_tree["hist"].array().to_numpy()

output = netG(torch.Tensor(test_hist_1).unsqueeze(1)).detach().numpy()

test1 = {
    "lambda": output[:, 0],
    "mu": output[:, 1],
    "nu": output[:, 2],
}

output = uproot.recreate("result.root", compression=uproot.ZLIB(4))
output["test1"] = test1
output.close()
