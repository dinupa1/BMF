import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import train_step, validation_step, reweighting_fn


class ReweightingModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ReweightingModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class ReweightingLoss(nn.Module):
    def __init__(self):
        super(ReweightingLoss, self).__init__()

    def forward(self, outputs, targets, weights):
        criterion = nn.BCELoss(reduction="none")
        loss = criterion(outputs, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


plt.rc("font", size=14)


hdf="../e906-LH2-data/reweight.hdf5"
batch_size = 512
early_stopping_patience = 20
epochs = 1000
learining_rate = 0.0002


infile = h5py.File(hdf, "r")

branches = ["mass", "pT", "xT", "xB", "xF", "weight"]

train_dic = {}
train_dic_mc = {}
test_dic_mc = {}

for branch in branches:
    train_dic[branch] = np.array(infile["train_tree"][branch])
    train_dic_mc[branch] = np.array(infile["train_tree_mc"][branch])
    test_dic_mc[branch] = np.array(infile["test_tree_mc"][branch])
    
    
mass = np.concatenate((train_dic["mass"], train_dic_mc["mass"])).reshape(-1, 1)
pT = np.concatenate((train_dic["pT"], train_dic_mc["pT"])).reshape(-1, 1)
xT = np.concatenate((train_dic["xT"], train_dic_mc["xT"])).reshape(-1, 1)
xB = np.concatenate((train_dic["xB"], train_dic_mc["xB"])).reshape(-1, 1)
xF = np.concatenate((train_dic["xF"], train_dic_mc["xF"])).reshape(-1, 1)
weight = np.concatenate((train_dic["weight"], train_dic_mc["weight"])).reshape(-1, 1)
label = np.concatenate((1. * np.ones(len(train_dic["mass"])), np.zeros(len(train_dic_mc["mass"])))).reshape(-1, 1)


X_data = np.concatenate((mass, pT, xT, xF), axis=-1)

X_train, X_val, W_train, W_val, Y_train, Y_val = train_test_split(X_data, weight, label, test_size=0.4, shuffle=True)

test_mass = test_dic_mc["mass"].reshape(-1, 1)
test_pT = test_dic_mc["pT"].reshape(-1, 1)
test_xT = test_dic_mc["xT"].reshape(-1, 1)
test_xB = test_dic_mc["xB"].reshape(-1, 1)
test_xF = test_dic_mc["xF"].reshape(-1, 1)
test_weight = test_dic_mc["weight"].reshape(-1, 1)

X_test = np.concatenate((test_mass, test_pT, test_xT, test_xF), axis=-1)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scale = scaler.transform(X_train)
X_val_scale = scaler.transform(X_val)
X_test_scale = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train_scale).float()
Y_train_tensor = torch.from_numpy(Y_train).float()
W_train_tensor = torch.from_numpy(W_train).float()

X_val_tensor = torch.from_numpy(X_val_scale).float()
Y_val_tensor = torch.from_numpy(Y_val).float()
W_val_tensor = torch.from_numpy(W_val).float()

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor, W_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor, W_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

X_test_tensor = torch.from_numpy(X_test_scale).float()

# print("---> train shape {}".format(X_train_tensor.shape))
# print("---> val shape {}".format(X_val_tensor.shape))
# print("---> test shape {}".format(X_test_tensor.shape))

model = ReweightingModel(input_dim=4, hidden_dim=32, output_dim=1)
criterion = ReweightingLoss()
optimizer = optim.Adam(model.parameters(), lr=learining_rate)


model = model.to(device=device)
print("using device : {}".format(device))
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print('total trainable params: {}'.format(total_trainable_params))


best_loss = float('inf')
best_model_weights = None
patience_counter = 0

train_loss, val_loss = [], []

for epoch in range(epochs):
    train_epoch_loss = train_step(model, train_loader, criterion, optimizer, device)
    val_epoch_loss = validation_step(model, val_loader, criterion, device)
    
    if epoch%10 == 0:
        print("Epoch {}: Train Loss = {:.4f}, Val. Loss = {:.4f}".format(epoch, train_epoch_loss, val_epoch_loss))

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        best_model_weights = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= early_stopping_patience:
        print("Early stopping at epoch {}".format(epoch))
        break



fig, axs = plt.subplots(figsize=(8, 8))
axs.plot(train_loss, label="train")
axs.plot(val_loss, label="val.")
axs.set_xlabel("epoch")
axs.set_ylabel("BCE loss")
# axs.set_yscale("log")
axs.legend(frameon=False)
plt.tight_layout()
plt.savefig("imgs/training_loss.png")
plt.close("all")
# plt.show()


model.load_state_dict(best_model_weights)
weights = reweighting_fn(model, X_test_tensor)


fig, axs = plt.subplots(figsize=(8, 8))
axs.hist(weights, bins=np.linspace(0.2, 1.5, 21), alpha=0.5)
axs.set_xlabel("weights")
axs.set_ylabel("counts")
plt.savefig("imgs/weights.png")
plt.close("all")
# plt.show()


outputs = h5py.File("../e906-LH2-data/output.hdf5", "w")

for branch in branches:
    outputs.create_dataset("train_tree/{}".format(branch), data=np.array(infile["train_tree"][branch]))
    outputs.create_dataset("test_tree_mc/{}".format(branch), data=np.array(infile["test_tree_mc"][branch]))
    
outputs.create_dataset("test_tree_mc/weights2", data=weights)

outputs.close()
infile.close()


torch.save(model, "reweighting_model.hdf5")
torch.save(best_model_weights, "reweighting_model_weights.hdf5")