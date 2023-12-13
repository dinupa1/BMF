import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import ParamExtractor

from sklearn.utils import resample

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)

batch_size = 1024
learning_rate = 0.0001
num_epochs = 200
step_size = 50
gamma = 0.1
n_steps = 50

tree = torch.load("unet-tensor.pt")

train_tree = tree["train_tree"]
val_tree = tree["val_tree"]
test_tree = tree["test_tree"]

X_par, X_preds = [], []

for i in range(n_steps):

	print("---> step {}".format(i))

	X_train, y_train = resample(train_tree["X_det"].numpy(), train_tree["X_par"].numpy(), n_samples=65000, replace=False)
	X_val, y_val = resample(val_tree["X_det"].numpy(), val_tree["X_par"].numpy(), n_samples=25000, replace=False)

	train_dic = {
	"X_det": torch.from_numpy(X_train),
	"X_par": torch.from_numpy(y_train),
	}

	val_dic = {
	"X_det": torch.from_numpy(X_val),
	"X_par": torch.from_numpy(y_val),
	}

	model = ParamExtractor(learning_rate, step_size, gamma)
	model.train(train_dic, val_dic, batch_size, num_epochs, device)

	model.network.to("cpu")

	model.network.eval()
	with torch.no_grad():
		outputs = model.network(test_tree["X_det"][:5])
		X_par.append(test_tree["X_par"][:5].numpy())
		X_preds.append(outputs.view(outputs.size(0), 4, 3).detach().numpy())

save = {
"X_par_mean": torch.from_numpy(np.mean(X_par, axis=0)).float(),
"X_par_std": torch.from_numpy(np.std(X_par, axis=0)).float(),
"X_pred_mean": torch.from_numpy(np.mean(X_preds, axis=0)).float(),
"X_pred_std": torch.from_numpy(np.std(X_preds, axis=0)).float(),
}

torch.save(save, "examples.pt")