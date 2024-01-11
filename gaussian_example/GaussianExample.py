import numpy as np
import matplotlib.pyplot as plt

import torch

from models import GaussFit


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for CUDA

plt.rc("font", size=14)

tensor_tree = torch.load("gauss_tensor.pt")
X0_train_tree = tensor_tree["X0_train_tree"]
X1_train_tree = tensor_tree["X1_train_tree"]
X0_test_tree = tensor_tree["X0_test_tree"]
X1_test_tree = tensor_tree["X1_test_tree"]

hidden_dim = 64
batch_size = 1024
learning_rate = 0.001
num_epochs =200
step_size = 20
early_stopping_patience = 20
gamma = 0.1
num_runs = 50

model = GaussFit(hidden_dim, learning_rate, step_size, gamma, batch_size)
best_model_weights = model.train(X0_train_tree, X1_train_tree, num_epochs, device, early_stopping_patience)
model.scan(best_model_weights, X0_test_tree, X1_test_tree)
model.reweight(best_model_weights, X0_test_tree, X1_test_tree)
model.fit(best_model_weights, X0_test_tree, X1_test_tree, num_runs, 1000)
