import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

m = nn.ReLU()
x = torch.linspace(-10, 10, 50)
y = m(x)


plt.plot(x.numpy(), y.numpy())
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("ReLU()")
plt.savefig("imgs/ReLU.png")
plt.close("all")

m = nn.Sigmoid()
y = m(x)

plt.plot(x.numpy(), y.numpy())
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Sigmoid()")
plt.savefig("imgs/Sigmoid.png")
plt.close("all")

