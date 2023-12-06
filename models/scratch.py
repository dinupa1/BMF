import torch
import torch.nn as nn

m1 = nn.Conv2d(3, 16, kernel_size=2, padding=0)
m2 = nn.ConvTranspose2d(16, 6, kernel_size=2, padding=0)

x = torch.rand(3, 12, 12)
print(x.shape)

y1 = m1(x)
print(y1.shape)

y2 = m2(y1)
print(y2.shape)