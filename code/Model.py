import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# # for now we use linear layers
# # TODO: use conv. layers
# class Generator(nn.Module):
#     def __init__(self, input_dim=20*20, output_dim=3):
#         super(Generator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128, bias=True),
#             nn.LeakyReLU(0.5, inplace=True),
#             # nn.Tanh(),
#             nn.Linear(128, 64, bias=True),
#             nn.LeakyReLU(0.5, inplace=True),
#             # nn.Tanh(),
#             nn.Linear(64, 32, bias=True),
#             nn.LeakyReLU(0.5, inplace=True),
#             # nn.Tanh(),
#             nn.Linear(32, output_dim)
#         )
#
#     def forward(self, x):
#         x = x.view(-1, 20*20)
#         x = self.model(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 20, bias=True),
            nn.ReLU(),
            # nn.Linear(10, 10, bias=True),
            # nn.ReLU(),
            # nn.Linear(10, 5, bias=True),
            # nn.ReLU(),
            nn.Linear(20, output_dim, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2* 2* 64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.view(-1, 2* 2* 64)
        # print(x.shape)
        x = self.fc(x)
        return x




class data_set(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = torch.Tensor(images)
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

# net_G = Generator()
# net_D = Discriminator()
# print(net_G)
# print(net_D)