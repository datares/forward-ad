#!/usr/bin/env python3
import fwad
import torch
import torch.nn.functional as F

model = fwad.ForwardModule(torch.nn.Linear(2, 2))

x = torch.rand(2, 2)
y = model(x)
print(y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 100)
        self.fc2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.flatten(1)))
        x = self.fc2(x)
        return x

model = Net()
x = torch.rand(4, 784)
y = model(x)
print(y)
