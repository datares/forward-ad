#!/usr/bin/env python3
import fwad
import torch

model = fwad.ForwardModule(torch.nn.Linear(2, 2))

x = torch.rand(2, 2)
y = model(x)
print(y)
