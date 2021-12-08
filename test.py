import torch
from models.model import *


model = Model('config/res50vit.yaml')

x = torch.randn(1, 3, 32, 32)
y = model(x)
print(y.shape)