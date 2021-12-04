import torch
from models.model import Model

model = Model('config/resnet.yaml')
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)



