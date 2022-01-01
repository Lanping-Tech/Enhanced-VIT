import torch
import torchvision
import yaml
from models import ViT
from models.resvit import ResViT
from models.rest import ResTNet

class Model(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.load_config(model_config)

        if self.config['model_name'] == 'vit':
            self.model = ViT(**self.config['model_args'])
        elif self.config['model_name'] == 'resvit':
            self.model = ResViT(**self.config['model_args'])
        elif self.config['model_name'] == 'restnet':
            self.model = ResTNet(**self.config['model_args'])
        else:
            resnet_model_func = getattr(torchvision.models, self.config['model_name'])
            self.model = resnet_model_func(**self.config['model_args'])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def load_config(self, model_config):
        with open(model_config, 'r') as config:
            self.config = yaml.load(config, Loader=yaml.SafeLoader)