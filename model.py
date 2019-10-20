import torch
import torch.nn as nn

from modules import IlluminationLayer
from unet import UNet


class Model(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.illumination_layer = IlluminationLayer(675)
        self.unets = [UNet(1, 16) for _ in range(self.num_heads)]

    def forward(self, x):
        illuminated_image = self.illumination_layer(x)
        results = [unet(illuminated_image) for unet in self.unets]
        return torch.stack(results)