import torch.nn as nn


class IlluminationLayer(nn.Module):

    def __init__(self, k_depth, num_channels):
        super().__init__()
        self.physical_layer = nn.Conv2d(k_depth, num_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.physical_layer(x)
