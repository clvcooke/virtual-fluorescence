import torch
import torch.nn as nn
import numpy as np

from utils import from_spiral, to_spiral


class IlluminationLayer(nn.Module):

    def __init__(self, k_depth, num_channels, init_strategy, init_params=None):
        super().__init__()
        self.physical_layer = nn.Conv2d(k_depth, num_channels, kernel_size=1, stride=1, bias=False)
        if init_strategy is not None and init_strategy != "learned":
            # we assume that if we are intitializing then we don't want to train
            for param in self.physical_layer.parameters():
                param.requires_grad = False
            if init_strategy == 'center':
                self.init_center()
            elif init_strategy == 'all':
                self.init_all()
            elif init_strategy == 'brightfield':
                self.init_brightfield(4, init_params)
            elif init_strategy == 'dpc':
                self.init_dpc()
            elif init_strategy == 'random':
                # don't do anything, weights are already random
                pass
            elif init_strategy == 'off_axis':
                self.init_off_axis()
            else:
                # don't allow unspecified strategies to silently fail
                raise RuntimeError

    def init_off_axis(self):
        pattern = np.zeros((15, 15))
        # off axis will be two off center?
        pattern[7, 9] = 1
        pattern = to_spiral(pattern).flatten()
        for i in range(3):
            for j in range(225):
                idx = i * 225 + j
                self.physical_layer.weight[0, idx] = pattern[j]

    def init_center(self):
        for i in range(675):
            if i % 225 == 0:
                self.physical_layer.weight[0, i] = 1
            else:
                self.physical_layer.weight[0, i] = 0

    def init_all(self):
        for i in range(675):
            self.physical_layer.weight[0, i] = 1 / 675

    def init_dpc(self, vertical=False):
        pattern = np.zeros((15, 15))
        # first 5  = -1
        # next  5  = 0
        # last  5  = 1
        pattern[0:5, :] = -1
        pattern[5:10, :] = 0
        pattern[10:, :] = 1
        if vertical:
            pattern = pattern.T
        pattern = from_spiral(pattern).flatten()

        for i in range(3):
            for j in range(225):
                idx = i * 225 + j
                self.physical_layer.weight[0, idx] = pattern[j]

    def init_brightfield(self, cutoff=4, brightfield_value=None, darkfield_value=0):
        '''

        :param cutoff: distance from center LED which qualifies as brightfield (cutoff = 0 is only center LED)
        :param brightfield_value: value to assign to brightfield LEDs (default to 1/(cutoff*cutoff))
        :param darkfield_value: value to assign to darkfield LEDs (default 0)
        :return:
        '''
        pattern = np.zeros((15, 15))
        if brightfield_value is None:
            brightfield_value = 1 / (cutoff * cutoff)
        pattern[:, :] = darkfield_value
        for i in range(15):
            for j in range(15):
                distance_x = np.abs(7 - i)
                distance_y = np.abs(7 - j)
                distance = np.sqrt(distance_x ** 2 + distance_y ** 2)
                if distance <= cutoff:
                    pattern[i, j] = brightfield_value
        for i in range(cutoff):
            start = 7 - 1
            end = 7 + i + 1
            pattern[start:end, start:end] = brightfield_value
        pattern = from_spiral(pattern).flatten()
        for i in range(3):
            for j in range(225):
                idx = i * 225 + j
                self.physical_layer.weight[0, idx] = pattern[j]

    def forward(self, x):
        return self.physical_layer(x)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.active = sigma > 0.0
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float().to("cuda")

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x