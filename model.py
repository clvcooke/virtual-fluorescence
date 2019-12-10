import os
import torch
import numpy as np
import torch.nn as nn

from modules import IlluminationLayer
from unet import UNet
from classifier import Classifier
import wandb
import os


class Model(nn.Module):
    def __init__(self, num_heads, num_channels=1, batch_norm=False, skip=False, initilization_strategy=None,
                 num_filters=16, task='hela'):
        super().__init__()
        self.num_heads = num_heads
        self.skip = skip
        if str(task).lower() == 'mnist':
            if skip:
                raise RuntimeError("We aren't testing this!")
            else:
                self.illumination_layer = IlluminationLayer(25, num_channels, initilization_strategy)
                self.nets = [Classifier(10, num_channels, batch_norm=batch_norm) for _ in range(self.num_heads)]
        else:
            if not skip:
                self.illumination_layer = IlluminationLayer(675, num_channels, initilization_strategy)
                self.nets = [UNet(1, num_filters, num_channels, batch_norm=batch_norm) for _ in range(self.num_heads)]
            else:
                self.nets = [UNet(1, num_filters, 675, batch_norm=batch_norm) for _ in range(self.num_heads)]
        try:
            self.run_name = os.path.basename(wandb.run.path)
        except:
            pass

    def forward(self, x):
        if self.skip:
            illuminated_image = x
        else:
            illuminated_image = self.illumination_layer(x)
        results = [net(illuminated_image) for net in self.nets]
        return torch.stack(results)

    def log_illumination(self, epoch, step):
        if self.skip:
            return
        # extract the illumination layers weight
        weight = self.illumination_layer.physical_layer.weight.detach().cpu().numpy()
        # save the weights
        weight_path = os.path.join('/hddraid5/data/colin/ctc/patterns', f'epoch_{epoch}_step_{step}.npy')
        np.save(weight_path, weight)

    def save_model(self, file_path=None, verbose=False):
        # if no path given try to get path from W&B
        # if that fails use a UUID
        if file_path is None:
            base_folder = '/hddraid5/data/colin/ctc/models'
            os.makedirs(base_folder, exist_ok=True)
            model_path = os.path.join(base_folder, f'model_{self.run_name}.pth')
            if not self.skip:
                torch.save(self.state_dict(), model_path)
            for u in range(self.num_heads):
                net_path = os.path.join(base_folder, f'net_{u}_{self.run_name}.pth')
                torch.save(self.nets[u].state_dict(), net_path)
                if verbose:
                    print("saved unet to : " + net_path)
            if verbose:
                print(f"Saved model to: {model_path}")
