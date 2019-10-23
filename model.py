import os
import torch
import numpy as np
import torch.nn as nn

from modules import IlluminationLayer
from unet import UNet
import wandb
import uuid
import os


class Model(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.illumination_layer = IlluminationLayer(1047)
        self.unets = [UNet(1, 16) for _ in range(self.num_heads)]
        self.run_name = os.path.basename(wandb.run.path)
    def forward(self, x):
        illuminated_image = self.illumination_layer(x)
        results = [unet(illuminated_image) for unet in self.unets]
        return torch.stack(results)

    def log_illumination(self, epoch, step):
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
            torch.save(self, model_path)
            if verbose:
                print(f"Saved model to: {model_path}")