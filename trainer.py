import numpy as np
import torch
from tqdm import tqdm
import time
from meters import AverageMeter
import wandb


class Trainer:
    def __init__(self, model, optimizer, training_dataset, validation_dataset, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = training_dataset
        self.val_loader = validation_dataset
        self.config = config
        self.batch_size = self.config.batch_size
        self.criterion = torch.nn.MSELoss()
        self.num_train = len(self.train_loader.sampler.indices)
        self.num_valid = len(self.val_loader.sampler.indices)
        self.lr = self.config.init_lr

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")
        best_val_loss = np.inf
        for epoch in range(self.config.epochs):
            print(f'\nEpoch {epoch}/{self.config.epochs} -- lr = {self.lr}')

            train_loss = self.run_one_epoch(training=True)
            val_loss = self.run_one_epoch(training=False)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            msg = f'train loss {train_loss:.3f} -- val loss {val_loss:.3f}'
            if is_best:
                msg += ' [*]'
            print(msg)
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)
            # TODO: add decay and patience

    def run_one_epoch(self, training):
        tic = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()
        if training:
            amnt = self.num_train
            dataset = self.train_loader
        else:
            dataset = self.val_loader
            amnt = self.num_valid
        with tqdm(total=amnt) as pbar:
            for i, data in enumerate(dataset):
                x, y = data
                # no memcopy
                y = y.view(1, -1, x.shape[0], x.shape[1]).expand(self.model.num_heads, -1, -1, -1, -1)
                output = self.model(x)
                loss = self.criterion(output, y)
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                try:
                    loss_data = loss.data[0]
                except IndexError:
                    loss_data = loss.data.item()
                losses.update(loss_data)
                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                pbar.set_description(f"{(toc - tic):.1f}s - loss: {loss_data:.3f}")
                pbar.update(self.batch_size)
        return losses.avg