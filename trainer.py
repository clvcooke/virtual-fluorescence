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
#        self.criterion = torch.nn.L1Loss()
        if str(config.task).lower() == 'mnist':
            self.criterion = torch.nn.CrossEntropyLoss()
            self.classification = True
        else:
            self.criterion = torch.nn.MSELoss()
            self.classification = False
        self.num_train = len(self.train_loader.sampler.indices)
        self.num_valid = len(self.val_loader.sampler.indices)
        self.lr = self.config.init_lr
        self.curr_epoch = 0
        wandb.watch(self.model, log='all')

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, validate on {self.num_valid} samples")
        best_val_loss = np.inf
        epochs_since_best = 0
        for epoch in range(self.config.epochs):
            self.curr_epoch = epoch
            print(f'\nEpoch {epoch}/{self.config.epochs} -- lr = {self.lr}')

            if self.classification:
                train_loss, train_acc = self.run_one_epoch(training=True)
                val_loss, val_acc = self.run_one_epoch(training=False)
                msg = f'train loss {train_loss:.3f} train acc {train_acc:.3f} -- val loss {val_loss:.3f} val acc {val_acc:.3f}'
            else:
                train_loss, _ = self.run_one_epoch(training=True)
                val_loss, _ = self.run_one_epoch(training=False)
                msg = f'train loss {train_loss:.3f} -- val loss {val_loss:.3f}'

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.model.save_model(verbose=True)
            epochs_since_best = (epochs_since_best + 1) * (1 - is_best)

            if is_best:
                msg += ' [*]'
            print(msg)
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss
            }, step=epoch)
            if epochs_since_best > self.config.lr_patience:
                epochs_since_best = 0
                self.lr = self.lr / np.sqrt(10)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def run_one_epoch(self, training):
        tic = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        if training:
            amnt = self.num_train
            dataset = self.train_loader
        else:
            dataset = self.val_loader
            amnt = self.num_valid
        with tqdm(total=amnt) as pbar:
            for i, data in enumerate(dataset):
                x, y = data
                # segmentation task
                if self.classification:
                    # assuming one-hot
                    y = y.view(1, -1).expand(self.model.num_heads, -1)
                else:
                    y = y.view(1, -1, 1, x.shape[-2], x.shape[-1]).expand(self.model.num_heads, -1, -1, -1, -1)
                if self.config.use_gpu:
                    x, y = x.cuda(), y.cuda()
                output = self.model(x)
                if training:
                    self.optimizer.zero_grad()
                loss = None

                for head in range(self.model.num_heads):
                    if loss is None:
                        loss = self.criterion(output[head], y[head])
                    else:
                        loss = loss + self.criterion(output[head], y[head])
                loss = loss / self.model.num_heads
                if training:
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
                if self.classification:
                    _, predicted = torch.max(output.data, -1)
                    total = self.batch_size*self.model.num_heads
                    correct = (predicted == y).sum().item()
                    acc = correct/total
                    accs.update(acc)
                    pbar.set_description(f"{(toc - tic):.1f}s - loss: {loss_data:.3f} acc {accs.avg:.3f}")
                else:
                    pbar.set_description(f"{(toc - tic):.1f}s - loss: {loss_data:.3f}")
                pbar.update(self.batch_size)
                if training and i % 2 == 0:
                    self.model.log_illumination(self.curr_epoch, i)
                if not training and i == 0 and not self.classification:
                    y_sample = y[0, 0].view(256, 256).detach().cpu().numpy()
                    p_sample = output[0, 0].view(256, 256).detach().cpu().numpy()
                    wandb.log({f"images_epoch{self.curr_epoch}": [
                        wandb.Image(np.round(p_sample * 255), caption="prediction"),
                        wandb.Image(np.round(y_sample * 255), caption="label")]}, step=self.curr_epoch)
        return losses.avg, accs.avg
