import wandb
import torch

from dataloader import get_train_val_loader
from config import get_config
from model import Model
from trainer import Trainer

wandb.init("CTC")


def main(config):
    # setup
    torch.manual_seed(config.random_seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
    # get data-loaders
    train_dataset, val_dataset = get_train_val_loader(config.level, config.batch_size, pin_memory=True)
    # create a model
    model = Model(1)
    if config.use_gpu:
        model.cuda()
        [unet.cuda() for unet in model.unets]
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr)
    trainer = Trainer(model, optimizer, train_dataset, val_dataset, config)
    trainer.train()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
