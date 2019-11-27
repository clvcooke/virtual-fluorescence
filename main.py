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
    torch.set_num_threads(1)
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
    # get data-loaders
    # create a model
    model = Model(config.num_heads, config.num_channels, batch_norm=config.batch_norm, skip=config.skip,
                  initilization_strategy=config.init_strategy, num_filters=config.num_filters)
    if config.use_gpu:
        model.cuda()
        [unet.cuda() for unet in model.unets]
    params = list(model.parameters())
    for unet in model.unets:
        params += list(unet.parameters())
    # setup optimizer
    optimizer = torch.optim.Adam(params, lr=config.init_lr)

    train_dataset, val_dataset = get_train_val_loader(config, pin_memory=True)

    trainer = Trainer(model, optimizer, train_dataset, val_dataset, config)
    wandb.config.update(config)
    trainer.train()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
