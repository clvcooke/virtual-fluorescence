import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='CTC')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=4,
                      help='# of images in each batch of data')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the train and valid indices')
data_arg.add_argument('--level', type=int, default=2, help='# of levels of mask to use')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=50,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=1e-3,
                       help='Initial learning rate value')
train_arg.add_argument('--lr_patience', type=int, default=10,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=1000,
                       help='Number of epochs to wait before stopping train')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
