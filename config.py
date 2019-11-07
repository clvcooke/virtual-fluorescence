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
train_arg.add_argument('--lr_patience', type=int, default=5,
                       help='Number of epochs to wait before reducing lr')
train_arg.add_argument('--train_patience', type=int, default=1000,
                       help='Number of epochs to wait before stopping train')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--random_seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--num_channels', type=int, default=1,
                      help='Number of images to form with the physical layer')
misc_arg.add_argument('--num_heads', type=int, default=1,
                      help='Number of models to attach physical layer to')
misc_arg.add_argument('--batch_norm', type=str2bool, default=False,
                      help='To use batchnorm or not ( every layer)')
misc_arg.add_argument('--task', type=str, default='pan',
                      help='Task to train on')
misc_arg.add_argument('--skip', type=str2bool, default=False,
                      help='Skip physical layer and feed directly into neural network')
misc_arg.add_argument('--init_strategy', type=str, default=None,
                      help='initialization strategy for physical layer')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
