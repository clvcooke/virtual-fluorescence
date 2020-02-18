import numpy as np
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        if data_y.dtype is torch.float16:
            self.convert = True
        else:
            self.convert = False

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]

        if self.convert:
            return image.float(), label.float()
        else:
            return image.float(), label


def load_progress(path, desc=''):
    try:
        mmap_array = np.load(path, mmap_mode='r')
        array = np.empty_like(mmap_array, dtype=np.float16)
        block_size = 2
        n_blocks = int(np.ceil(mmap_array.shape[0] / block_size))
        for b in tqdm(range(n_blocks), desc=desc):
            array[b * block_size:(b + 1) * block_size] = mmap_array[b * block_size:(b + 1) * block_size]
    finally:
        del mmap_array
    return array


def shift_data(shift_code, data, image_shape=[675, 256, 256]):
    if shift_code == '':
        return data
    # if we are going to shift  we can use N S E W along with a number (1-10)
    direction = shift_code[0].upper()
    amnt = int(shift_code[1:])
    assert direction in ['N', 'S', 'E', 'W']
    # we need to do a reshape first
    data = np.reshape(data, list(data.shape[0:1]) + [3, 15, 15, 256, 256])
    # now we need to cutoff the data on the correct axis
    if direction == 'S':
        data = data[:, :, amnt:]
        data = np.pad(data, pad_width=[[0, 0], [0, 0], [amnt, 0], [0, 0], [0, 0], [0, 0]])
    elif direction == 'N':
        data = data[:, :, :-amnt]
        data = np.pad(data, pad_width=[[0, 0], [0, 0], [0, amnt], [0, 0], [0, 0], [0, 0]])
    elif direction == 'E':
        data = data[:, :, :, :-amnt]
        data = np.pad(data, pad_width=[[0, 0], [0, 0], [0, 0], [0, amnt], [0, 0], [0, 0]])
    else:
        data = data[:, :, :, amnt:]
        data = np.pad(data, pad_width=[[0, 0], [0, 0], [0, 0], [amnt, 0], [0, 0], [0, 0]])
    # reshape the data back to the proper format
    data = np.reshape(data, [-1] + image_shape)
    return data


def get_train_val_loader(config, pin_memory, num_workers=1):
    data_dir = '/hddraid5/data/colin/'

    if str(config.task).lower() == 'malaria':
        batch_size = config.batch_size
        # save prev seed
        seed = config.random_seed
        # split data based on constant zero seed
        np.random.seed(0)
        train_x_path = '/content/malaria_norm.npy'
        train_y_path = '/content/malaria_labels.npy'
        train_split = 0.8
        x_data = load_progress(train_x_path, 'loading x data')
        y_data = load_progress(train_y_path, 'loading y data')
        y_data = y_data.astype(np.int)
        # (1021, 28, 28, 96)
        x_data = np.swapaxes(x_data, 1, 3).astype(np.float32)
        # (1021, 96, 28, 28)
        x_data = x_data / 255
        amnt = x_data.shape[0]
        train_amnt = int(amnt * train_split)
        indices = np.arange(0, amnt)
        np.random.shuffle(indices)
        train_indices = indices[:train_amnt]
        val_indices = indices[train_amnt:]
        train_x = torch.from_numpy(x_data[train_indices])
        train_y = torch.from_numpy(y_data[train_indices])
        val_x = torch.from_numpy(x_data[val_indices])
        val_y = torch.from_numpy(y_data[val_indices])
        # re-seed with specified seed
        np.random.seed(seed)
    elif str(config.task).lower() == 'mnist':
        batch_size = config.batch_size
        # save prev seed
        seed = config.random_seed
        # split data based on constant zero seed
        np.random.seed(0)
        train_x_path = os.path.join(data_dir, 'k-space-rl', 'train_data_norm_v2.npy')
        train_y_path = os.path.join(data_dir, 'k-space-rl', 'train_labels.npy')
        train_split = 0.8
        x_data = load_progress(train_x_path, 'loading x data')
        y_data = load_progress(train_y_path, 'loading y data')
        y_data = np.argmax(y_data, axis=-1).astype(np.int)
        # 784 60k 25
        x_data = np.swapaxes(x_data, 0, 1)
        # 60k 784 25
        x_data = np.swapaxes(x_data, 1, 2)
        # 60K 25 784
        x_data = x_data.reshape((-1, 25, 28, 28))
        amnt = x_data.shape[0]
        train_amnt = int(amnt * train_split)
        indices = np.arange(0, amnt)
        np.random.shuffle(indices)
        train_indices = indices[:train_amnt]
        val_indices = indices[train_amnt:]
        train_x = torch.from_numpy(x_data[train_indices])
        train_y = torch.from_numpy(y_data[train_indices])
        val_x = torch.from_numpy(x_data[val_indices])
        val_y = torch.from_numpy(y_data[val_indices])
        # re-seed with specified seed
        np.random.seed(seed)
    else:
        level, batch_size = config.level, config.batch_size
        bits = int(np.log2(level))
        if str(config.task).lower() == 'hela':
            mmap = False
            train_x_path = os.path.join(data_dir, 'ctc', 'train_x_norm.npy')
            train_y_path = os.path.join(data_dir, 'ctc', f'new_nuc_train_kb{bits}.npy')
            val_x_path = os.path.join(data_dir, 'ctc', 'val_x_norm.npy')
            val_y_path = os.path.join(data_dir, 'ctc', f'new_nuc_val_kb{bits}.npy')
        else:
            mmap = False
            train_x_path = os.path.join(data_dir, 'ctc', 'pan_train_x.npy')
            train_y_path = os.path.join(data_dir, 'ctc', f'pan_train_{bits}_y.npy')
            val_x_path = os.path.join(data_dir, 'ctc', 'pan_val_x.npy')
            val_y_path = os.path.join(data_dir, 'ctc', f'pan_val_{bits}_y.npy')

        # pytorch says channels fist
        if mmap:
            train_x_npy = shift_data(config.shift, np.load(train_x_path, mmap_mode='r'))
            train_x = torch.from_numpy(train_x_npy)
            train_y = torch.from_numpy(np.load(train_y_path, mmap_mode='r'))
            val_x_npy = shift_data(config.shift, np.load(val_x_path, mmap_mode='r'))
            val_x = torch.from_numpy(val_x_npy)
            val_y = torch.from_numpy(np.load(val_y_path, mmap_mode='r'))
        else:
            train_x_npy = shift_data(config.shift, load_progress(train_x_path, 'loading train x'))
            train_x = torch.from_numpy(train_x_npy)
            train_y = torch.from_numpy(load_progress(train_y_path, 'loading train_y'))
            val_x_npy = shift_data(config.shift, load_progress(val_x_path, 'loading val_x'))
            val_x = torch.from_numpy(val_x_npy)
            val_y = torch.from_numpy(load_progress(val_y_path, 'loading val_y'))

    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)

    train_idx, valid_idx = list(range(train_x.shape[0])), list(range(val_x.shape[0]))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader
