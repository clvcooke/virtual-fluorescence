from model import Model
import torch
import glob
import os
import numpy as np
from tqdm import tqdm


def predict(run_id, level):
    model_path = f'/hddraid5/data/colin/ctc/models/model_{run_id}.pth'
    unet_path = f'/hddraid5/data/colin/ctc/models/unet_0_{run_id}.pth'
    model = Model(num_heads=1)
    model_state = torch.load(model_path)
    unet_state = torch.load(unet_path)
    model.load_state_dict(model_state)
    model.unets[0].load_state_dict(unet_state)

    # for now we will only ever load test data
    data_dir = '/hddraid5/data/colin/'
    predictions_dir = os.path.join(data_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    test_data_path = os.path.join(data_dir, f'test_x_norm.npy')
    test_x = torch.from_numpy(np.load(test_data_path, mmap_mode='r'))
    test_amnt = test_x.shape[0]
    batch_size = 4
    indices = np.arange(0, test_amnt, batch_size)
    for index in tqdm(indices):
        test_x_batch = test_x[index:index + batch_size]
        with torch.no_grad():
            predictions = model(test_x_batch).cpu().numpy()
        out_path = os.path.join(predictions_dir, f'level_{level}_r{run_id}_i{index}.npy')
        np.save(os.path.join(predictions_dir, f'level_{level}_i{index}_x.npy'), test_x_batch.numpy())
        np.save(out_path, predictions.reshape(batch_size, 256, 256))
        break


if __name__ == "__main__":
    predict('uzyxesg9', 64)
