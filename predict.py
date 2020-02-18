from model import Model
import torch
import glob
import os
import numpy as np
from tqdm import tqdm


def predict(run_id, level, task):
    model_path = f'/hddraid5/data/colin/ctc/models/model_{run_id}.pth'
    unet_path = f'/hddraid5/data/colin/ctc/models/net_0_{run_id}.pth'
    model = Model(num_heads=1, batch_norm=True)
    model_state = torch.load(model_path)
    unet_state = torch.load(unet_path)
    model.load_state_dict(model_state)
    model.nets[0].load_state_dict(unet_state)

    # for now we will only ever load test data
    data_dir = '/hddraid5/data/colin/ctc/'
    predictions_dir = os.path.join(data_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    if task.lower() == 'hela':
        test_data_path = os.path.join(data_dir, f'test_x_norm.npy')
    elif task.lower() == 'pan':
        test_data_path = os.path.join(data_dir, 'pan_test_x.npy')
        pass
    else:
        raise RuntimeError
    test_x = torch.from_numpy(np.load(test_data_path, mmap_mode='r'))
    test_amnt = test_x.shape[0]
    batch_size = 4
    indices = np.arange(0, test_amnt, batch_size)
    for index in tqdm(indices):
        test_x_batch = test_x[index:index + batch_size].float()
        with torch.no_grad():
            predictions = model(test_x_batch).cpu().numpy()
        out_path = os.path.join(predictions_dir, f'level_{level}_r{run_id}_i{index//batch_size}.npy')
        # np.save(os.path.join(predictions_dir, f'level_{level}_i{index//batch_size}_x.npy'), test_x_batch.numpy())
        np.save(out_path, predictions.reshape(batch_size, 256, 256))
        break


if __name__ == "__main__":
    predict('0aa6qqix', 16, 'pan')



