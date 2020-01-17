from model import Model
import torch
from skimage import measure
import os
import numpy as np
from tqdm import tqdm
import wandb

api = wandb.Api()


def get_psnr(level, task, strategy):
    runs = get_runs(level, task, strategy)
    if not len(runs) > 0:
        return np.nan
    psnrs = [predict(run, level, task) for run in runs]
    return np.mean(psnrs)


def get_runs(level, task, strategy):
    runs = api.runs("colin-cooke/ctc", {
        "$and": [{"state": "finished"}, {"config.task": task},
                 {"config.level": level}, {"config.init_strategy": strategy}]})
    ids = []
    for run in tqdm(runs):
        hist = run.history(pandas=False)
        val_losses = [h['val_loss'] for h in hist]
        if len(val_losses) != 75:
            continue
        ids.append(run.id)
    return ids


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
    preds = []
    bits = int(np.log2(level))
    if task == 'pan':
        y_true = np.load(f'/hddraid5/data/colin/ctc/pan_test_{bits}_y.npy')
    else:
        y_true = np.load(f'/hddraid5/data/colin/ctc/new_nuc_test_kb{bits}.npy')
    for index in tqdm(indices):
        test_x_batch = test_x[index:index + batch_size].float()
        with torch.no_grad():
            predictions = model(test_x_batch).cpu().numpy()
        preds.append(predictions[0])
    preds = np.concatenate(preds, axis=0)[:,0]
    avg_psnr = np.mean([measure.compare_psnr(im_true, im_test) for im_true, im_test in zip(y_true, preds)])
    return avg_psnr


if __name__ == "__main__":
    levels = [2, 4, 8, 16, 32, 64, 128]
    tasks = ['pan']
    strategies = ['dpc', 'learned', 'off_axis', 'all', 'random', 'center']
    out_dir = os.path.join('/hddraid5/data/colin/ctc/psnrs')
    os.makedirs(out_dir, exist_ok=True)
    for task in tasks:
        for strategy in strategies:
            psnrs = np.array([get_psnr(level, task, strategy) for level in levels])
            np.save(os.path.join(out_dir, f'{task}_{strategy}.npy'), psnrs)