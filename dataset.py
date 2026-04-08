import torch
import numpy as np
import pandas as pd
from pathlib import Path
import ast

def NASA_dataloader(settings, data_root, train=True):
    # settings
    isa = settings['dataset']['isa']
    sensors = settings['dataset']['sensors']
    frame_length = settings['params']['frame_length']
    step = settings['params']['step_size']
    batch_size = settings['params']['batch_size']
    shuffle = settings['params']['shuffle']
    # selecting dataset
    if train:
        raw_time_series = np.load(data_root / 'nasa' / 'train' / f'{isa}.npy')[:,sensors].astype(np.float32)
    else:
        raw_time_series = np.load(data_root / 'nasa' / 'test' / f'{isa}.npy')[:,sensors].astype(np.float32)

    # importing testing labels
    if not train:
        data_info = pd.read_csv('nasa/labeled_anomalies.csv')
        anomalous_regions = ast.literal_eval(data_info[data_info['chan_id']==isa].iloc[0]['anomaly_sequences'])
        raw_labels = np.zeros(len(raw_time_series), dtype=np.float32)
        for region in anomalous_regions:
            for i in range(region[0], region[1]):
                raw_labels[i] = 1

    # slicing frames
    frames = []
    for i in range(0, len(raw_time_series) - frame_length, step):
        frame = raw_time_series[i:i+frame_length,:]
        label = 0 if train else 1 in raw_labels[i:i+frame_length]
        frames.append([frame, label])

    # returning dataloader
    return torch.utils.data.DataLoader(frames, batch_size=batch_size, shuffle=shuffle)