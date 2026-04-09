import torch
import numpy as np
import pandas as pd
from pathlib import Path
import ast

def NASA_dataloader(settings, data_root, train=True): # currently training and testing dataste are imported in the same way
    # settings
    isa = settings['dataset']['isa']
    sensors = settings['dataset']['sensors']
    frame_length = settings['params']['frame_length']
    step_size = settings['params']['step_size']
    batch_size = settings['params']['batch_size']
    num_workers = settings['params']['num_workers']
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
    for i in range(0, len(raw_time_series) - frame_length, step_size):
        frame = raw_time_series[i:i+frame_length,:]
        label = 0 if train else 1 in raw_labels[i:i+frame_length]
        frames.append([frame, label])

    # returning dataloader
    return torch.utils.data.DataLoader(frames, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def SWaT_dataloader(settings, data_root, train=True):
    # settings
    sensor_columns = settings['dataset']['sensors'] # all sensors are used if empty array is provided
    frame_length = settings['params']['frame_length']
    step_size = settings['params']['step_size']
    batch_size = settings['params']['batch_size']
    num_workers = settings['params']['num_workers']
    shuffle = settings['params']['shuffle']

    # selecting dataset
    if train:
        df = pd.read_csv(data_root / 'SWaT' / 'SWaT_Dataset_Normal_v1.csv')
    else:
        df = pd.read_csv(data_root / 'SWaT' / 'SWaT_Dataset_Attack_v0.csv')

    # extracting sensor data and labels
    if not sensor_columns:
        sensor_columns = df.drop(columns=['Timestamp', 'Normal/Attack']).columns
    label_column = 'Normal/Attack'

    sensors = df[sensor_columns].to_numpy().astype(np.float32)
    labels = df[label_column].map(lambda x: 0 if x == 'Normal' else 1).to_numpy().astype(np.float32)

    # slicing frames
    frames = []
    for i in range(0, len(sensors) - frame_length, step_size):
        frame = sensors[i:i+frame_length,:]
        label = 1 in labels[i:i+frame_length]
        frames.append([frame, label])

    # returning dataloader
    return torch.utils.data.DataLoader(frames, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)