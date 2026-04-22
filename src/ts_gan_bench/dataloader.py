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

def SWaT_dataloader(settings, train_start=21600):
    # settings
    data_root = settings.paths.data_root
    feature_columns = settings.dataset.features # all sensors are used if empty array is provided
    window_size = settings.params.window_size
    stride = settings.params.stride
    batch_size = settings.params.batch_size
    num_workers = settings.params.num_workers
    shuffle = settings.params.shuffle

    # loading datasets
    df_train = pd.read_csv(data_root / 'SWaT' / 'SWaT_Dataset_Normal_v1.csv')
    df_test = pd.read_csv(data_root / 'SWaT' / 'SWaT_Dataset_Attack_v0.csv')

    # extracting sensor data and labels
    if not feature_columns:
        feature_columns = df_train.drop(columns=['Timestamp', 'Normal/Attack']).columns
    label_column = 'Normal/Attack'

    features_train = df_train[feature_columns].to_numpy().astype(np.float32)[train_start:,:]
    features_test = df_test[feature_columns].to_numpy().astype(np.float32)
    labels_test = df_test[label_column].map(lambda x: 0 if x == 'Normal' else 1).to_numpy().astype(np.float32)

    # data normalization
    sensors = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201',  'DPIT301', 'FIT301', 'LIT301', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503', 'FIT601']
    pumps = ['P101', 'P102', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']
    motorized_valves = ['MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304']
    motorized_valve_mapping = np.array([0., -1., 1.])

    for i in range(features_train.shape[1]):
        name = feature_columns[i]
        if name in sensors:
            min_value = np.min(features_train[:,i])
            max_value = np.max(features_train[:,i])
            features_train[:,i] = (features_train[:,i] - min_value) / (max_value - min_value) * 2 - 1
            features_test[:,i] = (features_test[:,i] - min_value) / (max_value - min_value) * 2 - 1
        elif name in pumps:
            # values in pumps are [1., 2.] by default
            # we want to map them to [-1., 1.]
            features_train[:,i] = (features_train[:,i] - 1.5) * 2
            features_test[:,i] = (features_test[:,i] - 1.5) * 2
        elif name in motorized_valves:
            # values in motorized valves are [0., 1., 2.] by default
            # they are non linear, 0 represents moving
            # we want to map them to [0., -1., 1.]
            features_train[:,i] = motorized_valve_mapping[features_train[:,i].astype(int)]
            features_test[:,i] = motorized_valve_mapping[features_test[:,i].astype(int)]

    # slicing frames
    frames_train = []
    for i in range(0, len(features_train) - window_size, stride):
        frame = features_train[i:i+window_size,:]
        label = 0
        frames_train.append([frame, label])

    frames_test = []
    for i in range(0, len(features_test) - window_size, stride):
        frame = features_test[i:i+window_size,:]
        label = 1 in labels_test[i:i+window_size]
        frames_test.append([frame, label])

    # getting actuator index list (used for adding noise)
    actuator_idx = []
    for i, feature_name in enumerate(feature_columns):
        if feature_name in pumps or feature_name in motorized_valves:
            actuator_idx.append(i)

    # returning dataloaders
    train_loader = torch.utils.data.DataLoader(frames_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(frames_test, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader, feature_columns, actuator_idx

def get_SWaT_column_names(settings, data_root):
    # importing column_names from settings
    if settings['dataset']['sensors']:
        return settings['dataset']['sensors']
    
    # importing column_names from csv
    df = pd.read_csv(data_root / 'SWaT' / 'SWaT_Dataset_Normal_v1.csv')
    return df.drop(columns=['Timestamp', 'Normal/Attack']).columns