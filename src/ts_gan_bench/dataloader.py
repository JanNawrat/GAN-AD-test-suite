import torch
import numpy as np
import pandas as pd
from pathlib import Path
import ast

def apply_sliding_window(data, window_size, stride, labels=None):
    # leave labels as None for training data
    frames = []
    frame_labels = []
    for i in range(0, len(data) - window_size, stride):
        frame = data[i:i+window_size,:]
        frames.append(frame)

        if labels is not None:
            label = 1 in labels[i:i+window_size]
            frame_labels.append(label)
        else:
            frame_labels.append(0)
    return np.array(frames), np.array(frame_labels)

def wrap_in_dataloader(frames, frame_labels, batch_size=32, num_workers=0, shuffle=True, time_last=False):
    X_tensor = torch.tensor(frames, dtype=torch.float32)
    y_tensor = torch.tensor(frame_labels, dtype=torch.float32)
    if time_last:
        X_tensor = torch.permute(X_tensor, (0, 2, 1))
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

def load_SWaT(data_root, features=None, train_start=21600):
    # returns normalized train and test sets with motorized valve mapping fixed
    # aditionally returns feature names and indexes of actuators
    # leave features as None to load all
    df_train = pd.read_csv(data_root / 'SWaT' / 'train.csv')
    df_test = pd.read_csv(data_root / 'SWaT' / 'test.csv')

    # feature selection
    if not features:
        features = df_train.drop(columns=['Timestamp', 'Normal/Attack']).columns
    label_column = 'Normal/Attack'

    # extracting data to numpy and dropping warm up period from training data
    data_train = df_train[features].to_numpy().astype(np.float32)[train_start:,:]
    data_test = df_test[features].to_numpy().astype(np.float32)
    labels_test = df_test[label_column].map(lambda x: 0 if x == 'Normal' else 1).to_numpy().astype(np.float32)

    # data normalization
    sensors = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203', 'FIT201',  'DPIT301', 'FIT301', 'LIT301', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'PIT501', 'PIT502', 'PIT503', 'FIT601']
    pumps = ['P101', 'P102', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502', 'P601', 'P602', 'P603']
    motorized_valves = ['MV101', 'MV201', 'MV301', 'MV302', 'MV303', 'MV304']
    motorized_valve_mapping = np.array([0., -1., 1.])

    for i in range(data_train.shape[1]):
        name = features[i]
        if name in sensors:
            min_value = np.min(data_train[:,i])
            max_value = np.max(data_train[:,i])
            data_train[:,i] = (data_train[:,i] - min_value) / (max_value - min_value) * 2 - 1
            data_test[:,i] = (data_test[:,i] - min_value) / (max_value - min_value) * 2 - 1
        elif name in pumps:
            # values in pumps are [1., 2.] by default
            # we want to map them to [-1., 1.]
            data_train[:,i] = (data_train[:,i] - 1.5) * 2
            data_test[:,i] = (data_test[:,i] - 1.5) * 2
        elif name in motorized_valves:
            # values in motorized valves are [0., 1., 2.] by default
            # they are non linear, 0 represents moving
            # we want to map them to [0., -1., 1.]
            data_train[:,i] = motorized_valve_mapping[data_train[:,i].astype(int)]
            data_test[:,i] = motorized_valve_mapping[data_test[:,i].astype(int)]

    # getting actuator index list (used for dequantization)
    actuator_idx = []
    for i, feature_name in enumerate(features):
        if feature_name in pumps or feature_name in motorized_valves:
            actuator_idx.append(i)

    return data_train, data_test, labels_test, features, actuator_idx

def load_NASA(data_root, isa, features=None, train=True):
    raw_train_set = np.load(data_root / 'nasa' / 'train' / f'{isa}.npy')[:,sensors].astype(np.float32)
    raw_test_set = np.load(data_root / 'nasa' / 'test' / f'{isa}.npy')[:,sensors].astype(np.float32)

    # importing testing labels
    data_info = pd.read_csv('nasa/labeled_anomalies.csv')
    anomalous_regions = ast.literal_eval(data_info[data_info['chan_id']==isa].iloc[0]['anomaly_sequences'])
    raw_labels = np.zeros(len(raw_time_series), dtype=np.float32)
    for region in anomalous_regions:
        for i in range(region[0], region[1]):
            raw_labels[i] = 1

    return raw_train_set, raw_test_set, raw_labels
