import numpy as np
from scipy.signal.windows import gaussian
import torch

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def add_bounded_dequantization(samples, amount, idx):
    samples_quantized = samples.clone()
    actuators = samples_quantized[..., idx]
    noise = torch.rand_like(actuators) * amount
    # handling the poles (-1 and 1)
    pole_mask = (actuators != 0)
    actuators[pole_mask] -= torch.sign(actuators[pole_mask]) * noise[pole_mask]
    # handling the zeros
    zero_mask = (actuators == 0)
    if zero_mask.any():
        actuators[zero_mask] += noise[zero_mask] - amount / 2

    samples_quantized[..., idx] = actuators
    return samples_quantized

def map_anomaly_score_to_sequence(as_frames, window_size, stride, strategy='mean'):
    # assumes anomaly score frames as [n_frames, window_size] or [n_frames]
    # available strategies:
    #   "max" - maximal score for the point in any frame
    #   "mean" - mean of all scores for the point
    #   "gaussian_mean" - scores are weighted by a gausian kernel
    n_frames = as_frames.shape[0]
    dataset_size = window_size + (n_frames - 1) * stride
    frames_per_point = window_size // stride

    if as_frames.ndim == 1:
        as_frames = np.broadcast_to(as_frames[:,np.newaxis], (n_frames, window_size))

    weights = (
        np.array(gaussian(window_size, std=window_size/6, sym=True), dtype=np.float32)
        if strategy == 'gaussian_mean'
        else [1 for i in range(window_size)]
    )

    scores = np.zeros((dataset_size, frames_per_point), dtype=np.float32)
    score_weights = np.zeros((dataset_size, frames_per_point), dtype=np.float32)
    aggregated_scores = np.zeros(dataset_size, dtype=np.float32)
    idx = []
    for i in range(frames_per_point):
        idx += stride * [i]
    for i, frame in enumerate(as_frames):
        start = i * stride
        end = i * stride + window_size
        for j in range(window_size):
            scores[start + j, idx[j]] = weights[j] * frame[j]
            score_weights[start + j, idx[j]] = weights[j]

    for i in range(dataset_size):
        aggregated_scores[i] = (
            np.max(scores[i, np.where(score_weights[i] == 1)])
            if strategy == 'max'
            else np.sum(scores[i]) / np.sum(score_weights[i])
        )

    return aggregated_scores
