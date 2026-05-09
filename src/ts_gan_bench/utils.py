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
    # assumes anomaly score frames as [n_frames, window_size] or [n_frames] or [n_frames, 1]
    # available strategies:
    #   "max" - maximal score for the point in any frame
    #   "mean" - mean of all scores for the point
    #   "gaussian_mean" - scores are weighted by a gausian kernel
    if not isinstance(as_frames, torch.Tensor):
        as_frames = torch.tensor(as_frames)
    
    device = as_frames.device
    dtype = as_frames.dtype
    n_frames = as_frames.shape[0]
    dataset_size = window_size + (n_frames - 1) * stride

    # [n_frames] -> [n_frames, window_size]
    if as_frames.ndim == 1:
        as_frames = as_frames.unsqueeze(1).expand(n_frames, window_size)

    # [n_frames, 1] -> [n_frames, windows_size]
    elif as_frames.shape[1] == 1:
        as_frames = as_frames.expand(n_frames, window_size)

    if strategy == 'gaussian_mean':
        std = window_size / 6.0
        # Replicate scipy.signal.windows.gaussian mathematically
        n = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1.0) / 2.0
        weights = torch.exp(-n**2 / (2 * std**2))
    else:
        weights = torch.ones(window_size, device=device, dtype=dtype)

    # frame_idx: [n_frames, 1] - starting index of each frame
    frame_idx = (torch.arange(n_frames, device=device) * stride).unsqueeze(1)
    # window_idx: [1, window_size] - offset within the window
    window_idx = torch.arange(window_size, device=device).unsqueeze(0)
    
    # target_idx: [n_frames, window_size] - global sequence index for every score
    target_idx = frame_idx + window_idx

    target_idx_flat = target_idx.view(-1)
    as_frames_flat = as_frames.contiguous().view(-1)
    weights_flat = weights.unsqueeze(0).expand(n_frames, window_size).reshape(-1)

    if strategy == 'max':
        # init with -inf, required for scatter with include_self=False
        aggregated_scores = torch.full((dataset_size,), float('-inf'), device=device, dtype=dtype)

        aggregated_scores.scatter_reduce_(
            0, target_idx_flat, as_frames_flat, reduce="amax", include_self=False
        )
        
        # failsafe
        aggregated_scores[aggregated_scores == float('-inf')] = 0.0
        return aggregated_scores

    elif strategy in ['mean', 'gaussian_mean']:
        weighted_frames_flat = as_frames_flat * weights_flat

        aggregated_scores = torch.zeros(dataset_size, device=device, dtype=dtype)
        aggregated_weights = torch.zeros(dataset_size, device=device, dtype=dtype)

        aggregated_scores.scatter_add_(0, target_idx_flat, weighted_frames_flat)
        aggregated_weights.scatter_add_(0, target_idx_flat, weights_flat)

        # avoid division by zero
        aggregated_weights = aggregated_weights.clamp(min=1e-8)
        
        return aggregated_scores / aggregated_weights

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
