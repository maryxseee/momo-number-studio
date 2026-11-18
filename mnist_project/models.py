"""Architectures assembled exclusively from standard torch.nn layers."""
from torch import nn


def make_model(kind):
    if kind == 'mlp':
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    if kind == 'cnn':
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(64, 10))
    raise ValueError(f'Unknown architecture: {kind}')


def normalize(x):
    """Fixed scaling, no fitted statistics and no dependence on validation/test."""
    return (x - 0.5) / 0.5
