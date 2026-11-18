"""MNIST partitions and predeclared image transforms, using library operations."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF, InterpolationMode

ROOT = Path(__file__).resolve().parent
CONDITIONS = ['clean', 'rotate_plus20', 'rotate_minus20', 'shift_right3', 'noise_020', 'thick', 'thin']


def split_indices(labels, validation_size=10000):
    return train_test_split(np.arange(len(labels)), test_size=validation_size,
                            stratify=labels, random_state=42)


def load_training(download=True, validation_size=10000, split_seed=42):
    dataset = MNIST(ROOT / 'data', train=True, download=download)
    a, b = train_test_split(np.arange(len(dataset.targets)), test_size=validation_size,
                            stratify=dataset.targets.numpy(), random_state=split_seed)
    x = dataset.data.unsqueeze(1).float() / 255
    return x[a], dataset.targets[a], x[b], dataset.targets[b], a, b


def load_test():
    dataset = MNIST(ROOT / 'data', train=False, download=True)
    return dataset.data.unsqueeze(1).float() / 255, dataset.targets


def raw_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'data' / 'MNIST' / 'raw').glob('*.gz'))}


def augment_batch(x, generator=None, degrees=15, translation=0.1, scale_range=(0.9, 1.1)):
    """Independent affine transform per image. No learned algorithm implemented here."""
    n = len(x)
    angles = (torch.rand(n, generator=generator) * 2 - 1) * degrees * torch.pi / 180
    scales = scale_range[0] + (scale_range[1]-scale_range[0]) * torch.rand(n, generator=generator)
    translations = (torch.rand(n, 2, generator=generator) * 2 - 1) * (2 * translation)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = angles.cos() / scales
    theta[:, 1, 1] = angles.cos() / scales
    theta[:, 0, 1] = -angles.sin() / scales
    theta[:, 1, 0] = angles.sin() / scales
    theta[:, :, 2] = translations
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False).clamp(0, 1)


def corrupt(x, condition):
    if condition == 'clean':
        return x
    if condition.startswith('rotate_'):
        return TF.rotate(x, 20 if condition == 'rotate_plus20' else -20,
                         interpolation=InterpolationMode.BILINEAR, fill=0)
    if condition == 'shift_right3':
        return TF.affine(x, angle=0, translate=[3, 0], scale=1, shear=0,
                         interpolation=InterpolationMode.BILINEAR, fill=0)
    if condition == 'noise_020':
        noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(2026))
        return (x + 0.2 * noise).clamp(0, 1)
    if condition == 'thick':
        return F.max_pool2d(x, 3, stride=1, padding=1)
    if condition == 'thin':
        return -F.max_pool2d(-x, 3, stride=1, padding=1)
    raise ValueError(f'Unknown corruption: {condition}')
