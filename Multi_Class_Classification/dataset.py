"""
Dataset handling module for ECG multi-class classification.
Includes data loading, preprocessing, and PyTorch DataLoader creation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict
import logging

class ECGAugmentationPipeline:
    """
    Comprehensive ECG signal augmentation pipeline for training.
    Applies random transformations to improve model generalization.
    """
    
    def __init__(self, noise_std_range=(0.01, 0.03), amplitude_range=(0.85, 1.15),
                 shift_range=15, wander_amplitude_range=(0.02, 0.1),
                 noise_prob=0.6, amplitude_prob=0.4, shift_prob=0.4, wander_prob=0.3):
        self.noise_std_range = noise_std_range
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.wander_amplitude_range = wander_amplitude_range
        self.noise_prob = noise_prob
        self.amplitude_prob = amplitude_prob
        self.shift_prob = shift_prob
        self.wander_prob = wander_prob
    
    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline to a single ECG signal tensor."""
        import random
        signal = signal.clone()
        
        # Gaussian noise
        if random.random() < self.noise_prob:
            std = random.uniform(*self.noise_std_range)
            signal = signal + torch.randn_like(signal) * std
        
        # Amplitude scaling
        if random.random() < self.amplitude_prob:
            scale = random.uniform(*self.amplitude_range)
            signal = signal * scale
        
        # Time shift (circular roll)
        if random.random() < self.shift_prob:
            shift = random.randint(-self.shift_range, self.shift_range)
            signal = torch.roll(signal, shift)
        
        # Baseline wander (low-freq sine drift)
        if random.random() < self.wander_prob:
            length = signal.shape[-1]
            freq = random.uniform(0.5, 2.0)  # Hz equivalent in normalized frequency
            amp = random.uniform(*self.wander_amplitude_range)
            t = torch.linspace(0, 2 * 3.14159 * freq, length)
            wander = amp * torch.sin(t)
            signal = signal + wander
        
        return signal

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals.
    """
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            signals: ECG signals [num_samples, signal_length]
            labels: Class labels [num_samples]
            transform: Optional data augmentation transform
        """
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
        
        assert len(self.signals) == len(self.labels), "Signals and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            signal = self.transform(signal)
        
        return signal, label


class ECGDataModule:
    """
    Data module for managing ECG dataset loading and splitting.
    """
    
    def __init__(self, data_path: str, train_size: float = 0.7, val_size: float = 0.15,
                 test_size: float = 0.15, random_seed: int = 42, stratified: bool = True):
        """
        Args:
            data_path: Path to preprocessed ECG dataset (NPZ format)
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            test_size: Proportion of data for testing
            random_seed: Random seed for reproducibility
            stratified: Whether to use stratified split
        """
        self.data_path = data_path
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        self.stratified = stratified
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Class information
        self.num_classes = None
        self.class_distribution = None
    
    def load_data(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return raw data."""
        logger.info(f"Loading data from {self.data_path}")
        data = np.load(self.data_path)
        X = data['X']
        y = data['y']
        logger.info(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")

        if X.ndim != 2:
            raise ValueError(f"Expected X to have shape [N, 300], got {X.shape}")
        if X.shape[1] != 300:
            raise ValueError(f"Expected signal length 300, got {X.shape[1]}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D labels, got {y.shape}")

        unique_labels = np.unique(y)
        if np.any((unique_labels < 0) | (unique_labels > 4)):
            raise ValueError(f"Labels must be encoded in the range 0-4, found {unique_labels.tolist()}")
        
        if max_samples is not None and len(X) > max_samples:
            logger.info(f"Using stratified subset: {max_samples} samples (from {len(X)} total)")
            rng = np.random.RandomState(self.random_seed)
            indices = []
            classes = np.unique(y)
            class_counts = {cls: np.sum(y == cls) for cls in classes}
            total = float(len(X))

            for cls in classes:
                cls_idx = np.where(y == cls)[0]
                take = int(round(max_samples * (class_counts[cls] / total)))
                take = max(1, min(len(cls_idx), take))
                indices.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

            if len(indices) > max_samples:
                indices = rng.choice(indices, size=max_samples, replace=False).tolist()
            elif len(indices) < max_samples:
                remaining = np.setdiff1d(np.arange(len(X)), np.array(indices, dtype=int), assume_unique=False)
                extra = rng.choice(remaining, size=max_samples - len(indices), replace=False).tolist()
                indices.extend(extra)

            indices = np.array(indices, dtype=int)
            X = X[indices]
            y = y[indices]
            logger.info(f"Subset created: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y
    
    def prepare_data(self, max_samples: Optional[int] = None) -> None:
        """Load and split data."""
        X, y = self.load_data(max_samples=max_samples)
        
        # Get class information
        self.num_classes = len(np.unique(y))
        self.class_distribution = {i: np.sum(y == i) for i in range(self.num_classes)}
        
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class distribution:\n{self._format_class_dist()}")
        
        # First split: train+val vs test
        if self.stratified:
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_seed
            )
        else:
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_seed
            )
        
        # Second split: train vs val
        val_ratio = self.val_size / (self.train_size + self.val_size)
        
        if self.stratified:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                stratify=y_temp,
                random_state=self.random_seed
            )
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=self.random_seed
            )
        
        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Val set: {self.X_val.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        
        # Log class distribution in each split
        logger.info(f"Train class distribution:")
        for i in range(self.num_classes):
            count = np.sum(self.y_train == i)
            logger.info(f"  Class {i}: {count}")
        
        logger.info(f"Val class distribution:")
        for i in range(self.num_classes):
            count = np.sum(self.y_val == i)
            logger.info(f"  Class {i}: {count}")
        
        logger.info(f"Test class distribution:")
        for i in range(self.num_classes):
            count = np.sum(self.y_test == i)
            logger.info(f"  Class {i}: {count}")
    
    def _format_class_dist(self) -> str:
        """Format class distribution for logging."""
        lines = []
        for class_id, count in self.class_distribution.items():
            percentage = (count / sum(self.class_distribution.values())) * 100
            lines.append(f"  Class {class_id}: {count} ({percentage:.1f}%)")
        return "\n".join(lines)
    
    def get_train_dataloader(self, batch_size: int = 64, num_workers: int = 4,
                            weighted_sampling: bool = False, transform=None) -> DataLoader:
        """Get training dataloader."""
        if transform is None:
            transform = ECGAugmentationPipeline()
        dataset = ECGDataset(self.X_train, self.y_train, transform=transform)
        
        if weighted_sampling:
            # Compute class weights for weighted sampling
            class_counts = np.bincount(self.y_train, minlength=self.num_classes)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[self.y_train]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=False,  # ✅ Disable pin_memory on CPU
                drop_last=True
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,  # ✅ Disable pin_memory on CPU
                drop_last=True
            )
        
        return dataloader
    
    def get_val_dataloader(self, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
        """Get validation dataloader."""
        dataset = ECGDataset(self.X_val, self.y_val)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False  # ✅ Disable pin_memory on CPU
        )
        return dataloader
    
    def get_test_dataloader(self, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
        """Get test dataloader."""
        dataset = ECGDataset(self.X_test, self.y_test)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False  # ✅ Disable pin_memory on CPU
        )
        return dataloader
    
    def get_full_dataloader(self, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
        """Get dataloader for entire dataset (for inference or final training)."""
        X = np.concatenate([self.X_train, self.X_val, self.X_test])
        y = np.concatenate([self.y_train, self.y_val, self.y_test])
        dataset = ECGDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        return dataloader
    
    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights for handling imbalance."""
        total = len(self.y_train)
        weights = {}
        for class_id in range(self.num_classes):
            count = np.sum(self.y_train == class_id)
            weights[class_id] = total / (self.num_classes * count)
        return weights



