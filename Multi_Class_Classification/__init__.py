"""
ECG-Based Multi-Class Block Detection Framework
A production-ready deep learning system for cardiac abnormality detection and classification.
"""

__version__ = "1.0.0"
__author__ = "Your Team"
__description__ = "ECG-Based Non-Invasive Block Detection and Classification Framework Using Deep Learning"

from .model import HybridECGModel, create_model, hybrid_ecg_small, hybrid_ecg_base, hybrid_ecg_large
from .dataset import ECGDataModule, ECGDataset, ECGAugmentationPipeline
from .losses import get_loss_function, FocalLoss, WeightedCrossEntropyLoss
from .attention import MultiHeadAttention, MultiHeadSelfAttention, AttentionBlock
from .metrics import MetricsComputer, TrainingMetrics, EarlyStoppingTracker
from .utils import set_seed, get_device, count_parameters, save_checkpoint, load_checkpoint

__all__ = [
    'HybridECGModel',
    'create_model',
    'hybrid_ecg_small',
    'hybrid_ecg_base',
    'hybrid_ecg_large',
    'ECGDataModule',
    'ECGDataset',
    'ECGAugmentationPipeline',
    'get_loss_function',
    'FocalLoss',
    'WeightedCrossEntropyLoss',
    'MultiHeadAttention',
    'MultiHeadSelfAttention',
    'AttentionBlock',
    'MetricsComputer',
    'TrainingMetrics',
    'EarlyStoppingTracker',
    'set_seed',
    'get_device',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint',
]
