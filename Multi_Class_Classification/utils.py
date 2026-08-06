"""
Utility functions for ECG signal processing, data handling, and common operations.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, log_dir: Path, level=logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def get_device(device_name: str = "cuda") -> torch.device:
    """Get appropriate device (CUDA, MPS, or CPU)."""
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def print_device_info():
    """Print device information."""
    print("\n" + "="*80)
    print("DEVICE INFORMATION")
    print("="*80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print("="*80 + "\n")

# ============================================================================
# SEED MANAGEMENT
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def normalize_signal(signal: np.ndarray, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize signal using z-score normalization.
    
    Args:
        signal: Input ECG signal
        mean: Pre-computed mean (for inference)
        std: Pre-computed std (for inference)
    
    Returns:
        Normalized signal, mean, std
    """
    if mean is None:
        mean = np.mean(signal, keepdims=True)
    if std is None:
        std = np.std(signal, keepdims=True)
    
    normalized = (signal - mean) / (std + 1e-8)
    return normalized, mean, std

# ============================================================================
# ECG SIGNAL PROCESSING
# ============================================================================

def compute_rr_intervals(signal: np.ndarray, sampling_rate: int = 500) -> np.ndarray:
    """Compute RR intervals from ECG signal."""
    from scipy.signal import find_peaks
    
    peaks, _ = find_peaks(signal, distance=sampling_rate // 4, height=np.std(signal))
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate * 1000  # Convert to ms
        return rr_intervals
    return np.array([])

def compute_heart_rate(rr_intervals: np.ndarray) -> float:
    """Compute heart rate from RR intervals."""
    if len(rr_intervals) > 0:
        return 60000 / np.mean(rr_intervals)
    return 0.0

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None,
                   class_names: Dict[int, str] = None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for ROC-AUC)
        class_names: Dictionary mapping class indices to names
    
    Returns:
        Dictionary containing all metrics
    """
    num_classes = len(np.unique(y_true))
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class_metrics'] = {}
    for i in range(num_classes):
        class_name = class_names.get(i, f"Class_{i}") if class_names else f"Class_{i}"
        metrics['per_class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i])
        }
    
    # ROC-AUC (if probabilities provided and binary or use one-vs-rest)
    if y_proba is not None:
        try:
            if num_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            metrics['roc_auc'] = None
    
    return metrics

# ============================================================================
# FILE I/O
# ============================================================================

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                   metrics: Dict[str, float], save_path: Path):
    """Save model checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, load_path: Path,
                   device: torch.device) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    return model, optimizer, epoch, metrics

def save_results(results: Dict[str, Any], save_path: Path):
    """Save results to JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)

def load_results(load_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(load_path, 'r') as f:
        return json.load(f)

# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """Print model summary."""
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(model)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("="*80 + "\n")

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# ============================================================================
# TENSOR OPERATIONS
# ============================================================================

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move batch tensors to device."""
    return {k: v.to(device) for k, v in batch.items()}

def detach_cpu(tensor: torch.Tensor) -> np.ndarray:
    """Detach tensor and move to CPU as numpy array."""
    return tensor.detach().cpu().numpy()

# ============================================================================
# MIXUP AUGMENTATION
# ============================================================================

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation.
    
    Args:
        x: Input tensor [batch_size, channels, length]
        y: Target tensor [batch_size]
        alpha: Beta distribution parameter
    
    Returns:
        Mixed inputs, mixed targets, lambda
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha)
    
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================================
# GRADIENT UTILITIES
# ============================================================================

def clip_gradient(model: nn.Module, max_norm: float = 1.0):
    """Clip model gradients."""
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def check_nan_gradients(model: nn.Module) -> bool:
    """Check if model has NaN gradients."""
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

# ============================================================================
# MONITORING & PROFILING
# ============================================================================

def get_model_memory(model: nn.Module, input_shape: Tuple[int, ...], device: torch.device) -> float:
    """Estimate model memory usage in MB."""
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Count parameters
    param_size = count_parameters(model) * 4 / (1024 ** 2)  # Assuming float32
    
    # Estimate activation memory (rough)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        activation_size = output.numel() * 4 / (1024 ** 2)
    except:
        activation_size = 0
    
    return param_size + activation_size
