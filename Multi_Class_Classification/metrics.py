"""
Comprehensive metrics computation for multi-class ECG classification.
Includes accuracy, precision, recall, F1, ROC-AUC, and confusion matrix.
"""

import numpy as np
import torch
from torch.utils import data
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Compute various classification metrics."""
    
    def __init__(self, num_classes: int, class_names: Optional[Dict[int, str]] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: Dictionary mapping class index to name
        """
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"Class_{i}" for i in range(num_classes)}
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute all metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities [num_samples, num_classes]
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Macro averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted averages
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Per-class metrics
        metrics['per_class'] = self._compute_per_class_metrics(y_true, y_pred)
        
        # ROC-AUC
        if y_proba is not None:
            metrics['roc_auc'] = self._compute_roc_auc(y_true, y_proba)
        
        return metrics
    
    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict]:
        """Compute per-class metrics."""
        per_class = {}
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(self.num_classes):
            class_name = self.class_names.get(i, f"Class_{i}")
            per_class[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        return per_class
    
    def _compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute ROC-AUC score."""
        try:
            if self.num_classes == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                return float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            return None
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report."""
        target_names = [self.class_names.get(i, f"Class_{i}") for i in range(self.num_classes)]
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)


class TrainingMetrics:
    """Tracker for training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.losses = []
        self.accuracies = []
        self.all_preds = []
        self.all_labels = []
    
    def update(self, loss: float, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with batch results.
        
        Args:
            loss: Batch loss
            predictions: Model predictions [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        """
        self.losses.append(loss)
        
        # Accuracy
        preds = predictions.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        self.accuracies.append(accuracy)
        
        # Store for epoch metrics
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get metrics for the epoch."""
        avg_loss = np.mean(self.losses)
        avg_accuracy = np.mean(self.accuracies)
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics


class EarlyStoppingTracker:
    """Track metrics for early stopping."""
    
    def __init__(self, metric: str = 'val_loss', patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Args:
            metric: Metric to monitor
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.should_stop = False
    
    def check(self, current_value: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value
            epoch: Current epoch
        
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.best_epoch = epoch
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.best_epoch = epoch
                self.patience_counter = 0
                return False
            else:
                self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def get_best_value(self) -> float:
        """Get best metric value."""
        return self.best_value
    
    def get_best_epoch(self) -> int:
        """Get epoch with best metric value."""
        return self.best_epoch


class ConfusionMatrixTracker:
    """Compute confusion matrix at various confidence thresholds."""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    def compute_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        return confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
    
    def compute_matrix_at_confidence(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confusion matrix filtering predictions by confidence.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities [num_samples, num_classes]
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            Confusion matrix, mask of valid predictions
        """
        max_proba = np.max(y_proba, axis=1)
        mask = max_proba >= confidence_threshold
        
        y_pred = np.argmax(y_proba, axis=1)
        
        if mask.sum() > 0:
            cm = confusion_matrix(y_true[mask], y_pred[mask], labels=list(range(self.num_classes)))
        else:
            cm = np.zeros((self.num_classes, self.num_classes))
        
        return cm, mask


class ROCCurveComputer:
    """Compute ROC curves for each class."""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    def compute_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Compute ROC curves for one-vs-rest classification.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities [num_samples, num_classes]
        
        Returns:
            Dictionary containing FPR, TPR, AUC for each class
        """
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        roc_curves = {}
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            roc_curves[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }
        
        return roc_curves


class PrecisionRecallComputer:
    """Compute precision-recall curves for each class."""
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    def compute_pr_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Compute precision-recall curves for one-vs-rest classification.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities [num_samples, num_classes]
        
        Returns:
            Dictionary containing precision, recall, AP for each class
        """
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        pr_curves = {}
        
        for i in range(self.num_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ap = auc(recall, precision)
            
            pr_curves[f'class_{i}'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'ap': float(ap)
            }
        
        return pr_curves
