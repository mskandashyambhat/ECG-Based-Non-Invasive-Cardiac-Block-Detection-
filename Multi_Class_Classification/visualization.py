"""
Visualization module for ECG model training and evaluation results.
Includes plots for training curves, confusion matrices, ROC curves, and attention maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Visualize training metrics."""
    
    def __init__(self, output_dir: Path, dpi: int = 300, figsize: Tuple[int, int] = (14, 6)):
        """
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
            figsize: Figure size
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_training_curves(self, history: Dict[str, List[float]], save_name: str = 'training_curves.png'):
        """
        Plot training and validation loss/accuracy curves.
        
        Args:
            history: Dictionary with keys 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
            save_name: Name for saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Loss curve
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2.5, marker='o', markersize=3)
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2.5, marker='s', markersize=3)
            axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
            axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(alpha=0.3)
        
        # Accuracy curve
        if 'train_accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2.5, marker='o', markersize=3)
            axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2.5, marker='s', markersize=3)
            axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
            axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
        plt.close()
    
    def plot_all_metrics(self, history: Dict[str, List[float]], save_name: str = 'all_metrics.png'):
        """
        Plot all training metrics in a grid.
        
        Args:
            history: Dictionary containing training history
            save_name: Name for saved figure
        """
        metrics_keys = [k for k in history.keys() if k.startswith('train_')]
        num_metrics = len(metrics_keys)
        
        if num_metrics == 0:
            logger.warning("No metrics found in history")
            return
        
        num_cols = 2
        num_rows = (num_metrics + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows))
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        for idx, metric_key in enumerate(metrics_keys):
            val_key = metric_key.replace('train_', 'val_')
            
            if val_key in history:
                axes[idx].plot(history[metric_key], label='Train', linewidth=2.5, marker='o', markersize=3)
                axes[idx].plot(history[val_key], label='Val', linewidth=2.5, marker='s', markersize=3)
                axes[idx].set_xlabel('Epoch', fontsize=11, fontweight='bold')
                axes[idx].set_ylabel(metric_key.replace('train_', '').title(), fontsize=11, fontweight='bold')
                axes[idx].set_title(f"{metric_key.replace('train_', '').title()}", fontsize=12, fontweight='bold')
                axes[idx].legend(fontsize=10)
                axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics_keys), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved all metrics to {save_path}")
        plt.close()


class EvaluationVisualizer:
    """Visualize evaluation results."""
    
    def __init__(self, output_dir: Path, dpi: int = 300):
        """
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             save_name: str = 'confusion_matrix.png', normalize: bool = False):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_name: Name for saved figure
            normalize: Whether to normalize by row
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names,
                   yticklabels=class_names, cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, roc_data: Dict[str, Dict], save_name: str = 'roc_curves.png'):
        """
        Plot ROC curves for all classes.
        
        Args:
            roc_data: Dictionary with ROC curve data (fpr, tpr, auc for each class)
            save_name: Name for saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for class_name, data in roc_data.items():
            fpr = data['fpr']
            tpr = data['tpr']
            auc_score = data['auc']
            
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})', linewidth=2.5)
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - One vs Rest', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {save_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, pr_data: Dict[str, Dict], save_name: str = 'pr_curves.png'):
        """
        Plot precision-recall curves for all classes.
        
        Args:
            pr_data: Dictionary with PR curve data
            save_name: Name for saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for class_name, data in pr_data.items():
            precision = data['precision']
            recall = data['recall']
            ap = data['ap']
            
            ax.plot(recall, precision, label=f'{class_name} (AP = {ap:.3f})', linewidth=2.5)
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved PR curves to {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics: Dict[str, float], save_name: str = 'metrics_comparison.png'):
        """
        Plot comparison of different metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            save_name: Name for saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Metrics Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved metrics comparison to {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self, per_class_metrics: Dict[str, Dict],
                              save_name: str = 'per_class_metrics.png'):
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            per_class_metrics: Dictionary with per-class metrics
            save_name: Name for saved figure
        """
        class_names = list(per_class_metrics.keys())
        precision_vals = [per_class_metrics[c]['precision'] for c in class_names]
        recall_vals = [per_class_metrics[c]['recall'] for c in class_names]
        f1_vals = [per_class_metrics[c]['f1'] for c in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision_vals, width, label='Precision', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x, recall_vals, width, label='Recall', edgecolor='black', linewidth=1.2)
        bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', edgecolor='black', linewidth=1.2)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Metrics', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved per-class metrics to {save_path}")
        plt.close()


class SignalVisualizer:
    """Visualize ECG signals and model outputs."""
    
    def __init__(self, output_dir: Path, dpi: int = 300, sampling_rate: int = 500):
        """
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
            sampling_rate: Sampling rate of ECG signals
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.sampling_rate = sampling_rate
    
    def plot_ecg_signals(self, signals: np.ndarray, labels: np.ndarray, class_names: Dict[int, str],
                        num_samples_per_class: int = 3, save_name: str = 'ecg_signals.png'):
        """
        Plot sample ECG signals from each class.
        
        Args:
            signals: Array of ECG signals
            labels: Class labels
            class_names: Dictionary mapping class index to name
            num_samples_per_class: Number of samples to plot per class
            save_name: Name for saved figure
        """
        num_classes = len(np.unique(labels))
        duration = signals.shape[1] / self.sampling_rate
        time_axis = np.linspace(0, duration, signals.shape[1])
        
        fig, axes = plt.subplots(num_classes, num_samples_per_class, figsize=(15, 3 * num_classes))
        axes = axes.reshape(num_classes, num_samples_per_class) if num_classes > 1 else axes.reshape(1, -1)
        
        for class_id in range(num_classes):
            class_indices = np.where(labels == class_id)[0]
            sample_indices = np.random.choice(class_indices, size=min(num_samples_per_class, len(class_indices)), replace=False)
            
            for sample_idx, ax_idx in enumerate(sample_indices):
                ax = axes[class_id, sample_idx]
                signal = signals[ax_idx]
                
                ax.plot(time_axis, signal, linewidth=1.5, color='steelblue')
                ax.set_title(f'{class_names.get(class_id, f"Class {class_id}")} (Sample {sample_idx + 1})',
                           fontsize=10, fontweight='bold')
                ax.set_xlabel('Time (s)', fontsize=9)
                ax.set_ylabel('Amplitude', fontsize=9)
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved ECG signals to {save_path}")
        plt.close()
    
    def plot_attention_weights(self, attention_weights: np.ndarray, class_name: str,
                              save_name: str = 'attention_weights.png'):
        """
        Plot attention weights heatmap.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len] or averaged across heads
            class_name: Name of the class
            save_name: Name for saved figure
        """
        # Average across heads if 3D
        if attention_weights.ndim == 3:
            attention_weights = attention_weights.mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_weights, cmap='hot', cbar_kws={'label': 'Attention Weight'}, ax=ax)
        
        ax.set_xlabel('Sequence Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sequence Position', fontsize=12, fontweight='bold')
        ax.set_title(f'Attention Weights - {class_name}', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved attention weights to {save_path}")
        plt.close()
    
    def plot_signal_with_attention(self, signal: np.ndarray, attention_weights: np.ndarray,
                                  class_name: str, save_name: str = 'signal_attention.png'):
        """
        Plot ECG signal with attention weights highlighted.
        
        Args:
            signal: ECG signal [seq_len]
            attention_weights: Attention weights [seq_len] (averaged across time and heads)
            class_name: Name of the class
            save_name: Name for saved figure
        """
        duration = signal.shape[0] / self.sampling_rate
        time_axis = np.linspace(0, duration, signal.shape[0])
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot signal
        ax.plot(time_axis, signal, linewidth=2, color='steelblue', label='ECG Signal')
        
        # Color regions by attention
        for i in range(len(signal) - 1):
            color_intensity = attention_weights[i]
            ax.axvspan(time_axis[i], time_axis[i + 1], alpha=color_intensity * 0.3, color='red')
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        ax.set_title(f'ECG Signal with Attention - {class_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved signal with attention to {save_path}")
        plt.close()
