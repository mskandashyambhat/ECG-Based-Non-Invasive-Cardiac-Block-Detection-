"""
Main training script for ECG multi-class block detection system.
Implements complete training pipeline with validation and testing.
"""

import os
import sys
import argparse
import copy
import tempfile
import importlib.util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CURRENT_DIR / 'config.py'
CONFIG_SPEC = importlib.util.spec_from_file_location('ecg_local_config', CONFIG_PATH)
cfg = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
CONFIG_SPEC.loader.exec_module(cfg)
for name, value in vars(cfg).items():
    if not name.startswith('_'):
        globals()[name] = value
from utils import set_seed, get_device, print_device_info, setup_logger, count_parameters, save_checkpoint, load_checkpoint, clip_gradient, save_results, print_model_summary
from dataset import ECGDataModule, ECGAugmentationPipeline
from model import HybridECGModel
from losses import get_loss_function
from metrics import MetricsComputer, TrainingMetrics, EarlyStoppingTracker
from visualization import TrainingVisualizer, EvaluationVisualizer, SignalVisualizer

# Setup logging
logger = setup_logger('train', LOGS_DIR)

# ============================================================================
# MAIN TRAINING CLASS
# ============================================================================

class ECGTrainer:
    """Main trainer class for ECG model."""
    
    def __init__(self, config_dict: Dict = None):
        """Initialize trainer."""
        self.cfg = config_dict or {
            'device': DEVICE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'class_weights': CLASS_WEIGHTS,
            'max_train_samples': MAX_TRAIN_SAMPLES
        }
        
        # Set seed
        set_seed(RANDOM_SEED)
        
        # Device
        self.device = get_device(self.cfg['device'])
        print_device_info()
        self.use_amp = bool(USE_AMP and self.device.type == 'cuda')
        
        # Create directories
        for directory in [MODELS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Data
        logger.info("Loading data...")
        self.data_module = ECGDataModule(
            data_path=str(cfg.PREPROCESSED_DATA),
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
            random_seed=RANDOM_SEED,
            stratified=STRATIFIED_SPLIT
        )
        self.data_module.prepare_data(max_samples=self.cfg['max_train_samples'])
        
        # Model
        logger.info("Creating model...")
        self.model = HybridECGModel(
            num_classes=self.cfg['num_classes'],
            input_length=SIGNAL_LENGTH,
            input_channels=NUM_LEADS,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_num_layers=LSTM_NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            dropout=DROPOUT_RATES[0]
        ).to(self.device)
        
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        print_model_summary(self.model, (1, SIGNAL_LENGTH))

        self.train_loader = self.data_module.get_train_dataloader(
            batch_size=self.cfg['batch_size'],
            num_workers=NUM_WORKERS,
            weighted_sampling=True
        )
        self.val_loader = self.data_module.get_val_dataloader(
            batch_size=self.cfg['batch_size'],
            num_workers=NUM_WORKERS
        )
        self.test_loader = self.data_module.get_test_dataloader(
            batch_size=self.cfg['batch_size'],
            num_workers=NUM_WORKERS
        )
        
        # Loss
        self.criterion = get_loss_function(
            LOSS_TYPE,
            num_classes=self.cfg['num_classes'],
            class_weights=self.cfg['class_weights'],
            label_smoothing=LABEL_SMOOTHING
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['learning_rate'],
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=getattr(cfg, 'COSINE_T0', 10),
            T_mult=getattr(cfg, 'COSINE_T_MULT', 2),
            eta_min=REDUCE_LR_MIN
        )
        
        # Metrics
        self.metrics_computer = MetricsComputer(
            num_classes=self.cfg['num_classes'],
            class_names=self.cfg['class_names']
        )
        
        # Early stopping
        self.early_stopping = EarlyStoppingTracker(
            metric='val_loss',
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            mode='min'
        )
        
        # AMP
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': []
        }
        
        # Visualizers
        self.train_viz = TrainingVisualizer(VISUALIZATIONS_DIR)
        self.eval_viz = EvaluationVisualizer(VISUALIZATIONS_DIR)
        self.signal_viz = SignalVisualizer(VISUALIZATIONS_DIR)
        
        logger.info("Trainer initialized successfully")

    def _get_train_batch(self):
        return next(iter(self.train_loader))

    def _batch_to_device(self, batch):
        signals, labels = batch
        return signals.to(self.device), labels.to(self.device)

    def run_pre_training_audit(self) -> Dict[str, object]:
        """Run a compact pre-training audit before a long training run."""
        logger.info("\n" + "=" * 80)
        logger.info("PRE-TRAINING AUDIT")
        logger.info("=" * 80)

        audit: Dict[str, object] = {}
        X_full, y_full = self.data_module.load_data(max_samples=None)

        audit['dataset_shape'] = tuple(X_full.shape)
        audit['label_set'] = sorted(np.unique(y_full).tolist())
        audit['class_distribution'] = {int(k): int(np.sum(y_full == k)) for k in sorted(np.unique(y_full))}

        if X_full.ndim != 2 or X_full.shape[1] != SIGNAL_LENGTH:
            raise ValueError(f"Dataset shape check failed: expected (N, {SIGNAL_LENGTH}), got {X_full.shape}")
        if audit['label_set'] != [0, 1, 2, 3, 4]:
            raise ValueError(f"Label encoding check failed: expected labels 0-4, found {audit['label_set']}")

        train_hashes = {tuple(row.tolist()) for row in np.asarray(self.data_module.X_train)}
        val_hashes = {tuple(row.tolist()) for row in np.asarray(self.data_module.X_val)}
        test_hashes = {tuple(row.tolist()) for row in np.asarray(self.data_module.X_test)}
        leakage = (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)
        audit['split_leakage_count'] = len(leakage)
        if leakage:
            logger.warning(
                f"Found {len(leakage)} exact waveform duplicates across splits. "
                "This can happen with repeated beats and is not a direct leakage proof."
            )

        batch = self._get_train_batch()
        signals, labels = self._batch_to_device(batch)
        audit['dataloader_signal_shape'] = tuple(signals.shape)
        audit['dataloader_label_shape'] = tuple(labels.shape)

        logits, attention_weights = self.model(signals)
        audit['forward_logits_shape'] = tuple(logits.shape)
        audit['attention_shape'] = tuple(attention_weights.shape) if attention_weights is not None else None

        loss_value = self.criterion(logits, labels)
        audit['initial_loss'] = float(loss_value.item())

        if torch.cuda.is_available() and self.device.type == 'cuda':
            audit['gpu_memory_allocated_mb'] = float(torch.cuda.memory_allocated(self.device) / (1024 ** 2))
            audit['gpu_memory_reserved_mb'] = float(torch.cuda.memory_reserved(self.device) / (1024 ** 2))
        else:
            audit['gpu_memory_allocated_mb'] = None
            audit['gpu_memory_reserved_mb'] = None

        sanity_model = copy.deepcopy(self.model)
        sanity_optimizer = optim.AdamW(
            sanity_model.parameters(),
            lr=self.cfg['learning_rate'],
            weight_decay=WEIGHT_DECAY
        )
        sanity_model.train()
        sanity_optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            with autocast():
                sanity_logits, _ = sanity_model(signals)
                sanity_loss = self.criterion(sanity_logits, labels)
            self.scaler.scale(sanity_loss).backward()
            self.scaler.unscale_(sanity_optimizer)
            clip_gradient(sanity_model, GRADIENT_CLIP)
            self.scaler.step(sanity_optimizer)
            self.scaler.update()
        else:
            sanity_logits, _ = sanity_model(signals)
            sanity_loss = self.criterion(sanity_logits, labels)
            sanity_loss.backward()
            clip_gradient(sanity_model, GRADIENT_CLIP)
            sanity_optimizer.step()

        audit['sanity_train_loss'] = float(sanity_loss.item())

        checkpoint_path = MODELS_DIR / 'audit_checkpoint.pt'
        save_checkpoint(sanity_model, sanity_optimizer, epoch=0, metrics={'loss': audit['sanity_train_loss']}, save_path=checkpoint_path)
        reloaded_model = copy.deepcopy(sanity_model).to(self.device)
        reloaded_optimizer = optim.AdamW(reloaded_model.parameters(), lr=self.cfg['learning_rate'], weight_decay=WEIGHT_DECAY)
        reloaded_model, reloaded_optimizer, checkpoint_epoch, checkpoint_metrics = load_checkpoint(
            reloaded_model,
            reloaded_optimizer,
            checkpoint_path,
            self.device
        )
        audit['checkpoint_epoch'] = int(checkpoint_epoch)
        audit['checkpoint_metrics'] = checkpoint_metrics

        try:
            from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

            preds = sanity_logits.argmax(dim=1)
            tm_accuracy = MulticlassAccuracy(num_classes=self.cfg['num_classes'], average='micro').to(self.device)
            tm_precision = MulticlassPrecision(num_classes=self.cfg['num_classes'], average='macro').to(self.device)
            tm_recall = MulticlassRecall(num_classes=self.cfg['num_classes'], average='macro').to(self.device)
            tm_f1 = MulticlassF1Score(num_classes=self.cfg['num_classes'], average='macro').to(self.device)
            audit['torchmetrics'] = {
                'accuracy': float(tm_accuracy(preds, labels).item()),
                'precision_macro': float(tm_precision(preds, labels).item()),
                'recall_macro': float(tm_recall(preds, labels).item()),
                'f1_macro': float(tm_f1(preds, labels).item()),
            }
        except Exception as exc:
            audit['torchmetrics'] = {'available': False, 'error': str(exc)}

        logger.info(f"Audit dataset shape: {audit['dataset_shape']}")
        logger.info(f"Audit label set: {audit['label_set']}")
        logger.info(f"Audit split leakage count: {audit['split_leakage_count']}")
        logger.info(f"Audit train batch shape: {audit['dataloader_signal_shape']}")
        logger.info(f"Audit logits shape: {audit['forward_logits_shape']}")
        logger.info(f"Audit initial loss: {audit['initial_loss']:.4f}")
        logger.info(f"Audit sanity train loss: {audit['sanity_train_loss']:.4f}")
        logger.info("Pre-training audit passed")

        return audit
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = TrainingMetrics()

        batch_pbar = tqdm(
            self.train_loader,
            desc="Train",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5
        )

        for batch_idx, (signals, labels) in enumerate(batch_pbar):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # In-batch ECG augmentation (training only)
            if self.model.training:
                with torch.no_grad():
                    # Random Gaussian noise
                    noise_mask = torch.rand(signals.size(0)) < 0.5
                    if noise_mask.any():
                        signals[noise_mask] += torch.randn_like(signals[noise_mask]) * 0.02
                    # Random amplitude scaling
                    amp_mask = torch.rand(signals.size(0)) < 0.3
                    if amp_mask.any():
                        scale = 0.85 + 0.3 * torch.rand(amp_mask.sum(), 1, device=signals.device)
                        signals[amp_mask] = signals[amp_mask] * scale
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    # MixUp augmentation
                    if USE_MIXUP and self.model.training:
                        import numpy as np
                        alpha = getattr(cfg, 'MIXUP_ALPHA', 0.2)
                        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                        batch_size = signals.size(0)
                        index = torch.randperm(batch_size, device=signals.device)
                        signals = lam * signals + (1 - lam) * signals[index]
                        labels_a, labels_b = labels, labels[index]
                        # Use mixed loss
                        logits, _ = self.model(signals)
                        loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                    else:
                        logits, _ = self.model(signals)
                        loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_gradient(self.model, GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # MixUp augmentation
                if USE_MIXUP and self.model.training:
                    import numpy as np
                    alpha = getattr(cfg, 'MIXUP_ALPHA', 0.2)
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    batch_size = signals.size(0)
                    index = torch.randperm(batch_size, device=signals.device)
                    signals = lam * signals + (1 - lam) * signals[index]
                    labels_a, labels_b = labels, labels[index]
                    # Use mixed loss
                    logits, _ = self.model(signals)
                    loss = lam * self.criterion(logits, labels_a) + (1 - lam) * self.criterion(logits, labels_b)
                else:
                    logits, _ = self.model(signals)
                    loss = self.criterion(logits, labels)
                loss.backward()
                clip_gradient(self.model, GRADIENT_CLIP)
                self.optimizer.step()
            
            metrics.update(loss.item(), logits.detach(), labels.detach())

            batch_pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{metrics.accuracies[-1]:.4f}"
            )
        
        epoch_metrics = metrics.get_epoch_metrics()
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        metrics = TrainingMetrics()
        
        all_preds = []
        all_probs = []

        val_pbar = tqdm(
            self.val_loader,
            desc="Val  ",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5
        )
        
        with torch.no_grad():
            for signals, labels in val_pbar:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(signals)
                loss = self.criterion(logits, labels)
                
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                metrics.update(loss.item(), logits, labels)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{metrics.accuracies[-1]:.4f}")
        
        # Compute comprehensive metrics
        y_true = np.array(metrics.all_labels)
        y_pred = np.array(metrics.all_preds)
        y_proba = np.concatenate(all_probs, axis=0)
        
        detailed_metrics = self.metrics_computer.compute_metrics(y_true, y_pred, y_proba)
        
        epoch_metrics = {
            'loss': np.mean(metrics.losses),
            'accuracy': detailed_metrics['accuracy'],
            'f1': detailed_metrics['f1_weighted'],
            'precision': detailed_metrics['precision_weighted'],
            'recall': detailed_metrics['recall_weighted']
        }
        
        return epoch_metrics, y_true, y_pred, y_proba
    
    def train(self):
        """Main training loop."""
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        
        best_val_loss = float('inf')
        
        epoch_pbar = tqdm(range(self.cfg['num_epochs']), desc="Epochs", dynamic_ncols=True)

        for epoch in epoch_pbar:
            logger.info(f"\nEpoch {epoch + 1}/{self.cfg['num_epochs']}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['train_f1'].append(train_metrics.get('f1', 0))
            self.history['train_precision'].append(train_metrics.get('precision', 0))
            self.history['train_recall'].append(train_metrics.get('recall', 0))
            
            # Validate
            val_metrics, y_true_val, y_pred_val, y_proba_val = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            epoch_pbar.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                val_acc=f"{val_metrics['accuracy']:.4f}"
            )
            
            logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f} | Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            
            # Save checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    MODELS_DIR / 'best_model.pt'
                )
                logger.info("✓ Best model saved")
            
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    MODELS_DIR / f'checkpoint_epoch_{epoch + 1}.pt'
                )

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if self.early_stopping.check(val_metrics['loss'], epoch):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Visualize training curves
        logger.info("Generating training visualizations...")
        self.train_viz.plot_training_curves(self.history)
        self.train_viz.plot_all_metrics(self.history)
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
    
    def evaluate_on_test(self):
        """Evaluate on test set."""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*80)
        
        # Load best model
        logger.info("Loading best model...")
        best_model_path = MODELS_DIR / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in self.test_loader:
                signals = signals.to(self.device)
                
                logits, _ = self.model(signals)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_proba = np.concatenate(all_probs, axis=0)
        
        # Compute metrics
        test_metrics = self.metrics_computer.compute_metrics(y_true, y_pred, y_proba)
        
        logger.info("\nTEST RESULTS:")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision (Macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"Recall (Macro): {test_metrics['recall_macro']:.4f}")
        logger.info(f"F1 (Macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"Precision (Weighted): {test_metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (Weighted): {test_metrics['recall_weighted']:.4f}")
        logger.info(f"F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
        if 'roc_auc' in test_metrics:
            logger.info(f"ROC-AUC (Macro): {test_metrics['roc_auc']:.4f}")
        
        logger.info("\nPER-CLASS METRICS:")
        logger.info(self.metrics_computer.get_classification_report(y_true, y_pred))
        
        # Visualizations
        logger.info("\nGenerating evaluation visualizations...")
        
        # Confusion matrix
        cm = self.metrics_computer.get_confusion_matrix(y_true, y_pred)
        self.eval_viz.plot_confusion_matrix(cm, list(self.cfg['class_names'].values()), normalize=False)
        self.eval_viz.plot_confusion_matrix(cm, list(self.cfg['class_names'].values()), normalize=True, save_name='confusion_matrix_normalized.png')
        
        # Per-class metrics
        self.eval_viz.plot_per_class_metrics(test_metrics['per_class'])
        
        # Metrics comparison
        summary_metrics = {
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision_weighted'],
            'Recall': test_metrics['recall_weighted'],
            'F1-Score': test_metrics['f1_weighted']
        }
        self.eval_viz.plot_metrics_comparison(summary_metrics)

        # Explainability: save attention visualization for the highest-confidence sample per predicted class
        try:
            sample_loader = self.data_module.get_test_dataloader(batch_size=1, num_workers=NUM_WORKERS)
            saved_classes = set()
            with torch.no_grad():
                for signals, labels in sample_loader:
                    signals = signals.to(self.device)
                    logits, attention_weights = self.model(signals, return_attention=True)
                    probabilities = torch.softmax(logits, dim=1)
                    pred_class = int(torch.argmax(probabilities, dim=1).item())
                    if pred_class in saved_classes or attention_weights is None:
                        continue
                    saved_classes.add(pred_class)
                    attention_map = attention_weights[0].mean(dim=0).detach().cpu().numpy()
                    signal = signals[0].detach().cpu().numpy().squeeze()
                    class_name = self.cfg['class_names'].get(pred_class, f'Class_{pred_class}')
                    self.signal_viz.plot_signal_with_attention(
                        signal,
                        attention_map.mean(axis=0),
                        class_name,
                        save_name=f'attention_{pred_class}_{class_name.replace(" ", "_").lower()}.png'
                    )
                    if len(saved_classes) == self.cfg['num_classes']:
                        break
        except Exception as exc:
            logger.warning(f"Attention visualization skipped: {exc}")
        
        # Save results
        results = {
            'model_config': {
                'num_classes': self.cfg['num_classes'],
                'input_length': SIGNAL_LENGTH,
                'parameters': count_parameters(self.model)
            },
            'test_metrics': test_metrics,
            'history': self.history
        }
        
        save_results(results, RESULTS_DIR / 'test_results.json')
        logger.info(f"✓ Results saved to {RESULTS_DIR / 'test_results.json'}")
        
        logger.info("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    try:
        parser = argparse.ArgumentParser(description='ECG multi-class training and audit runner')
        parser.add_argument('--audit-only', action='store_true', help='Run the pre-training audit and stop')
        parser.add_argument('--skip-audit', action='store_true', help='Skip the pre-training audit and start training immediately')
        parser.add_argument('--max-train-samples', type=int, default=MAX_TRAIN_SAMPLES, help='Limit training samples for a faster run')
        parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
        parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
        args = parser.parse_args()

        # Create trainer
        trainer = ECGTrainer(config_dict={
            'device': DEVICE,
            'num_epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': LEARNING_RATE,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'class_weights': CLASS_WEIGHTS,
            'max_train_samples': args.max_train_samples,
        })

        if not args.skip_audit:
            audit_report = trainer.run_pre_training_audit()
            save_results(audit_report, RESULTS_DIR / 'pre_training_audit.json')
            logger.info(f"✓ Pre-training audit saved to {RESULTS_DIR / 'pre_training_audit.json'}")

            if args.audit_only:
                return
        elif args.audit_only:
            return
        
        # Train
        trainer.train()
        
        # Evaluate
        trainer.evaluate_on_test()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
