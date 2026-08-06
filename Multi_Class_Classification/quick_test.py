"""
Quick test script to verify all components are working correctly.
Run this before full training to catch any issues early.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from model import HybridECGModel, hybrid_ecg_small, hybrid_ecg_base, hybrid_ecg_large
from dataset import ECGDataModule
from losses import get_loss_function
from attention import MultiHeadSelfAttention
from utils import set_seed, get_device, count_parameters
from metrics import MetricsComputer

def test_device():
    """Test device configuration."""
    print("\n" + "="*80)
    print("TEST 1: Device Configuration")
    print("="*80)
    
    device = get_device(DEVICE)
    print(f"✓ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: True")
        print(f"✓ Device Count: {torch.cuda.device_count()}")
        print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")
    
    return device

def test_model_creation(device):
    """Test model creation and forward pass."""
    print("\n" + "="*80)
    print("TEST 2: Model Creation & Architecture")
    print("="*80)
    
    # Create model
    model = hybrid_ecg_base(num_classes=NUM_CLASSES).to(device)
    params = count_parameters(model)
    print(f"✓ Model created with {params:,} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(4, SIGNAL_LENGTH).to(device)
    logits, attn = model(dummy_input, return_attention=True)
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Attention weights shape: {attn.shape if attn is not None else 'None'}")
    
    # Test model variants
    small = hybrid_ecg_small(num_classes=NUM_CLASSES).to(device)
    large = hybrid_ecg_large(num_classes=NUM_CLASSES).to(device)
    print(f"✓ Small model: {count_parameters(small):,} parameters")
    print(f"✓ Base model: {params:,} parameters")
    print(f"✓ Large model: {count_parameters(large):,} parameters")
    
    return model

def test_attention_module(device):
    """Test attention mechanism."""
    print("\n" + "="*80)
    print("TEST 3: Attention Mechanisms")
    print("="*80)
    
    # Create attention module
    attn = MultiHeadSelfAttention(hidden_dim=512, num_heads=8, dropout=0.1).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 37, 512).to(device)  # [batch, seq_len, hidden_dim]
    output, weights = attn(dummy_input)
    
    print(f"✓ Attention input shape: {dummy_input.shape}")
    print(f"✓ Attention output shape: {output.shape}")
    print(f"✓ Attention weights shape: {weights.shape}")
    print(f"✓ Parameters: {count_parameters(attn):,}")

def test_losses():
    """Test loss functions."""
    print("\n" + "="*80)
    print("TEST 4: Loss Functions")
    print("="*80)
    
    criterion_ce = get_loss_function('cross_entropy', num_classes=NUM_CLASSES)
    criterion_focal = get_loss_function('focal', num_classes=NUM_CLASSES)
    criterion_ls = get_loss_function('label_smoothing', num_classes=NUM_CLASSES)
    
    # Test forward pass
    dummy_logits = torch.randn(32, NUM_CLASSES)
    dummy_labels = torch.randint(0, NUM_CLASSES, (32,))
    
    loss_ce = criterion_ce(dummy_logits, dummy_labels)
    loss_focal = criterion_focal(dummy_logits, dummy_labels)
    loss_ls = criterion_ls(dummy_logits, dummy_labels)
    
    print(f"✓ Cross-Entropy Loss: {loss_ce.item():.4f}")
    print(f"✓ Focal Loss: {loss_focal.item():.4f}")
    print(f"✓ Label Smoothing Loss: {loss_ls.item():.4f}")

def test_data_loading():
    """Test data module."""
    print("\n" + "="*80)
    print("TEST 5: Data Loading & Processing")
    print("="*80)
    
    if not PREPROCESSED_DATA.exists():
        print(f"⚠ Dataset not found at {PREPROCESSED_DATA}")
        print("  Skipping data loading test")
        return
    
    data_module = ECGDataModule(str(PREPROCESSED_DATA))
    data_module.prepare_data()
    
    print(f"✓ Data loaded successfully")
    print(f"✓ Train set: {data_module.X_train.shape[0]:,} samples")
    print(f"✓ Val set: {data_module.X_val.shape[0]:,} samples")
    print(f"✓ Test set: {data_module.X_test.shape[0]:,} samples")
    print(f"✓ Num classes: {data_module.num_classes}")
    print(f"✓ Class distribution: {data_module.class_distribution}")
    
    # Test data loader
    train_loader = data_module.get_train_dataloader(batch_size=32, num_workers=0)
    batch_signals, batch_labels = next(iter(train_loader))
    print(f"✓ Batch signals shape: {batch_signals.shape}")
    print(f"✓ Batch labels shape: {batch_labels.shape}")

def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*80)
    print("TEST 6: Metrics Computation")
    print("="*80)
    
    metrics_computer = MetricsComputer(num_classes=NUM_CLASSES, class_names=CLASS_NAMES)
    
    # Generate dummy predictions
    y_true = np.random.randint(0, NUM_CLASSES, 100)
    y_pred = np.random.randint(0, NUM_CLASSES, 100)
    y_proba = np.random.dirichlet(np.ones(NUM_CLASSES), 100)
    
    metrics = metrics_computer.compute_metrics(y_true, y_pred, y_proba)
    
    print(f"✓ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"✓ Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"✓ F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ Confusion Matrix shape: {np.array(metrics['confusion_matrix']).shape}")

def test_output_directories():
    """Test output directory creation."""
    print("\n" + "="*80)
    print("TEST 7: Output Directories")
    print("="*80)
    
    directories = [MODELS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR, LOGS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        exists = directory.exists()
        print(f"✓ {directory.name}: {exists}")

def test_device_memory(device, model):
    """Test device memory."""
    print("\n" + "="*80)
    print("TEST 8: Device Memory")
    print("="*80)
    
    if device.type == 'cuda':
        print(f"✓ GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"✓ GPU Memory Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        
        # Test forward pass memory
        dummy_input = torch.randn(32, SIGNAL_LENGTH).to(device)
        _ = model(dummy_input)
        
        print(f"✓ GPU Memory After Forward Pass: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    else:
        print("✓ Using CPU (memory tracking not available)")

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("ECG MULTI-CLASS MODEL - QUICK TEST SUITE")
    print("="*80)
    
    try:
        # Setup
        set_seed(RANDOM_SEED)
        device = test_device()
        
        # Tests
        model = test_model_creation(device)
        test_attention_module(device)
        test_losses()
        test_data_loading()
        test_metrics()
        test_output_directories()
        test_device_memory(device, model)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nSystem is ready for training!")
        print(f"Run: python train.py")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    run_all_tests()
