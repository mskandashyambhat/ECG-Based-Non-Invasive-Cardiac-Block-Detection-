"""
Configuration module for multi-class ECG block detection system.
Centralized configuration management for all hyperparameters and paths.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
PREPROCESSED_DATA = DATASET_ROOT / "preprocessed_dataset" / "merged_ecg_dataset_all5_complete.npz"

# ==========================================================================
# TRAINING CONFIGURATION
# ==========================================================================
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Output directories

# Fast training controls
MAX_TRAIN_SAMPLES = 120000
OUTPUT_DIR = Path(__file__).parent / "output"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Ensure directories exist
for directory in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
SIGNAL_LENGTH = 300  # ECG segment length in samples
SAMPLING_RATE = 500  # Hz
NUM_CLASSES = 5
NUM_LEADS = 1  # Lead II only

CLASS_NAMES = {
    0: "Normal",
    1: "AV Block",
    2: "Complete Heart Block",
    3: "RBBB",
    4: "LBBB"
}

CLASS_WEIGHTS = {
    0: 1.0,      # Normal
    1: 2.0,      # AV Block
    2: 3.0,      # Complete Heart Block
    3: 2.5,      # RBBB
    4: 2.5       # LBBB
}

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_SEED = 42
STRATIFIED_SPLIT = True

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# ResNet1D Backbone
RESNET_BLOCKS = [64, 128, 256, 512]
KERNEL_SIZE = 7
STRIDE = 1
PADDING = 3

# BiLSTM
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True

# Multi-Head Attention
NUM_ATTENTION_HEADS = 8
ATTENTION_DROPOUT = 0.2

# Dense Layers
DENSE_DIMS = [512, 256, 128]
DROPOUT_RATES = [0.3, 0.25, 0.2]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 128  # ✅ BALANCED: 3M samples ÷ 128 = ~12K batches/epoch (much better)
NUM_EPOCHS = 5  # ✅ MINIMAL: Only 5 epochs with early stopping patience 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0

# Optimization
OPTIMIZER = "adamw"  # adamw, adam, sgd
SCHEDULER = "plateau"  # plateau, cosine, step, exponential
WARMUP_EPOCHS = 1

# Mixed Precision Training
USE_AMP = True
SCALER_INIT_SCALE = 65536.0
SCALER_GROWTH_FACTOR = 2.0
SCALER_BACKOFF_FACTOR = 0.5
SCALER_GROWTH_INTERVAL = 2000

# ============================================================================
# LOSS CONFIGURATION
# ============================================================================
LOSS_TYPE = "cross_entropy"  # cross_entropy, focal
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# ============================================================================
# REGULARIZATION
# ============================================================================
USE_MIXUP = True
MIXUP_ALPHA = 0.2

USE_CUTMIX = False
CUTMIX_ALPHA = 1.0

USE_DROPOUT = True
DROPOUT_BASE = 0.3

USE_L2 = True
L2_WEIGHT_DECAY = 1e-5

# ============================================================================
# EARLY STOPPING & LR SCHEDULING
# ============================================================================
EARLY_STOP_PATIENCE = 3
EARLY_STOP_METRIC = "val_loss"  # val_loss, val_accuracy
EARLY_STOP_MIN_DELTA = 1e-4

# ReduceLROnPlateau
REDUCE_LR_PATIENCE = 1
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = "cpu"  # ✅ Use CPU (stable, still ~3-4 hours for 20 epochs)
NUM_WORKERS = 0  # ✅ No multiprocessing on CPU - dataloader construction is causing hangs
PIN_MEMORY = False  # Not needed on Mac
PREFETCH_FACTOR = 1  # Reduced for 16GB RAM

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
COMPUTE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
    "per_class_accuracy",
    "macro_f1",
    "weighted_f1"
]

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
DPI = 300
FIGSIZE = (14, 6)
CMAP = "viridis"

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================
SAVE_BEST_ONLY = True
SAVE_LAST = True
CHECKPOINT_INTERVAL = 5  # Save every N epochs

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_BATCH_SIZE = 128
INFERENCE_DEVICE = "cuda"
SAVE_PROBABILITIES = True
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
LOG_INTERVAL = 1000  # ✅ Log every 1000 batches (less verbose, faster)
