# ECG-Based Non-Invasive Block Detection and Classification Framework

## Project Overview

This project proposes an AI-driven ECG interpretation system where a patient can upload an ECG signal, and the model automatically analyzes waveform patterns to identify conduction blocks, arrhythmias, and major cardiac abnormalities.

This work develops an ECG-based Non-Invasive cardiac block detection and classification framework using deep learning.

### Key Capabilities

The system provides multi-class classification across 5 categories:
- **Class 0**: Normal ECG
- **Class 1**: AV Block (1st, 2nd, 3rd degree)
- **Class 2**: Complete Heart Block
- **Class 3**: Right Bundle Branch Block (RBBB)
- **Class 4**: Left Bundle Branch Block (LBBB)

### Expected Output

When a patient uploads an ECG, the system generates:
- Block Detected (Yes/No)
- Block Type (First/Mobitz I/Mobitz II/Complete)
- Location (AV Node/RBBB/LBBB)
- Severity (Mild/Moderate/Severe/Critical)
- Arrhythmia Detected (AF/PVC/SVT/None)
- MI Pattern (Acute/Inferior/Anterior/Not Detected)
- Interval Report (PR, QRS, QT measurements)

---

## Architecture

### Hybrid Deep Learning Model

The system employs a state-of-the-art hybrid architecture:

```
ECG Input Signal (300 samples, 500 Hz, Lead II)
           ↓
    ┌──────────────────┐
    │  ResNet1D        │  (Feature Extraction)
    │  Backbone        │  [64, 128, 256, 512 channels]
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  BiLSTM          │  (Temporal Dependencies)
    │  2 layers, 256D  │
    │  Bidirectional   │
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  Multi-Head      │  (Important Regions)
    │  Attention       │  8 attention heads
    │  (Self-Attention)│
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  Global Avg Pool │  (Temporal Aggregation)
    └──────────────────┘
           ↓
    ┌──────────────────┐
    │  Dense Layers    │  (Classification)
    │  512 → 256 → 128 │
    │  + BatchNorm     │
    │  + Dropout       │
    └──────────────────┘
           ↓
      Output (5 classes)
```

### Model Components

**ResNet1D Backbone**
- 1D residual blocks adapted for time-series ECG signals
- Progressive downsampling (stride 2 after each stage)
- Batch normalization and skip connections
- Total depth: 18 layers (ResNet18 variant)

**BiLSTM Temporal Processing**
- 2-layer bidirectional LSTM
- 256-dimensional hidden state
- Captures long-range temporal dependencies in ECG rhythm

**Multi-Head Self-Attention**
- 8 attention heads
- Learns which ECG regions are important for classification
- Position-independent relevance assessment
- Produces explainable attention maps

**Fully Connected Classifier**
- 3-layer dense network with progressive dimensionality reduction
- Batch normalization and dropout for regularization
- Final softmax layer for multi-class probabilities

---

## Dataset

### Source Datasets

- **PTB-XL ECG Dataset**: ~21,800 clinical 12-lead ECG records
- **PhysioNet ECG Arrhythmia Dataset**: ~10,000+ ECG recordings
- **CPSC 2018 ECG Challenge Dataset**: ~6,877 ECG recordings
- **MIT-BIH Arrhythmia Database**: 48 half-hour ECG recordings
- **Lobachevsky University ECG Database**: 200 annotated 12-lead ECG recordings

### Data Characteristics

- **Signal Length**: 300 samples @ 500 Hz (0.6 seconds)
- **Lead**: Lead II (single-lead)
- **Preprocessing**: Bandpass filtering, notch filtering, R-peak detection, Z-score normalization
- **Train/Val/Test Split**: 70% / 15% / 15% (stratified)
- **Class Weights**: Applied for imbalanced data handling

---

## Training

### Configuration

Key training parameters (see `config.py`):

```python
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
USE_AMP = True  # Mixed Precision Training
```

### Training Pipeline

1. **Mixed Precision Training**: Automatic mixed precision with gradient scaling
2. **Gradient Clipping**: Prevent exploding gradients
3. **Learning Rate Scheduling**: Cosine annealing with warm restarts
4. **Early Stopping**: Monitor validation loss with patience=15
5. **Weighted Sampling**: Handle class imbalance during training
6. **Best Model Checkpointing**: Save model with lowest validation loss

### Optimization Details

- **Optimizer**: AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Loss Function**: Cross-Entropy Loss with label smoothing
- **Class Weights**: Applied to handle imbalanced classes

### Data Augmentation

Optional augmentations (can be enabled in config):
- Gaussian noise addition
- Time stretching/compression
- Amplitude scaling
- Cutout (temporal masking)

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support) or CPU
- PyTorch 2.0+

### Setup

```bash
# Navigate to project directory
cd Multi_Class_Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
# Start training from scratch
python train.py

# The script will:
# 1. Load and preprocess data
# 2. Initialize model and optimizer
# 3. Train for up to 100 epochs
# 4. Save best model to output/models/best_model.pt
# 5. Generate visualizations in output/visualizations/
# 6. Save results to output/results/test_results.json
```

**Training Output**:
- `output/models/best_model.pt` - Best model checkpoint
- `output/models/checkpoint_epoch_N.pt` - Periodic checkpoints
- `output/results/test_results.json` - Comprehensive metrics
- `output/visualizations/training_curves.png` - Training history
- `output/visualizations/confusion_matrix.png` - Test confusion matrix
- `output/visualizations/per_class_metrics.png` - Per-class precision/recall/F1

### Inference

```bash
# Single signal prediction
python predict.py --model output/models/best_model.pt --signal path/to/signal.npy

# Batch prediction
python predict.py --model output/models/best_model.pt --signal path/to/batch_signals.npz

# Specify device
python predict.py --model output/models/best_model.pt --signal data.npy --device cuda
```

**Example Python Usage**:

```python
from predict import ECGInferencer
import numpy as np

# Load inferencer
inferencer = ECGInferencer('./output/models/best_model.pt')

# Load ECG signal
signal = np.load('ecg_signal.npy')

# Make prediction
result = inferencer.predict(signal, return_attention=True)

print(f"Predicted: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
```

---

## Metrics and Evaluation

### Metrics Computed

- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise and weighted average
- **Recall**: Class-wise and weighted average
- **F1-Score**: Macro and weighted averages
- **ROC-AUC**: Multi-class ROC-AUC (one-vs-rest)
- **Confusion Matrix**: Detailed prediction errors
- **Per-Class Metrics**: Individual precision/recall/F1

### Visualization Outputs

- Training loss and accuracy curves
- Confusion matrix (normalized and unnormalized)
- Per-class metrics bar charts
- Metrics comparison radar chart
- ROC curves for each class
- Precision-Recall curves

---

## Model Explainability

### Attention Visualization

The model produces interpretable attention weights showing which ECG regions influenced the prediction:

```python
result = inferencer.predict(signal, return_attention=True)
attention_weights = result['attention_weights']
# Shape: [num_heads, sequence_length, sequence_length]
```

### Feature Map Extraction

Access intermediate feature representations:

```python
features = inferencer.get_intermediate_features(signal)
# Contains: resnet_features, bilstm_features, attention_features
```

---

## Project Structure

```
Multi_Class_Classification/
├── config.py                 # Configuration and hyperparameters
├── model.py                  # Model architecture (ResNet1D + BiLSTM + Attention)
├── attention.py              # Multi-Head Attention modules
├── losses.py                 # Custom loss functions (Focal, Label Smoothing, etc.)
├── metrics.py                # Metrics computation
├── dataset.py                # Data loading and preprocessing
├── train.py                  # Main training script
├── predict.py                # Inference script
├── visualization.py          # Plotting and visualization
├── utils.py                  # Utility functions
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── output/
    ├── models/              # Saved model checkpoints
    ├── results/             # JSON results and metrics
    ├── visualizations/      # PNG plots
    └── logs/                # Training logs
```

---

## Key Features

✅ **State-of-the-art Architecture**
- Hybrid design combining CNN, RNN, and Attention
- Proven effective for time-series classification
- Suitable for medical signal processing

✅ **Production-Ready Code**
- Modular, well-documented architecture
- Comprehensive error handling
- Efficient data loading and preprocessing
- GPU optimization with mixed precision training

✅ **Explainability**
- Attention weight visualization
- Feature map extraction
- Per-class prediction confidence
- Attention-weighted ECG signal plots

✅ **Scalability**
- Handles large datasets efficiently
- Batch prediction support
- Model variants (small, base, large)
- Multi-GPU training compatible

✅ **Research Quality**
- Comprehensive metrics and evaluation
- Publication-ready visualizations
- Reproducible results (fixed random seed)
- Detailed logging and tracking

---

## Performance

Expected performance on test set:
- **Accuracy**: >95% (5-class classification)
- **Macro F1**: >92%
- **Weighted F1**: >94%
- **Per-class ROC-AUC**: >0.98

*Note: Final performance depends on data quality, preprocessing, and hyperparameter tuning.*

---

## Advanced Configuration

### Custom Model Variants

```python
from model import hybrid_ecg_small, hybrid_ecg_base, hybrid_ecg_large

# Small model (fewer parameters, faster inference)
model = hybrid_ecg_small(num_classes=5)

# Base model (balanced)
model = hybrid_ecg_base(num_classes=5)

# Large model (maximum capacity)
model = hybrid_ecg_large(num_classes=5)
```

### Custom Loss Functions

```python
from losses import get_loss_function

# Focal Loss (for imbalanced data)
loss_fn = get_loss_function('focal', num_classes=5, alpha=0.25, gamma=2.0)

# Label Smoothing
loss_fn = get_loss_function('label_smoothing', num_classes=5, smoothing=0.1)

# Combined (CE + Focal)
loss_fn = get_loss_function('combined', num_classes=5)
```

---

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size in config.py
BATCH_SIZE = 32  # or lower
```

### Poor Performance
- Verify data preprocessing is applied correctly
- Check class distribution (may need weighted sampling)
- Increase number of training epochs
- Adjust learning rate or use different scheduler

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

---

## References

### Key Related Work

- Wagner et al. (2020): PTB-XL large-scale ECG classification
- Rajpurkar et al. (2017): Cardiologist-level arrhythmia detection with 34-layer CNN
- Hannun et al. (2019): Deep CNN for clinical ECG interpretation
- Lin et al. (2017): Focal Loss for addressing class imbalance

### Datasets Used

- PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
- PhysioNet Arrhythmia: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- CPSC 2018: https://physionet.org/content/challenge-2018/1.0.0/

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ecg_block_detection_2024,
  title={ECG-Based Non-Invasive Block Detection and Classification Framework},
  author={Your Name},
  year={2024}
}
```

---

## License

This project is provided for educational and research purposes.

---

**Version**: 1.0.0  
**Status**: Production-Ready  
**Last Updated**: August 2026
