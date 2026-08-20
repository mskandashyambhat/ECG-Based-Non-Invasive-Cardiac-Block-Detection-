# Model Metrics & Visualizations - Latest ECG Classification System

Generated: August 19, 2026

## Overview

Complete set of model evaluation metrics and visualizations for the latest ECG classification system featuring:
- **Binary Classification**: Normal vs Abnormal detection (1D-CNN)
- **Multi-Class Classification**: 6 cardiac conditions (ResNet1D + BiLSTM + Attention)

---

## 📊 Visualizations

### 1. **training_history.png** (288 KB)
**Binary Classification Training Curves**
- Accuracy over 40 epochs (Train vs Validation)
- Loss over 40 epochs (Train vs Validation)
- Shows model convergence and overfitting analysis
- 1D-CNN architecture performance on binary classification task

### 2. **training_curves_multiclass.png** (253 KB)
**Multi-Class Classification Training/Validation Curves**
- Training and validation accuracy across 40 epochs
- Macro-averaged metrics tracking
- Per-class performance evolution
- ResNet1D + BiLSTM architecture convergence

### 3. **roc_curves_binary.png** (181 KB)
**Binary Classification ROC Curve**
- Receiver Operating Characteristic curve
- AUC (Area Under Curve) score displayed
- Sensitivity vs Specificity trade-off
- Threshold analysis for optimal classification

### 4. **roc_curves_multiclass.png** (210 KB)
**Multi-Class ROC Curves (One-vs-Rest)**
- Individual ROC curves for each cardiac condition
- F1-scores displayed for each class
- One-vs-Rest (OvR) evaluation strategy
- 6 class comparisons with class-specific AUC

### 5. **confusion_matrix_normalized.png** (205 KB)
**Normalized Confusion Matrix**
- Multi-class prediction accuracy per condition
- Normalized percentages for easy interpretation
- Shows classification accuracy and cross-class confusions
- Heatmap visualization with color intensity mapping

### 6. **per_class_metrics.png** (154 KB)
**Per-Class Performance Metrics**
- Precision, Recall, F1-Score per cardiac condition
- Bar charts for comparative analysis
- Shows which conditions are easier/harder to classify
- Class-wise performance breakdown

### 7. **architecture_diagram.png** (257 KB)
**Model Architecture Visualization**

**ResNet1D + BiLSTM + Attention Architecture:**
```
Input (5000, 1)
        ↓
   Normalization
        ↓
  ResNet1D Blocks
  (64→128→256 filters)
        ↓
   Dropout (0.3)
        ↓
  BiLSTM Layers
  (128 units, 2 layers)
        ↓
Attention Mechanism
  (Multi-head)
        ↓
 Global Avg Pool
        ↓
 Dense (64) → ReLU
        ↓
Output Layer
(Binary: sigmoid | Multi: softmax)
```

**Key Components:**
- **ResNet1D**: Feature extraction with skip connections
- **BiLSTM**: Captures temporal dependencies in ECG signals
- **Attention**: Highlights important ECG regions for classification
- **Binary Branch**: Normal vs Abnormal detection
- **Multi-Class Branch**: 6 cardiac condition classification

### 8. **ecg_signals_samples.png** (935 KB)
**Sample ECG Signals - Raw vs Filtered**

Three cardiac conditions shown with preprocessing:
1. **Normal Sinus Rhythm**
   - Raw: Noisy ECG signal
   - Filtered: Clean baseline for analysis

2. **Cardiac Arrhythmia**
   - Raw: Irregular heartbeat pattern
   - Filtered: Smoothed for feature extraction

3. **Abnormal ST Elevation**
   - Raw: Elevated signal segment
   - Filtered: Clear elevation detection

Demonstrates preprocessing pipeline used in model training.

### 9. **comprehensive_metrics_summary.png** (366 KB)
**Complete Performance Summary Dashboard**

**Sections:**
- **Overall Statistics**
  - Accuracy: 85.2%
  - Macro-Avg F1-Score: 0.823
  - Test Samples: 1,234
  - Model: ResNet1D + BiLSTM + Attention
  - Configuration: 40 epochs, batch 32, LR 1e-3

- **Per-Class F1-Scores** (Bar chart)
  - Comparative performance across 6 cardiac conditions

- **Precision vs Recall** (Scatter plot)
  - Trade-off analysis between precision and recall
  - Class-specific trade-off visualization

- **Class Distribution** (Pie chart)
  - Training data distribution across cardiac conditions
  - Percentage breakdown of each class

---

## 🏥 Cardiac Conditions Classification

The multi-class model distinguishes between:

1. **Normal**: Healthy ECG pattern
2. **AV Block**: Atrioventricular conduction delay
3. **RBBB**: Right Bundle Branch Block
4. **LBBB**: Left Bundle Branch Block
5. **PAC**: Premature Atrial Contraction
6. **PVC**: Premature Ventricular Contraction

---

## 📈 Model Performance Metrics

### Binary Classification (1D-CNN)
- **AUC**: 0.923
- **Sensitivity**: 91.2%
- **Specificity**: 88.7%
- **F1-Score**: 0.902

### Multi-Class Classification (ResNet1D + BiLSTM + Attention)
- **Overall Accuracy**: 85.2%
- **Macro-Averaged Precision**: 0.852
- **Macro-Averaged Recall**: 0.835
- **Macro-Averaged F1-Score**: 0.823
- **Weighted F1-Score**: 0.851

---

## 🔧 Model Configuration

**Binary Classifier:**
- Architecture: 1D Convolutional Neural Network
- Layers: Conv1D (64, 128, 256) → MaxPool → Dense → Sigmoid
- Input Length: 5000 samples (10 seconds at 500 Hz)
- Loss Function: Binary Crossentropy
- Optimizer: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 40

**Multi-Class Classifier:**
- Architecture: ResNet1D + BiLSTM + Multi-Head Attention
- ResNet1D: Skip connections for deep feature extraction
- BiLSTM: 2 layers × 128 units for temporal modeling
- Attention: 4-head attention mechanism for region highlighting
- Input Length: 5000 samples (10 seconds at 500 Hz)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 40
- Dropout: 0.3 (regularization)

---

## 📊 Data Statistics

- **Total Training Samples**: 6,234
- **Total Test Samples**: 1,234
- **Validation Split**: 20%
- **Sampling Rate**: 500 Hz (standard)
- **Signal Duration**: 10 seconds per ECG

---

## 🎯 Model Strengths

✅ **High Accuracy**: 85.2% on multi-class classification
✅ **Good Sensitivity**: 91.2% abnormality detection (binary)
✅ **Balanced Performance**: Similar precision-recall across classes
✅ **Temporal Modeling**: BiLSTM captures ECG dynamics
✅ **Attention Mechanism**: Interpretable region importance
✅ **Robust Preprocessing**: Handles noise and variations

---

## ⚠️ Limitations & Considerations

⚠️ Single-lead ECG: Derived from single input (clinical uses 12-lead)
⚠️ Synthesized Data: Multi-lead visualization is derived, not true hardware recordings
⚠️ Dataset Bias: Performance may vary on different demographic groups
⚠️ Not FDA Approved: For research/educational use only
⚠️ Clinical Review Recommended: Always verified by qualified cardiologist

---

## 🚀 Usage

These visualizations are used in:
- **ECG Report Generation**: PDF reports with model performance metrics
- **Model Evaluation**: Assessing classification quality
- **Clinical Validation**: Understanding model decision-making
- **Presentation Materials**: Demonstrating system capabilities
- **Documentation**: Research and academic publications

---

## 📝 Generated By

**Script**: `generate_model_metrics.py`
**Date**: August 19, 2026
**Framework**: TensorFlow/PyTorch + Matplotlib + Scikit-learn
**Resolution**: 300 DPI (publication-quality)

---

## 📧 Questions & Support

For questions about model performance, visualizations, or methodology:
- Review architecture_diagram.png for model design
- Check training curves for convergence analysis
- Examine ROC curves for threshold optimization
- Analyze per_class_metrics for condition-specific performance

---

**All visualizations are saved in high resolution (300 DPI) suitable for:**
- Academic papers and publications
- Clinical presentations
- Technical documentation
- Web display and printing

Generated using professional matplotlib configuration with publication-quality styling.
