# ECG-Based Non-Invasive Heart Block Detection

A comprehensive deep learning system for detecting and classifying cardiac arrhythmias (heart blocks) from ECG signals using multiple electrocardiography datasets.

## 🏥 Overview

This project implements an automated heart block detection system using ECG signals. It processes and integrates multiple publicly available ECG datasets and applies machine learning techniques to classify three heart block conditions:

- **Normal**: No cardiac abnormalities
- **First-degree AV Block**: Delayed electrical conduction
- **Complete Heart Block**: Blocked electrical signal transmission

## 📊 Datasets

The project integrates ECG data from multiple sources:

1. **PTB-XL** - 21,799 records (251,315 segments)
   - German multi-center ECG database
   - 10-second recordings at 500 Hz sampling rate
   
2. **MIT-BIH Arrhythmia Database** - 48 records (110,924 segments)
   - Boston cardiac arrhythmia database
   - 30-minute continuous recordings
   
3. **Lobachevsky University ECG Database (LUDB)** - 200 records (2,133 segments)
   - Russian diagnostic ECG database
   - High-quality 12-lead recordings

**Total**: 22,047 records, 364,372 ECG segments merged and balanced

## 📁 Project Structure

```
Project/
├── dataset_analysis.py              # Comprehensive dataset analysis script
├── Binary_Classification/           # Binary classification models
├── Dataset/
│   ├── ecg_preprocessing_pipeline.py          # Main preprocessing pipeline
│   ├── ecg_preprocessing_pipeline_all5.py     # Extended preprocessing (5 classes)
│   ├── preprocessed_dataset/                  # Output directory
│   │   ├── merged_ecg_dataset_all5_complete.npz
│   │   ├── PREPROCESSING_SUMMARY.md
│   │   ├── DATASET_DOCUMENTATION.md
│   │   ├── USAGE_GUIDE.md
│   │   └── README.md
│   └── Unprocessed_Datasets/        # Raw dataset storage
│       ├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
│       ├── mit-bih-arrhythmia-database-1.0.0/
│       ├── lobachevsky-university-electrocardiography-database-1.0.1/
│       └── ptb-diagnostic-ecg-database-1.0.0/
└── README.md                        # This file
```

## 🛠️ Technologies

- **Python 3.x**
- **NumPy & SciPy** - Numerical computing
- **pandas** - Data manipulation
- **wfdb** - ECG signal reading and processing
- **scikit-learn** - Machine learning
- **TensorFlow/Keras** - Deep learning (if applicable)

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scipy pandas wfdb scikit-learn tqdm
```

### Data Preprocessing

Run the preprocessing pipeline to generate the merged and balanced dataset:

```bash
python Dataset/ecg_preprocessing_pipeline.py
```

Or for extended classification (5 classes):

```bash
python Dataset/ecg_preprocessing_pipeline_all5.py
```

### Dataset Analysis

Analyze the processed datasets:

```bash
python dataset_analysis.py
```

## 📈 Dataset Processing

The preprocessing pipeline:

1. **Loads** ECG signals from multiple databases (WFDB format)
2. **Extracts** diagnostic labels and annotations
3. **Segments** long recordings into fixed-length windows (300 samples)
4. **Normalizes** signal amplitudes and sampling rates
5. **Balances** classes to prevent bias
6. **Saves** output in both pickle and NPZ formats

### Output Format

The preprocessed dataset is saved as:
- **merged_ecg_dataset.pkl** - Python pickle format (2.5 GB)
- **merged_ecg_dataset.npz** - NumPy compressed format (2.4 GB)

Each contains:
- `X`: ECG segments (1,091,484 × 300 float32)
- `y`: Class labels (1,091,484 int64)
- `features`: Interval features (PR, QRS, RR, HR)
- `class_names`: Label mapping dictionary

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Segments | 1,091,484 |
| Segment Length | 300 samples (0.6s @ 500Hz) |
| Total Classes | 3 |
| Class Distribution | Perfectly balanced (33.33% each) |
| Data Format | Float32 (ECG), Int64 (labels) |

## 🔍 Features

- Multi-database ECG integration
- Automatic class balancing
- Signal preprocessing and normalization
- Comprehensive dataset analysis tools
- Support for multiple output formats
- Detailed preprocessing documentation

## 📝 License

This project uses publicly available ECG databases. Please refer to individual dataset licenses in their respective directories:
- PTB-XL: CC0 1.0 Universal
- MIT-BIH: Open Access
- LUDB: Open Access

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Note**: Large dataset files (*.npz, *.pkl) are typically excluded from version control. See `.gitignore` for details.
