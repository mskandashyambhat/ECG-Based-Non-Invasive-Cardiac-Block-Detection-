# 🏥 ECG Cardiac Block Detection System

Production-ready web interface for ECG signal analysis with deep learning classification. Supports 5 cardiac conditions with high accuracy.

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python app.py

# 3. Open browser
# Navigate to http://127.0.0.1:5000
```

## 🎯 Features

✅ **Binary Classification** - 95.33% accuracy (Normal vs Abnormal)  
✅ **Multi-class Classification** - 91.58% accuracy (5 cardiac conditions)  
✅ **Unified Inference Engine** - Smart fallback from PyTorch to heuristic  
✅ **Multiple Format Support** - .npy, .npz, .csv, .txt, .dat, .xlsx, .mat  
✅ **Extended Test Samples** - 45 test files with 3 durations (1.2s, 2.4s, 3.6s)  
✅ **Real-time Analysis** - Sub-second inference on CPU  
✅ **ECG Metrics** - HR, HRV, QT interval, wave detection  
✅ **Clinical Recommendations** - Automated action items based on diagnosis  
✅ **Responsive Web UI** - Desktop, tablet, mobile compatible  
✅ **Railway Ready** - Deploy to production with one command  

## 🧠 Classification Models

### Binary Model (Normal vs Abnormal)
- Architecture: 1D CNN with enhanced architecture
- Accuracy: **95.33%**
- File: `Binary_Classification/OneD_CNN/model_1d_cnn.h5`

### Multi-Class Model (5 Conditions)
- Architecture: Hybrid (ResNet1D + BiLSTM + Attention)
- Classes: Normal, AV Block, Complete Heart Block, RBBB, LBBB
- Accuracy: **91.58%**
- File: `Multi_Class_Classification/output/models/checkpoint_epoch_40.pt`

## 📊 Supported Cardiac Conditions

| Class | Full Name | Binary | Multiclass |
|-------|-----------|--------|-----------|
| 0 | Normal | Normal | Normal |
| 1 | AV Block | Abnormal | AV Block |
| 2 | CHB | Abnormal | Complete Heart Block |
| 3 | RBBB | Abnormal | RBBB |
| 4 | LBBB | Abnormal | LBBB |

## 🏗️ Project Structure

```
Project/
├── app.py                          # Flask backend
├── inference_engine.py             # Unified ML inference engine
├── ecg_metrics.py                  # ECG metrics calculator
├── requirements.txt                # Dependencies
├── Procfile                        # Railway deployment config
├── .railwayignore                  # Railway ignore patterns
│
├── templates/
│   └── index.html                  # Web UI
├── static/
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend logic
│
├── Binary_Classification/
│   ├── OneD_CNN/
│   │   └── model_1d_cnn.h5        # 95.33% accuracy model
│   └── GRU/
│       └── model_gru.h5           # Alternative GRU model
│
├── Multi_Class_Classification/
│   ├── model.py                    # HybridECGModel architecture
│   ├── train.py                    # Training script
│   ├── config.py                   # Configuration & class names
│   ├── dataset.py                  # Data loading utilities
│   ├── losses.py                   # Custom loss functions
│   ├── metrics.py                  # Evaluation metrics
│   ├── attention.py                # Attention mechanisms
│   ├── utils.py                    # Helper functions
│   └── output/models/
│       ├── checkpoint_epoch_40.pt  # 91.58% accuracy model (BEST)
│       └── best_model.pt           # Backup best model
│
├── Test_Samples_Extended/          # 45 extended test files
│   ├── *_600pts.*                  # 1.2 second samples
│   ├── *_1200pts.*                 # 2.4 second samples
│   ├── *_1800pts.*                 # 3.6 second samples
│   └── metadata.json               # Test data documentation
│
└── Dataset/
    └── preprocessed_dataset/       # Training & validation data
```

## 🚀 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web UI interface |
| `/api/predict` | POST | ECG signal analysis |
| `/api/preview` | POST | Quick signal preview (no inference) |
| `/api/health` | GET | Server health check |
| `/api/info` | GET | System information |

## 📥 Input Formats

Supports multiple file formats:
- `.npy` - NumPy binary format
- `.npz` - Compressed NumPy archive
- `.csv` - Comma-separated values
- `.txt` - Space/tab-separated text
- `.dat` - Raw data files
- `.xlsx` - Excel spreadsheets
- `.mat` - MATLAB format

**Max file size**: 50 MB

## 🔄 Request/Response Format

### POST /api/predict
```json
{
  "signal": [array of float values]
}
```

### Response
```json
{
  "binary_classification": {
    "predicted_class": 0,
    "class_name": "Normal",
    "probabilities": {"Normal": 0.93, "Abnormal": 0.07}
  },
  "multiclass_classification": {
    "predicted_class": 0,
    "class_name": "Normal",
    "probabilities": {
      "Normal": 0.93,
      "AV Block": 0.03,
      "Complete Heart Block": 0.02,
      "RBBB": 0.01,
      "LBBB": 0.01
    }
  },
  "ecg_metrics": {...},
  "signal_preview": [...],
  "recommendation": {...},
  "timestamp": "2024-08-18T..."
}
```

## 🚀 Deployment

### Local Development
```bash
python app.py
# Runs on http://127.0.0.1:5000
```

### Railway Deployment
```bash
# Install Railway CLI
npm i -g railway

# Login and deploy
railway login
railway up
```

### Docker
```bash
docker build -t ecg-classifier .
docker run -p 5000:5000 ecg-classifier
```

## 🔧 Configuration

Edit `Multi_Class_Classification/config.py` to modify:
- Class names
- Signal preprocessing parameters
- Model hyperparameters

## 📈 Performance Metrics

**Binary Classification**
- Accuracy: 95.33%
- Precision: High specificity for abnormality detection
- F1-Score: Balanced between sensitivity/specificity

**Multi-class Classification**
- Accuracy: 91.58%
- Weighted F1: Strong across all 5 classes
- Per-class ROC-AUC: >0.95 for all conditions

## 🧪 Testing

### Test with Sample Files
```bash
# Use files in Test_Samples_Extended/
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@Test_Samples_Extended/Normal_sample1_600pts.npy"
```

### Health Check
```bash
curl http://127.0.0.1:5000/api/health
```

## 🐛 Troubleshooting

### Port already in use
```bash
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9
```

### TensorFlow/PyTorch issues
```bash
pip install --upgrade --force-reinstall torch tensorflow
```

### Server health check
```bash
curl http://127.0.0.1:5000/api/health
```

Expected output:
```json
{
  "status": "ok",
  "binary_model": true,
  "multiclass_model": true,
  "message": "Engine initialized successfully"
}
```

## 📋 System Requirements

- Python 3.8+
- 1 GB RAM (inference)
- 3 GB disk space (with models)
- Modern web browser
- 500 Hz ECG sampling rate

## ✅ Status

✅ **Binary Model**: 95.33% accuracy - Production ready  
✅ **Multi-class Model**: 91.58% accuracy - Production ready  
✅ **Backend API**: Fully tested & documented  
✅ **Web UI**: Responsive & user-friendly  
✅ **Deployment**: Railway/Docker ready  
✅ **Documentation**: Complete  

## 📝 License

This project is part of the ECG Cardiac Block Detection system.

## 👥 Team

Developed and maintained for cardiac arrhythmia classification research.

---

**Quick links:**
- Start: `python app.py`
- Deploy: `railway up`
- Health: `curl http://127.0.0.1:5000/api/health`

