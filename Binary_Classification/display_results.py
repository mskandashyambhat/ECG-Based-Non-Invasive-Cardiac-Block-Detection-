import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

base_dir = "/Users/skandashyam/Desktop/MajorProject/Project/Binary_Classification"
models = ['OneD_CNN', 'GRU']

print("\n" + "="*80)
print("ECG BINARY CLASSIFICATION - RESULTS SUMMARY")
print("="*80)

for model_name in models:
    model_dir = os.path.join(base_dir, model_name)
    
    if not os.path.exists(os.path.join(model_dir, "metrics.json")):
        print(f"\n⏳ {model_name} model results not ready yet...")
        continue
    
    print(f"\n{'-'*80}")
    print(f"{model_name.upper()} MODEL RESULTS")
    print(f"{'-'*80}")
    
    # Load metrics
    with open(os.path.join(model_dir, "metrics.json"), 'r') as f:
        metrics = json.load(f)
    
    print(f"\nTest Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    print(f"\nClassification Report:")
    cr = metrics['classification_report']
    for label in ['0', '1']:
        if label in cr:
            print(f"  Class {label} ({['No Block', 'Block Present'][int(label)]})")
            print(f"    Precision: {cr[label]['precision']:.4f}")
            print(f"    Recall: {cr[label]['recall']:.4f}")
            print(f"    F1-Score: {cr[label]['f1-score']:.4f}")

print("\n" + "="*80)
print("Results saved in:")
print(f"  - {os.path.join(base_dir, 'OneD_CNN')}/")
print(f"  - {os.path.join(base_dir, 'GRU')}/")
print("="*80 + "\n")
