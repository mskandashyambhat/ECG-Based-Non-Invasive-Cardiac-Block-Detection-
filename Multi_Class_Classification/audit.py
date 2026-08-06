"""
Standalone pre-training audit entry point.
Run this before a long training job to verify the data path, model, and checkpoint flow.
"""

import sys
from pathlib import Path
import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from train import ECGTrainer
from utils import save_results

CONFIG_PATH = CURRENT_DIR / 'config.py'
CONFIG_SPEC = importlib.util.spec_from_file_location('ecg_local_config', CONFIG_PATH)
cfg = importlib.util.module_from_spec(CONFIG_SPEC)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
CONFIG_SPEC.loader.exec_module(cfg)

RESULTS_DIR = cfg.RESULTS_DIR
MAX_TRAIN_SAMPLES = cfg.MAX_TRAIN_SAMPLES
NUM_EPOCHS = cfg.NUM_EPOCHS
BATCH_SIZE = cfg.BATCH_SIZE
DEVICE = cfg.DEVICE
LEARNING_RATE = cfg.LEARNING_RATE
NUM_CLASSES = cfg.NUM_CLASSES
CLASS_NAMES = cfg.CLASS_NAMES
CLASS_WEIGHTS = cfg.CLASS_WEIGHTS


def main():
    trainer = ECGTrainer(config_dict={
        'device': DEVICE,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'class_weights': CLASS_WEIGHTS,
        'max_train_samples': MAX_TRAIN_SAMPLES,
    })

    report = trainer.run_pre_training_audit()
    save_results(report, RESULTS_DIR / 'pre_training_audit.json')
    print(f"Pre-training audit saved to {RESULTS_DIR / 'pre_training_audit.json'}")


if __name__ == '__main__':
    main()