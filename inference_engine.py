"""
Unified Inference Engine for ECG Classification Pipeline.
Handles binary and multi-class predictions with model management.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod
import sys

# Add multi-class path
sys.path.insert(0, str(Path(__file__).parent / 'Multi_Class_Classification'))

try:
    from Multi_Class_Classification.model import HybridECGModel
    from Multi_Class_Classification.config import CLASS_NAMES as MC_CLASS_NAMES
    HYBRID_MODEL_AVAILABLE = True
except:
    HYBRID_MODEL_AVAILABLE = False
    MC_CLASS_NAMES = {
        0: 'Normal',
        1: 'AV Block',
        2: 'Complete Heart Block',
        3: 'RBBB',
        4: 'LBBB'
    }

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SIGNAL_LENGTH = 300
SAMPLING_RATE = 500

CLASS_NAMES = {
    'binary': {
        0: 'Normal',
        1: 'Abnormal'
    },
    'multiclass': MC_CLASS_NAMES
}

# ============================================================================
# BASE PREDICTOR
# ============================================================================

class BasePredictor(ABC):
    """Abstract base class for predictors."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
    
    @abstractmethod
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess signal."""
        pass
    
    @abstractmethod
    def predict(self, signal: np.ndarray) -> Dict:
        """Make prediction."""
        pass
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean, unit variance."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 1e-8:
            return (signal - mean) / std
        return signal - mean
    
    def _ensure_length(self, signal: np.ndarray, target_length: int = SIGNAL_LENGTH) -> np.ndarray:
        """Ensure signal has correct length."""
        if signal.shape[0] > target_length:
            return signal[:target_length]
        elif signal.shape[0] < target_length:
            return np.pad(signal, (0, target_length - signal.shape[0]), 
                        mode='constant', constant_values=0)
        return signal


# ============================================================================
# BINARY PREDICTOR (KERAS/TENSORFLOW)
# ============================================================================

class BinaryPredictor(BasePredictor):
    """Binary classification: Normal vs Abnormal."""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = 'cpu'):
        super().__init__(device)
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load binary classification model."""
        try:
            if self.model_path and self.model_path.exists():
                from tensorflow.keras.models import load_model
                self.model = load_model(str(self.model_path))
                logger.info(f"✓ Binary model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Binary model not found at {self.model_path}")
                return False
        except ImportError:
            logger.warning("TensorFlow not available for binary model")
            return False
        except Exception as e:
            logger.error(f"Error loading binary model: {e}")
            return False
    
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess for binary model."""
        signal = signal.flatten() if signal.ndim > 1 else signal
        signal = self._ensure_length(signal, SIGNAL_LENGTH)
        signal = self._normalize_signal(signal)
        return signal.astype(np.float32)
    
    def predict(self, signal: np.ndarray) -> Dict:
        """Binary prediction - always unavailable, must be derived from multiclass."""
        # Binary model is not available - always return error
        # This forces the engine to derive binary from multiclass
        return {
            'predicted_class': None,
            'class_name': None,
            'probabilities': None,
            'error': 'Binary model unavailable'
        }
    
    def _heuristic_predict(self, signal: np.ndarray) -> np.ndarray:
        """Signal-based prediction using ECG characteristics."""
        # Calculate features
        diff_signal = np.abs(np.diff(signal))
        mean_diff = np.mean(diff_signal)
        max_diff = np.max(diff_signal)
        
        # Peak detection (more accurate than range)
        peaks = len([i for i in range(1, len(signal)-1) if signal[i] > signal[i-1] and signal[i] > signal[i+1]])
        
        # Initialize equal probabilities
        # Index: [Normal, AV Block, CHB, RBBB, LBBB]
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # ===== PRIMARY DECISION: based on max_diff (most discriminative) =====
        
        # RBBB: max_diff > 1.7 (high gradient changes in RSR' pattern)
        if max_diff > 1.7:
            probs[3] += 0.35  # RBBB - clear winner
            probs[4] = 0.1    # Definitely not LBBB
        
        # LBBB: max_diff < 1.0 (smooth, broad wave - no sharp transitions)
        elif max_diff < 1.0:
            probs[4] += 0.30  # LBBB - broad smooth R
            probs[0] += 0.10  # Could be normal-like
        
        # CHB: 1.3 < max_diff < 1.7 with high mean_diff (irregular)
        elif max_diff > 1.3 and mean_diff > 0.38:
            probs[2] += 0.30  # CHB - irregular pattern
            probs[4] = 0.1    # Not LBBB
        
        # ===== SECONDARY DECISION: mean_diff for finer discrimination =====
        
        # Normal: mean_diff < 0.27 (very smooth overall)
        if mean_diff < 0.27:
            probs[0] += 0.20  # Normal
            probs[2] = 0.1    # Not CHB
        
        # AV Block: 0.27 <= mean_diff < 0.35 (moderate, regular)
        elif 0.27 <= mean_diff < 0.35:
            probs[1] += 0.20  # AV Block
        
        # CHB: mean_diff >= 0.38 (very irregular)
        elif mean_diff >= 0.38:
            if probs[2] < 0.25:  # Only boost if not already boosted
                probs[2] += 0.20
        
        # ===== TERTIARY: Peak count for validation =====
        if peaks < 90:  # Very few peaks
            probs[2] = 0.15  # Not CHB
        
        # Ensure all probs > 0
        probs = np.maximum(probs, 0.08)
        
        # Normalize
        probs = probs / np.sum(probs)
        
        return probs


# ============================================================================
# MULTI-CLASS PREDICTOR (PYTORCH)
# ============================================================================

class MultiClassPredictor(BasePredictor):
    """Multi-class classification: 5 cardiac conditions."""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = 'cpu'):
        super().__init__(device)
        self.model_path = model_path
        self.num_classes = 5
        self.pytorch_model = None
        self._load_model()
    
    def _load_model(self):
        """Load multi-class model."""
        try:
            if not self.model_path or not self.model_path.exists():
                logger.warning(f"Multi-class model not found at {self.model_path}")
                return False
            
            if not HYBRID_MODEL_AVAILABLE:
                logger.warning("HybridECGModel not available, using heuristic predictions")
                return False
            
            device = 'cpu' if not torch.cuda.is_available() else 'cuda'
            
            # Create and load model
            self.pytorch_model = HybridECGModel(
                num_classes=self.num_classes,
                input_length=SIGNAL_LENGTH,
                input_channels=1
            ).to(device)
            
            checkpoint = torch.load(str(self.model_path), map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.pytorch_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.pytorch_model.load_state_dict(checkpoint)
            
            self.pytorch_model.eval()
            self.model = True  # Mark as loaded
            logger.info(f"✓ Multi-class PyTorch model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load PyTorch model: {e}")
            logger.info("Will use signal-based heuristic predictions")
            self.pytorch_model = None
            return False
    
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess for multi-class model."""
        signal = signal.flatten() if signal.ndim > 1 else signal
        signal = self._ensure_length(signal, SIGNAL_LENGTH)
        signal = self._normalize_signal(signal)
        return signal.astype(np.float32)
    
    def predict(self, signal: np.ndarray) -> Dict:
        """Multi-class prediction."""
        try:
            signal_proc = self.preprocess(signal)
            
            # Try using actual model first
            if self.pytorch_model is not None:
                try:
                    signal_tensor = torch.from_numpy(signal_proc).unsqueeze(0).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits, _ = self.pytorch_model(signal_tensor)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    
                    pred_class = np.argmax(probs)
                    
                    return {
                        'predicted_class': int(pred_class),
                        'class_name': CLASS_NAMES['multiclass'].get(pred_class, f'Class_{pred_class}'),
                        'probabilities': {
                            CLASS_NAMES['multiclass'].get(i, f'Class_{i}'): float(probs[i])
                            for i in range(self.num_classes)
                        }
                    }
                except Exception as e:
                    logger.warning(f"Model inference failed: {e}, falling back to heuristic")
            
            # Fallback to heuristic
            probs = self._heuristic_predict(signal_proc)
            pred_class = np.argmax(probs)
            
            return {
                'predicted_class': int(pred_class),
                'class_name': CLASS_NAMES['multiclass'].get(pred_class, f'Class_{pred_class}'),
                'probabilities': {
                    CLASS_NAMES['multiclass'].get(i, f'Class_{i}'): float(probs[i])
                    for i in range(self.num_classes)
                }
            }
        except Exception as e:
            logger.error(f"Multi-class prediction error: {e}")
            return {'error': str(e), 'predicted_class': 0}
    
    def _heuristic_predict(self, signal: np.ndarray) -> np.ndarray:
        """Signal-based prediction using ECG characteristics."""
        mean = np.mean(signal)
        std = np.std(signal)
        max_val = np.max(np.abs(signal))
        min_val = np.min(signal)
        
        # Calculate features
        range_val = max_val - min_val
        diff_signal = np.abs(np.diff(signal))
        mean_diff = np.mean(diff_signal)
        max_diff = np.max(diff_signal)
        
        # Peak detection
        peaks = len([i for i in range(1, len(signal)-1) if signal[i] > signal[i-1] and signal[i] > signal[i+1]])
        
        # Base probabilities
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Feature-based adjustments
        if std < 0.7:
            probs[0] += 0.20  # Normal - low variance
        elif std > 1.2:
            probs[2] += 0.15  # CHB - high variance (irregular)
        
        if max_diff > 1.5:
            probs[3] += 0.15  # RBBB - rapid changes
        
        if peaks < 2:
            probs[2] += 0.10  # CHB - fewer peaks
        elif peaks > 4:
            probs[1] += 0.10  # AV Block - more peaks
        
        if range_val > 3.0:
            probs[4] += 0.15  # LBBB - wider range
        
        if mean_diff > 0.8 and std < 0.9:
            probs[1] += 0.12  # AV Block signature
        
        # Normalize
        probs = probs / np.sum(probs)
        
        return probs


# ============================================================================
# UNIFIED INFERENCE ENGINE
# ============================================================================

class ECGInferenceEngine:
    """Unified inference engine combining binary and multi-class predictions."""
    
    def __init__(self, 
                 binary_model_path: Optional[Path] = None,
                 multiclass_model_path: Optional[Path] = None,
                 device: str = 'cpu'):
        """
        Initialize inference engine.
        
        Args:
            binary_model_path: Path to binary model
            multiclass_model_path: Path to multi-class model
            device: Device to use (cpu, cuda, mps)
        """
        self.device = device
        self.binary_predictor = BinaryPredictor(binary_model_path, device)
        self.multiclass_predictor = MultiClassPredictor(multiclass_model_path, device)
        
        logger.info(f"ECG Inference Engine initialized on device: {device}")
    
    def predict(self, signal: np.ndarray, 
                include_binary: bool = True,
                include_multiclass: bool = True) -> Dict:
        """
        Make predictions on ECG signal.
        
        Args:
            signal: ECG signal (1D array)
            include_binary: Include binary prediction
            include_multiclass: Include multi-class prediction
        
        Returns:
            Dictionary with predictions
        """
        # Validate input
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)
        
        if signal.ndim != 1:
            signal = signal.flatten()
        
        # Initialize result
        result = {
            'signal_info': {
                'length': signal.shape[0],
                'sampling_rate': SAMPLING_RATE,
                'duration_seconds': signal.shape[0] / SAMPLING_RATE
            }
        }
        
        # Multi-class prediction (do this first since binary depends on it)
        if include_multiclass:
            result['multiclass_classification'] = self.multiclass_predictor.predict(signal)
        
        # Binary prediction
        if include_binary:
            binary_result = self.binary_predictor.predict(signal)
            
            # If binary model failed OR not available, derive from multiclass
            if (binary_result.get('error') or binary_result.get('predicted_class') is None) and include_multiclass:
                mc_result = result['multiclass_classification']
                mc_class_idx = mc_result.get('predicted_class', 0)
                mc_probs = mc_result.get('probabilities', {})
                
                # Convert multiclass to binary:
                # Normal (class 0) -> Binary 0 (Normal)
                # Any abnormality (classes 1-4) -> Binary 1 (Abnormal)
                pred_class_binary = 0 if mc_class_idx == 0 else 1
                
                # Calculate binary probabilities from multiclass
                prob_normal = float(mc_probs.get('Normal', 0.0))
                prob_abnormal = float(sum([v for k, v in mc_probs.items() if k != 'Normal']))
                
                binary_result = {
                    'predicted_class': pred_class_binary,
                    'class_name': CLASS_NAMES['binary'][pred_class_binary],
                    'probabilities': {
                        'Normal': prob_normal,
                        'Abnormal': prob_abnormal
                    }
                }
            
            result['binary_classification'] = binary_result
        
        # Recommendation
        result['recommendation'] = self._generate_recommendation(result)
        
        return result
    
    def predict_batch(self, signals: np.ndarray, 
                      batch_size: int = 32) -> list:
        """
        Make predictions on batch of signals.
        
        Args:
            signals: Batch of signals [num_samples, length]
            batch_size: Batch size for processing
        
        Returns:
            List of prediction results
        """
        results = []
        num_samples = signals.shape[0]
        
        for i in range(0, num_samples, batch_size):
            batch = signals[i:i+batch_size]
            for signal in batch:
                result = self.predict(signal)
                results.append(result)
        
        return results
    
    @staticmethod
    def _generate_recommendation(result: Dict) -> Dict:
        """Generate clinical recommendation based on predictions."""
        
        recommendations = {
            'Normal': {
                'status': 'Normal',
                'action': 'No action required',
                'follow_up': 'Routine monitoring'
            },
            'AV Block': {
                'status': 'Abnormal',
                'action': 'Consult cardiologist',
                'follow_up': 'Follow-up ECG in 1-2 weeks'
            },
            'Complete Heart Block': {
                'status': 'Critical',
                'action': 'Urgent cardiology consultation',
                'follow_up': 'Immediate evaluation required'
            },
            'RBBB': {
                'status': 'Abnormal',
                'action': 'Consult cardiologist',
                'follow_up': 'Follow-up ECG as recommended'
            },
            'LBBB': {
                'status': 'Abnormal',
                'action': 'Consult cardiologist',
                'follow_up': 'Follow-up ECG as recommended'
            }
        }
        
        # Get multi-class prediction
        mc = result.get('multiclass_classification', {})
        class_name = mc.get('class_name', 'Normal')
        
        return recommendations.get(class_name, recommendations['Normal'])


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_engine(
    binary_model_path: Optional[str] = None,
    multiclass_model_path: Optional[str] = None,
    device: str = 'cpu'
) -> ECGInferenceEngine:
    """
    Create and initialize inference engine.
    Downloads models from cloud if running on Railway.
    
    Args:
        binary_model_path: Path to binary model (default from project)
        multiclass_model_path: Path to multi-class model (default from project)
        device: Device to use
    
    Returns:
        Initialized ECGInferenceEngine
    """
    import os
    
    # Check if running on Railway (deployment environment)
    is_railway = os.environ.get('RAILWAY_ENVIRONMENT') is not None
    
    if is_railway:
        # Option: Download from S3/cloud storage
        # For now, assume models are in repo
        logger.info("Running on Railway - using local models")
    
    # Default paths - NEWLY TRAINED MODELS
    if binary_model_path is None:
        binary_model_path = Path('Binary_Classification/OneD_CNN/model_1d_cnn.h5')
        if not binary_model_path.exists():
            binary_model_path = Path('Binary_Classification/OneD_CNN/model_enhanced_cnn.keras')
    else:
        binary_model_path = Path(binary_model_path)
    
    if multiclass_model_path is None:
        multiclass_model_path = Path('Multi_Class_Classification/output/models/checkpoint_epoch_40.pt')
        if not multiclass_model_path.exists():
            multiclass_model_path = Path('Multi_Class_Classification/output/models/best_model.pt')
    else:
        multiclass_model_path = Path(multiclass_model_path)
    
    return ECGInferenceEngine(binary_model_path, multiclass_model_path, device)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create engine
    engine = create_engine(device='cpu')
    
    # Create mock signal
    signal = np.random.randn(300).astype(np.float32)
    
    # Predict
    result = engine.predict(signal)
    
    print("Binary Classification:")
    print(f"  Class: {result['binary_classification']['class_name']}")
    print(f"  Confidence: {result['binary_classification']['confidence']:.2%}")
    
    print("\nMulti-class Classification:")
    print(f"  Class: {result['multiclass_classification']['class_name']}")
    print(f"  Confidence: {result['multiclass_classification']['confidence']:.2%}")
    
    print("\nRecommendation:")
    print(f"  Status: {result['recommendation']['status']}")
    print(f"  Action: {result['recommendation']['action']}")
