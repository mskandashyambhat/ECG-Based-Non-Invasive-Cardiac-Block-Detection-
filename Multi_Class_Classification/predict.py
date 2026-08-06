"""
Inference script for ECG block detection.
Loads trained model and makes predictions on new ECG signals.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from config import *
from model import HybridECGModel
from utils import get_device, setup_logger, load_checkpoint

logger = setup_logger('predict', LOGS_DIR)


class ECGInferencer:
    """Inference engine for ECG predictions."""
    
    def __init__(self, model_path: Path, device: str = 'cuda'):
        """
        Initialize inferencer.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use for inference
        """
        self.device = get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = HybridECGModel(
            num_classes=NUM_CLASSES,
            input_length=SIGNAL_LENGTH,
            input_channels=NUM_LEADS,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_num_layers=LSTM_NUM_LAYERS,
            num_attention_heads=NUM_ATTENTION_HEADS,
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Class names
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal.
        
        Args:
            signal: Raw ECG signal [signal_length]
        
        Returns:
            Preprocessed signal [signal_length]
        """
        # Handle 2D input
        if signal.ndim == 2:
            signal = signal[:, 0]
        
        # Trim or pad to correct length
        if signal.shape[0] > SIGNAL_LENGTH:
            signal = signal[:SIGNAL_LENGTH]
        elif signal.shape[0] < SIGNAL_LENGTH:
            signal = np.pad(signal, (0, SIGNAL_LENGTH - signal.shape[0]), mode='constant', value=0)
        
        # Normalize
        mean = np.mean(signal)
        std = np.std(signal)
        signal = (signal - mean) / (std + 1e-8)
        
        return signal
    
    def predict(self, signal: np.ndarray, return_attention: bool = False) -> Dict:
        """
        Make prediction on ECG signal.
        
        Args:
            signal: ECG signal [signal_length] or [signal_length, 1]
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary containing predictions and confidence
        """
        # Preprocess
        signal = self.preprocess_signal(signal)
        
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits, attention_weights = self.model(signal_tensor, return_attention=return_attention)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction
        pred_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
        class_name = self.class_names.get(pred_class, f"Class_{pred_class}")
        
        # Prepare result
        result = {
            'predicted_class': pred_class,
            'predicted_class_name': class_name,
            'confidence': confidence,
            'probabilities': {
                self.class_names.get(i, f"Class_{i}"): float(probabilities[0, i].item())
                for i in range(self.num_classes)
            }
        }
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights[0].cpu().numpy()
        
        return result
    
    def predict_batch(self, signals: np.ndarray, batch_size: int = 32) -> list:
        """
        Make predictions on batch of signals.
        
        Args:
            signals: Batch of signals [num_samples, signal_length] or [num_samples, signal_length, 1]
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        num_samples = signals.shape[0]
        results = []
        
        for i in range(0, num_samples, batch_size):
            batch_signals = signals[i:i+batch_size]
            
            # Preprocess batch
            batch_processed = np.array([self.preprocess_signal(sig) for sig in batch_signals])
            
            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_processed).float().to(self.device)
            
            # Forward pass
            with torch.no_grad():
                logits, _ = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
            
            # Process results
            for j in range(len(batch_signals)):
                pred_class = logits[j].argmax(dim=0).item()
                confidence = probabilities[j, pred_class].item()
                class_name = self.class_names.get(pred_class, f"Class_{pred_class}")
                
                result = {
                    'predicted_class': pred_class,
                    'predicted_class_name': class_name,
                    'confidence': confidence,
                    'probabilities': {
                        self.class_names.get(k, f"Class_{k}"): float(probabilities[j, k].item())
                        for k in range(self.num_classes)
                    }
                }
                results.append(result)
        
        return results
    
    def get_intermediate_features(self, signal: np.ndarray) -> Dict:
        """
        Get intermediate feature maps for explainability.
        
        Args:
            signal: ECG signal
        
        Returns:
            Dictionary with intermediate features
        """
        # Preprocess
        signal = self.preprocess_signal(signal)
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(self.device)
        
        # Get features
        with torch.no_grad():
            features = self.model.get_intermediate_features(signal_tensor)
        
        result = {
            'resnet_features': features['resnet'].cpu().numpy(),
            'bilstm_features': features['bilstm'].cpu().numpy(),
            'attention_features': features['attention'].cpu().numpy(),
            'attention_weights': features['attention_weights'].cpu().numpy()
        }
        
        return result


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG inference')
    parser.add_argument('--model', type=str, default='./output/models/best_model.pt',
                       help='Path to saved model')
    parser.add_argument('--signal', type=str, help='Path to ECG signal file (NPZ or NPY)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'mps'],
                       help='Device to use')
    args = parser.parse_args()
    
    # Create inferencer
    inferencer = ECGInferencer(Path(args.model), device=args.device)
    
    if args.signal:
        # Load signal
        if args.signal.endswith('.npz'):
            data = np.load(args.signal)
            signal = data['X'] if 'X' in data else list(data.values())[0]
        else:
            signal = np.load(args.signal)
        
        # Handle batch or single signal
        if signal.ndim == 2:
            print("Making batch predictions...")
            results = inferencer.predict_batch(signal)
            for i, result in enumerate(results):
                print(f"\nSample {i+1}:")
                print(f"  Predicted Class: {result['predicted_class_name']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Probabilities: {result['probabilities']}")
        else:
            print("Making single prediction...")
            result = inferencer.predict(signal, return_attention=True)
            print(f"\nPredicted Class: {result['predicted_class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
    else:
        print("Please provide a signal file using --signal argument")


if __name__ == '__main__':
    main()
