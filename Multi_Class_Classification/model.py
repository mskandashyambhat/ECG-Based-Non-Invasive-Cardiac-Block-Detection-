"""
State-of-the-art hybrid deep learning models for ECG-based block detection.
Includes ResNet1D backbone, BiLSTM, and Multi-Head Attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from attention import MultiHeadSelfAttention, TemporalAttention, AttentionBlock


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block for ECG signal processing.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7,
                 stride: int = 1, padding: int = 3, dropout: float = 0.3):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolution
            stride: Stride for convolution
            padding: Padding for convolution
            dropout: Dropout rate
        """
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, length]
        
        Returns:
            output: [batch_size, out_channels, length]
        """
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class ResNet1DBackbone(nn.Module):
    """
    ResNet1D backbone for ECG signal feature extraction.
    """
    
    def __init__(self, in_channels: int = 1, layers: List[int] = [2, 2, 2, 2],
                 channels: List[int] = [64, 128, 256, 512], kernel_size: int = 7, dropout: float = 0.3):
        """
        Args:
            in_channels: Number of input channels
            layers: Number of residual blocks in each stage
            channels: Number of channels in each stage
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super(ResNet1DBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.padding = kernel_size // 2
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size=kernel_size,
                              stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0], stride=1,
                                       kernel_size=kernel_size, dropout=dropout)
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2,
                                       kernel_size=kernel_size, dropout=dropout)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2,
                                       kernel_size=kernel_size, dropout=dropout)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2,
                                       kernel_size=kernel_size, dropout=dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                   stride: int = 1, kernel_size: int = 7, dropout: float = 0.3) -> nn.Sequential:
        """Create residual layer."""
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, kernel_size, stride,
                                     kernel_size // 2, dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size, 1,
                                         kernel_size // 2, dropout))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, length]
        
        Returns:
            features: [batch_size, 512, length/8]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class HybridECGModel(nn.Module):
    """
    State-of-the-art hybrid model for ECG block detection and classification.
    
    Architecture:
    - ResNet1D Feature Extractor
    - BiLSTM for temporal dependencies
    - Multi-Head Attention
    - Fully Connected Classifier
    """
    
    def __init__(self, num_classes: int = 5, input_length: int = 300, input_channels: int = 1,
                 resnet_channels: List[int] = [64, 128, 256, 512],
                 lstm_hidden_dim: int = 256, lstm_num_layers: int = 2,
                 num_attention_heads: int = 8, dropout: float = 0.3):
        """
        Args:
            num_classes: Number of output classes
            input_length: Length of input ECG signal
            input_channels: Number of input channels (1 for single lead)
            resnet_channels: Channels for ResNet stages
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_num_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(HybridECGModel, self).__init__()
        
        self.num_classes = num_classes
        self.input_length = input_length
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # ===== Feature Extraction: ResNet1D =====
        self.resnet_backbone = ResNet1DBackbone(
            in_channels=input_channels,
            layers=[2, 2, 2, 2],
            channels=resnet_channels,
            dropout=dropout
        )
        
        # Compute the output length after ResNet (after 3 maxpools)
        self.resnet_output_length = input_length // 8
        self.resnet_output_channels = resnet_channels[-1]
        
        # ===== Temporal Processing: BiLSTM =====
        self.bilstm = nn.LSTM(
            input_size=self.resnet_output_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # After BiLSTM, we have bidirectional output
        self.lstm_output_dim = lstm_hidden_dim * 2
        
        # ===== Attention Mechanism =====
        self.attention = MultiHeadSelfAttention(
            hidden_dim=self.lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # ===== Global Pooling =====
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # ===== Classification Head =====
        # Dense layers
        self.fc1 = nn.Linear(self.lstm_output_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout * 0.8)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout * 0.6)
        
        # Output layer
        self.classifier = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for dense layers."""
        for m in [self.fc1, self.fc2, self.fc3, self.classifier]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input ECG signal [batch_size, input_length] or [batch_size, 1, input_length]
            return_attention: Whether to return attention weights
        
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, num_heads, seq_len, seq_len] if return_attention=True
        """
        batch_size = x.size(0)
        
        # Reshape if needed: [batch_size, input_length] -> [batch_size, 1, input_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # ===== ResNet1D Feature Extraction =====
        # [batch_size, 1, 300] -> [batch_size, 512, 37]
        resnet_features = self.resnet_backbone(x)
        
        # Transpose for LSTM: [batch_size, 512, 37] -> [batch_size, 37, 512]
        lstm_input = resnet_features.transpose(1, 2)
        
        # ===== BiLSTM =====
        # [batch_size, 37, 512] -> [batch_size, 37, 512]
        lstm_out, (h_n, c_n) = self.bilstm(lstm_input)
        
        # ===== Multi-Head Attention =====
        # [batch_size, 37, 512] -> [batch_size, 37, 512]
        attn_out, attention_weights = self.attention(lstm_out)
        
        # ===== Global Average Pooling =====
        # Transpose for pooling: [batch_size, 37, 512] -> [batch_size, 512, 37]
        pooled = attn_out.transpose(1, 2)
        pooled = self.global_avg_pool(pooled)  # [batch_size, 512, 1]
        pooled = pooled.view(batch_size, -1)    # [batch_size, 512]
        
        # ===== Classification Head =====
        x = self.fc1(pooled)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        logits = self.classifier(x)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits, None
    
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization and analysis.
        
        Args:
            x: Input ECG signal
        
        Returns:
            Dictionary containing features from each stage
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = {}
        
        # ResNet features
        resnet_out = self.resnet_backbone(x)
        features['resnet'] = resnet_out.detach()
        
        # BiLSTM features
        lstm_input = resnet_out.transpose(1, 2)
        lstm_out, _ = self.bilstm(lstm_input)
        features['bilstm'] = lstm_out.detach()
        
        # Attention features
        attn_out, attn_weights = self.attention(lstm_out)
        features['attention'] = attn_out.detach()
        features['attention_weights'] = attn_weights.detach()
        
        return features


def create_model(num_classes: int = 5, pretrained: bool = False, **kwargs) -> HybridECGModel:
    """
    Factory function to create the hybrid ECG model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments for model
    
    Returns:
        HybridECGModel instance
    """
    model = HybridECGModel(
        num_classes=num_classes,
        input_length=kwargs.get('input_length', 300),
        input_channels=kwargs.get('input_channels', 1),
        resnet_channels=kwargs.get('resnet_channels', [64, 128, 256, 512]),
        lstm_hidden_dim=kwargs.get('lstm_hidden_dim', 256),
        lstm_num_layers=kwargs.get('lstm_num_layers', 2),
        num_attention_heads=kwargs.get('num_attention_heads', 8),
        dropout=kwargs.get('dropout', 0.3)
    )
    
    if pretrained:
        # Load pretrained weights if available
        pass
    
    return model


# Model variants
def hybrid_ecg_small(num_classes: int = 5) -> HybridECGModel:
    """Small model variant."""
    return create_model(
        num_classes=num_classes,
        resnet_channels=[32, 64, 128, 256],
        lstm_hidden_dim=128,
        num_attention_heads=4,
        dropout=0.2
    )


def hybrid_ecg_base(num_classes: int = 5) -> HybridECGModel:
    """Base model variant (default)."""
    return create_model(
        num_classes=num_classes,
        resnet_channels=[64, 128, 256, 512],
        lstm_hidden_dim=256,
        num_attention_heads=8,
        dropout=0.3
    )


def hybrid_ecg_large(num_classes: int = 5) -> HybridECGModel:
    """Large model variant."""
    return create_model(
        num_classes=num_classes,
        resnet_channels=[128, 256, 512, 1024],
        lstm_hidden_dim=512,
        num_attention_heads=16,
        dropout=0.4
    )
