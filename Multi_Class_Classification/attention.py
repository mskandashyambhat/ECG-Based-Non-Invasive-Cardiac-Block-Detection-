"""
Attention mechanisms for ECG signal processing.
Includes Multi-Head Self-Attention and specialized attention for time-series data.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for learning important ECG regions.
    Allows the model to focus on different temporal patterns simultaneously.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, query_len, hidden_dim]
            key: [batch_size, key_len, hidden_dim]
            value: [batch_size, value_len, hidden_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch_size, query_len, hidden_dim]
            attention_weights: [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.query(query)  # [batch_size, query_len, hidden_dim]
        K = self.key(key)      # [batch_size, key_len, hidden_dim]
        V = self.value(value)  # [batch_size, value_len, hidden_dim]
        
        # Split into multiple heads
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, num_heads, query_len, key_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        # [batch_size, num_heads, query_len, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.hidden_dim)
        
        # Final linear projection
        out = self.fc_out(out)
        
        return out, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention where Q=K=V (self-attention).
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Self-attention: Q=K=V=x
        attn_out, attention_weights = self.attention(x, x, x, mask)
        
        # Residual connection and normalization
        out = self.norm(x + self.dropout(attn_out))
        
        return out, attention_weights


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism specialized for ECG time-series.
    Learns which time points are most important for classification.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of input features
            dropout: Dropout rate
        """
        super(TemporalAttention, self).__init__()
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        
        Returns:
            weighted_output: [batch_size, hidden_dim]
            attention_weights: [batch_size, seq_len]
        """
        # Compute attention scores
        scores = self.fc1(x)  # [batch_size, seq_len, hidden_dim]
        scores = torch.relu(scores)
        scores = self.dropout(scores)
        scores = self.fc2(scores)  # [batch_size, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # Apply attention to input
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        weighted_output = x * attention_weights_expanded  # Element-wise multiplication
        weighted_output = weighted_output.sum(dim=1)  # [batch_size, hidden_dim]
        
        return weighted_output, attention_weights


class ConvolutionalAttention(nn.Module):
    """
    Convolutional Attention mechanism for capturing local patterns in ECG.
    Uses depthwise separable convolutions for efficiency.
    """
    
    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of input features
            kernel_size: Kernel size for convolution
            dropout: Dropout rate
        """
        super(ConvolutionalAttention, self).__init__()
        
        padding = kernel_size // 2
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        # Store for residual connection
        residual = x
        
        # Transpose for conv1d: [batch_size, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.depthwise(x)
        x = torch.relu(x)
        x = self.pointwise(x)
        x = torch.sigmoid(x)  # Attention gate
        
        # Transpose back: [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        # Apply attention and residual connection
        out = residual * x
        out = self.norm(out + self.dropout(residual))
        
        return out


class AttentionBlock(nn.Module):
    """
    Complete attention block combining multi-head self-attention with feed-forward network.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1, ff_dim: int = None):
        """
        Args:
            hidden_dim: Dimension of features
            num_heads: Number of attention heads
            dropout: Dropout rate
            ff_dim: Feed-forward dimension (default: 4 * hidden_dim)
        """
        super(AttentionBlock, self).__init__()
        
        if ff_dim is None:
            ff_dim = 4 * hidden_dim
        
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Self-attention
        attn_out, attention_weights = self.self_attn(x, mask)
        
        # Feed-forward
        ff_out = self.ff(attn_out)
        out = self.norm(attn_out + ff_out)
        
        return out, attention_weights


class VisualizableAttention(nn.Module):
    """
    Attention module that saves attention weights for visualization.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(VisualizableAttention, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.last_attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        out, attention_weights = self.attention(x)
        self.last_attention_weights = attention_weights.detach()
        return out
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights for visualization."""
        return self.last_attention_weights
