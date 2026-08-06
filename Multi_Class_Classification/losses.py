"""
Custom loss functions for ECG classification.
Includes standard cross-entropy and focal loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy/hard examples
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with class weights.
    Handles class imbalance by weighting each class differently.
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        """
        Args:
            weights: Class weights [num_classes]. If None, uniform weights.
            label_smoothing: Label smoothing factor [0, 1]
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Weighted cross-entropy loss
        """
        return F.cross_entropy(
            inputs, targets,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    Prevents overconfident predictions by smoothing label distributions.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: Number of classes
            smoothing: Label smoothing factor [0, 1]
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        
        Returns:
            Label smoothed cross-entropy loss
        """
        pred = self.log_softmax(pred)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=1))


class CombinedLoss(nn.Module):
    """
    Combined loss function: weighted cross-entropy + focal loss.
    Balances handling of both easy and hard examples with class imbalance.
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None, 
                 alpha: float = 0.25, gamma: float = 2.0, 
                 lambda_ce: float = 0.5, lambda_focal: float = 0.5):
        """
        Args:
            weights: Class weights for cross-entropy
            alpha: Focal loss alpha
            gamma: Focal loss gamma
            lambda_ce: Weight for cross-entropy component
            lambda_focal: Weight for focal loss component
        """
        super(CombinedLoss, self).__init__()
        self.ce_loss = WeightedCrossEntropyLoss(weights=weights)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_ce = lambda_ce
        self.lambda_focal = lambda_focal
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combine both losses."""
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.lambda_ce * ce + self.lambda_focal * focal


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class classification.
    Particularly useful for imbalanced datasets.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
            reduction: 'mean' or 'sum'
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Dice loss
        """
        # Convert to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Compute Dice loss
        intersection = (probs * targets_one_hot).sum(dim=0)
        cardinality = (probs.sum(dim=0) + targets_one_hot.sum(dim=0))
        dice_loss = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


def get_loss_function(loss_type: str, num_classes: int = 5, class_weights: Optional[dict] = None,
                     **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing', 'combined', 'dice')
        num_classes: Number of classes
        class_weights: Dictionary mapping class index to weight
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function module
    """
    
    # Convert class weights dict to tensor if provided
    weights_tensor = None
    if class_weights is not None:
        weights_list = [class_weights.get(i, 1.0) for i in range(num_classes)]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
    
    if loss_type == 'cross_entropy':
        label_smoothing = kwargs.get('label_smoothing', 0.0)
        return WeightedCrossEntropyLoss(weights=weights_tensor, label_smoothing=label_smoothing)
    
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropyLoss(num_classes=num_classes, smoothing=smoothing)
    
    elif loss_type == 'combined':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        lambda_ce = kwargs.get('lambda_ce', 0.5)
        lambda_focal = kwargs.get('lambda_focal', 0.5)
        return CombinedLoss(weights=weights_tensor, alpha=alpha, gamma=gamma, 
                           lambda_ce=lambda_ce, lambda_focal=lambda_focal)
    
    elif loss_type == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
