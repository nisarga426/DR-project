import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced DR grading.

    Args:
        gamma: Focusing parameter (2.0 is standard)
        weight: Per-class weights tensor (optional)

    Example:
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(logits, labels)
    """
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
