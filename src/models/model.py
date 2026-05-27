import torch
import torch.nn as nn
import timm

class DRModel(nn.Module):
    """EfficientNet-B4 backbone with 5-class DR grading head.

    Args:
        pretrained: Load ImageNet weights (True for training from scratch)
        dropout:    Dropout probability before classifier

    Example:
        model = DRModel(pretrained=True)
        dummy = torch.randn(2, 3, 512, 512)
        logits = model(dummy)   # shape: (2, 5)
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features   # 1792 for B4
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
