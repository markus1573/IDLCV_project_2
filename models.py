import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Optional
import torchvision.models as tv_models


class single_frame_model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # resnet18
        self.model = tv_models.resnet18(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4, "x must be a 4D tensor: [batch, channels, height, width]"
        return self.model(x)

class early_fusion_model(nn.Module):
    def __init__(self, num_frames=10, num_classes=10):
        super().__init__() 
        
        # Create ResNet18
        self.model = tv_models.resnet18(num_classes=num_classes)
        
        # Replace first conv to accept C*T channels 
        self.model.conv1 = nn.Conv2d(
            in_channels=3 * num_frames,
            out_channels=64,             #
            kernel_size=7,               # 
            stride=2,                    # Copied from resnet18.
            padding=3,                   #
            bias=False                   #
        )
        
    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 5, "x must be a 5D tensor: [batch, channels, num_frames, height, width]"
        
        B, C, T, H, W = x.shape
        x = x.reshape(B, C*T, H, W)  # [batch, channels*num_frames, height, width]
        return self.model(x)

        
class late_fusion_model(nn.Module):
    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()
        self.model = tv_models.resnet18(num_classes=num_classes)
        
        # Get feature dimension before removing fc
        feature_dim = self.model.fc.in_features  # 512 for ResNet18
        
        # Remove fc layer - use Identity instead of None
        self.model.fc = nn.Identity()
        
        # New classifier that takes concatenated features
        self.fc = nn.Linear(feature_dim * num_frames, num_classes)
        
    def forward(self, x: torch.Tensor):
        # x shape: [batch, channels, num_frames, height, width]
        assert len(x.shape) == 5, "x must be a 5D tensor: [batch, channels, num_frames, height, width]"
        
        B, C, T, H, W = x.shape
        
        # Process each frame independently
        outputs = []
        for i in range(T):
            frame = x[:, :, i, :, :]  # [B, C, H, W]
            feat = self.model(frame)   
            outputs.append(feat)
        
        # Concatenate all features
        fused = torch.cat(outputs, dim=1)
        
        # Final classification
        out = self.fc(fused)  # [B, num_classes]
        
        return out

class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return

class C3D(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return




class ActionRecognitionModel(pl.LightningModule):
    """
    TODO
    """
    
    def __init__(
        self,
        model_type: str,
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = self.get_model()

    def get_model(self):
        if self.hparams['model_type'] == "aggregation":
            return single_frame_model()
        elif self.hparams['model_type'] == "early_fusion":
            return early_fusion_model()
        elif self.hparams['model_type'] == "late_fusion":
            return late_fusion_model()
        elif self.hparams['model_type'] == "CNN3D":
            return CNN3D()
        elif self.hparams['model_type'] == "C3D":
            return C3D()
        else:
            raise ValueError(f"Unknown model_type: {self.hparams['model_type']}")

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        return
    
    def validation_step(self, batch, batch_idx):
        
        return 
    
    def test_step(self, batch, batch_idx):

        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay']
        )
        
        scheduler = None
        
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

