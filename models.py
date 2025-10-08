import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Optional
import torchvision.models as tv_models


class single_frame_model(nn.Module):
    def __init__(self):
        super().__init__()
        # resnet18
        self.model = tv_models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

class early_fusion_model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return

class late_fusion_model(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return

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

