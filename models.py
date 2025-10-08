import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Optional


class ActionRecognitionModel(pl.LightningModule):
    """
    TODO
    """
    
    def __init__(
        self,
        model_type: str = "",
        num_classes: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        
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

