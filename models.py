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
        assert (
            len(x.shape) == 4
        ), "x must be a 4D tensor: [batch, channels, height, width]"
        return self.model(x)


class early_fusion_model(nn.Module):
    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()

        # Create ResNet18
        self.model = tv_models.resnet18(num_classes=num_classes)

        # Replace first conv to accept C*T channels
        self.model.conv1 = nn.Conv2d(
            in_channels=3 * num_frames,
            out_channels=64,  #
            kernel_size=7,  #
            stride=2,  # Copied from resnet18.
            padding=3,  #
            bias=False,  #
        )

    def forward(self, x: torch.Tensor):
        assert (
            len(x.shape) == 5
        ), "x must be a 5D tensor: [batch, channels, num_frames, height, width]"

        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)  # [batch, channels*num_frames, height, width]
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
        assert (
            len(x.shape) == 5
        ), "x must be a 5D tensor: [batch, channels, num_frames, height, width]"

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
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = tv_models.video.r3d_18(num_classes=num_classes)

    def forward(self, x):
        assert (
            len(x.shape) == 5
        ), "x must be a 5D tensor: [batch, channels, num_frames, height, width]"
        out = self.model(x)
        return out


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
        num_frames: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.get_model()
        self.criterion = nn.CrossEntropyLoss()

    def get_model(self):
        if self.hparams['model_type'] == "single_frame":
            return single_frame_model(num_classes=self.hparams['num_classes'])
        elif self.hparams['model_type'] == "early_fusion":
            return early_fusion_model(num_classes=self.hparams['num_classes'], num_frames=self.hparams['num_frames'])
        elif self.hparams['model_type'] == "late_fusion":
            return late_fusion_model(num_classes=self.hparams['num_classes'], num_frames=self.hparams['num_frames'])
        elif self.hparams['model_type'] == "CNN3D":
            return CNN3D(num_classes=self.hparams['num_classes'])
        elif self.hparams['model_type'] == "C3D":
            return C3D()
        else:
            raise ValueError(f"Unknown model_type: {self.hparams['model_type']}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # batch should be (inputs, targets)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch  # batch should be (inputs, targets)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return 
    
    def test_step(self, batch, batch_idx):
        x, y = batch  # batch should be (inputs, targets)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            weight_decay=self.hparams['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams['max_epochs'], eta_min=1e-6)
        
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# create main function to test all models
def main():
    for model_type in ["single_frame", "early_fusion", "late_fusion", "CNN3D"]:
        model = ActionRecognitionModel(
            model_type=model_type,
            num_classes=10,
            num_frames=10,
        )
        if model_type == "single_frame":
            x = torch.randn(1, 3, 112, 112)
        elif model_type == "early_fusion":
            x = torch.randn(1, 3, 10, 112, 112)
        elif model_type == "late_fusion":
            x = torch.randn(1, 3, 10, 112, 112)
        elif model_type == "CNN3D":
            x = torch.randn(1, 3, 10, 112, 112)
        model.forward(x)
        print(model.forward(x).shape)

if __name__ == "__main__":
    main()