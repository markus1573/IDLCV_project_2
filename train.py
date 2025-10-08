import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from omegaconf import DictConfig
import hydra

from datamodules import ActionRecognitionDataModule
from models import ActionRecognitionModel


def _validate_dataset_model_compatibility(dataset_type: str, model_type: str) -> None:
    """Validate that dataset type is compatible with model type."""
    # 2D models (single frame) should use frame_image dataset
    # 3D models (video) should use frame_video dataset
    if model_type in ["single_frame_model", "early_fusion_model", "late_fusion_model"]:
        if dataset_type != "frame_image":
            raise ValueError(
                f"Model type '{model_type}' requires 'frame_image' dataset, "
                f"but got '{dataset_type}'"
            )
    elif model_type in ["CNN3D", "C3D"]:
        if dataset_type != "frame_video":
            raise ValueError(
                f"Model type '{model_type}' requires 'frame_video' dataset, "
                f"but got '{dataset_type}'"
            )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train Action Recognition Model with Hydra configuration."""

    print(f"\nLoaded configuration from conf/config.yaml")

    # Validate dataset-model compatibility
    _validate_dataset_model_compatibility(cfg.data.dataset_type, cfg.model.model_type)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize data module
    datamodule = ActionRecognitionDataModule(
        root_dir=cfg.data.root_dir,
        dataset_type=cfg.data.dataset_type,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=tuple(cfg.data.image_size),
        n_sampled_frames=cfg.data.n_sampled_frames,
        stack_frames=cfg.data.stack_frames,
        augment=cfg.data.augment,
    )

    # Initialize model
    model = ActionRecognitionModel(
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        dropout=cfg.model.dropout,
    )

    # Create experiment directory
    experiment_dir = os.path.join(cfg.logging.log_dir, cfg.logging.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(
                cfg.logging.checkpoint_dir, cfg.logging.experiment_name
            ),
            filename="{epoch:02d}-{val/loss:.3f}-{val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Loggers
    logger = CSVLogger(save_dir=cfg.logging.log_dir, name=cfg.logging.experiment_name)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
    )

    # Print configuration
    print("\n" + "=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    print(hydra.utils.instantiate(cfg, _convert_="partial"))
    print("=" * 50 + "\n")

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    print("\n" + "=" * 50)
    print("Testing best model...")
    print("=" * 50 + "\n")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
