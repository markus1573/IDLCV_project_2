import warnings

warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path

from datamodules import ActionRecognitionDataModule
from models import ActionRecognitionModel


def _validate_dataset_model_compatibility(dataset_type: str, model_type: str) -> None:
    """Validate that dataset type is compatible with model type."""
    if model_type == "single_frame":
        if dataset_type != "frame_image":
            raise ValueError(
                f"Model type '{model_type}' requires 'frame_image' dataset, "
                f"but got '{dataset_type}'"
            )
    elif model_type in ["early_fusion", "late_fusion", "CNN3D"]:
        if dataset_type != "frame_video":
            raise ValueError(
                f"Model type '{model_type}' requires 'frame_video' dataset, "
                f"but got '{dataset_type}'"
            )
    elif model_type == "flow_resnet":
        if dataset_type != "frame_video_flow":
            raise ValueError(
                f"Model type '{model_type}' requires 'frame_video_flow' dataset, "
                f"but got '{dataset_type}'"
            )
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train Action Recognition Model with Hydra configuration."""

    print(f"\nLoaded configuration from conf/config.yaml")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model type: {cfg.model.model_type}")

    # Validate dataset-model compatibility
    _validate_dataset_model_compatibility(cfg.data.dataset_type, cfg.model.model_type)

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Check for CUDA and enable Tensor Core optimization
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        print(f"CUDA available: {device_name}")

        # Check if GPU has Tensor Cores (RTX series, A100, V100, etc.)
        has_tensor_cores = any(
            keyword in device_name.upper()
            for keyword in ["RTX", "A100", "V100", "A40", "A30", "A10", "T4", "H100"]
        )

        if has_tensor_cores:
            torch.set_float32_matmul_precision("high")
            print("Enabled high precision matmul for Tensor Cores")
        else:
            print("GPU detected but no Tensor Cores found, using default precision")
    else:
        print("CUDA not available, using CPU")

    # Initialize data module
    datamodule = ActionRecognitionDataModule(
        root_dir=to_absolute_path(cfg.data.root_dir),
        dataset_type=cfg.data.dataset_type,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=tuple(cfg.data.image_size),
        n_sampled_frames=cfg.data.n_sampled_frames,
        stack_frames=cfg.data.stack_frames,
        augment=cfg.data.augment,
        seed=cfg.seed,
    )

    # Initialize model
    model = ActionRecognitionModel(
        model_type=cfg.model.model_type,
        num_classes=cfg.model.num_classes,
        num_frames=cfg.model.num_frames,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
    )

    # Hydra automatically creates versioned directories
    # Current working directory is the versioned directory for this run
    version_dir = os.getcwd()

    print(f"Starting training for {cfg.model.model_type}")
    print(f"Version directory: {version_dir}")

    # Ensure standard subdirectories exist within this run directory
    os.makedirs(os.path.join(version_dir, "best-val"), exist_ok=True)
    os.makedirs(os.path.join(version_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(version_dir, "outputs"), exist_ok=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="best-val",  # Save checkpoints under best-val within run dir
        filename="best-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,  # Only save the best model for this training run
        save_last=False,  # Don't save last checkpoint
    )

    callbacks = [
        checkpoint_cb,
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.training.patience,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # Loggers: CSV for local logs and wandb for online tracking
    loggers = [
        CSVLogger(save_dir="logs", name="", version=""),
    ]

    # Add wandb logger if configured
    if hasattr(cfg.logging, "wandb") and cfg.logging.wandb:
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            project=wandb_config.project,
            entity=wandb_config.entity if wandb_config.entity else None,
            tags=wandb_config.tags if wandb_config.tags else [],
            notes=wandb_config.notes if wandb_config.notes else "",
            offline=wandb_config.offline if wandb_config.offline else False,
            name=f"{cfg.model.model_type}_{cfg.logging.experiment_name}",
            save_dir="logs",
        )
        loggers.append(wandb_logger)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
        log_every_n_steps=10,
    )

    # Log hyperparameters to wandb if available
    if hasattr(cfg.logging, "wandb") and cfg.logging.wandb:
        wandb_logger = next(
            (logger for logger in loggers if isinstance(logger, WandbLogger)), None
        )
        if wandb_logger:
            # Log hyperparameters
            wandb_logger.log_hyperparams(
                {
                    "model_type": cfg.model.model_type,
                    "num_classes": cfg.model.num_classes,
                    "num_frames": cfg.model.num_frames,
                    "learning_rate": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                    "max_epochs": cfg.training.max_epochs,
                    "batch_size": cfg.data.batch_size,
                    "image_size": cfg.data.image_size,
                    "n_sampled_frames": cfg.data.n_sampled_frames,
                    "augment": cfg.data.augment,
                    "seed": cfg.seed,
                }
            )

    # Print configuration
    print("\n" + "=" * 50)
    print("Training Configuration:")
    print("=" * 50)
    print(hydra.utils.instantiate(cfg, _convert_="partial"))
    print("=" * 50 + "\n")

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model using the best checkpoint
    print("\n" + "=" * 50)
    print("Testing best model...")
    print("=" * 50 + "\n")

    # Resolve the best checkpoint path (prefer callback metadata, fallback to search)
    best_ckpt_path = (
        checkpoint_cb.best_model_path if checkpoint_cb.best_model_path else None
    )
    if not best_ckpt_path:
        ckpt_dir = os.path.join(version_dir, "best-val")
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if ckpt_files:
            best_ckpt = max(
                ckpt_files, key=lambda x: float(x.split("-")[1].replace(".ckpt", ""))
            )
            best_ckpt_path = os.path.join(ckpt_dir, best_ckpt)

    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        print(f"Loading best checkpoint: {best_ckpt_path}")
        best_model = ActionRecognitionModel.load_from_checkpoint(best_ckpt_path)
        test_results = trainer.test(best_model, datamodule=datamodule)
        if test_results:
            metrics = test_results[0]
            acc = metrics.get("test/acc")
            if acc is not None:
                print(f"Test accuracy (best ckpt): {acc:.4f}")
    else:
        print("No checkpoint found, testing current model...")
        test_results = trainer.test(model, datamodule=datamodule)
        if test_results:
            metrics = test_results[0]
            acc = metrics.get("test/acc")
            if acc is not None:
                print(f"Test accuracy (current model): {acc:.4f}")


if __name__ == "__main__":
    main()
