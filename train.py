import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from datamodules import ActionRecognitionDataModule
from models import ActionRecognitionModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model')
    
    # Data arguments
    parser.add_argument('--root_dir', type=str, default='data/ufc10',
                        help='Path to dataset root directory')
    parser.add_argument('--dataset_type', type=str, default='frame_video',
                        choices=['frame_image', 'frame_video'],
                        help='Type of dataset to use')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, nargs=2, default=[112, 112],
                        help='Image size (height width)')
    parser.add_argument('--n_sampled_frames', type=int, default=10,
                        help='Number of frames to sample from each video')
    parser.add_argument('--stack_frames', action='store_true', default=True,
                        help='Stack frames in video dataset')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='cnn3d',
                        choices=['simple_cnn', 'cnn3d', 'cnn_lstm'],
                        help='Type of model architecture')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of action classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for LSTM (cnn_lstm only)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (cnn_lstm only)')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (auto, gpu, cpu, mps)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['16', '32', 'bf16'],
                        help='Training precision')
    
    # Checkpoint and logging
    parser.add_argument('--experiment_name', type=str, default='action_recognition',
                        help='Name of the experiment')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run a quick test with 1 batch')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    
    # Initialize data module
    datamodule = ActionRecognitionDataModule(
        root_dir=args.root_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        n_sampled_frames=args.n_sampled_frames,
        stack_frames=args.stack_frames,
        augment=not args.no_augment,
    )
    
    # Initialize model
    model = ActionRecognitionModel(
        model_type=args.model_type,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    
    # Create experiment directory
    experiment_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_dir, args.experiment_name),
            filename='{epoch:02d}-{val/loss:.3f}-{val/acc:.3f}',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=args.patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar(),
    ]
    
    # Loggers
    loggers = [
        TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.experiment_name,
            version=None,
        ),
        CSVLogger(
            save_dir=args.log_dir,
            name=args.experiment_name,
        ),
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        log_every_n_steps=10,
    )
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    for key, value in vars(args).items():
        print(f"{key:25s}: {value}")
    print("="*50 + "\n")
    
    # Train the model
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Test the model
    print("\n" + "="*50)
    print("Testing best model...")
    print("="*50 + "\n")
    trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    # Print confusion matrix
    if hasattr(model, 'test_confusion'):
        confusion_matrix = model.test_confusion.compute()
        print("\nConfusion Matrix:")
        print(confusion_matrix)
        
        # Save confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            confusion_matrix.cpu().numpy(),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(args.num_classes),
            yticklabels=range(args.num_classes)
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, 'confusion_matrix.png'))
        print(f"\nConfusion matrix saved to {os.path.join(experiment_dir, 'confusion_matrix.png')}")


if __name__ == '__main__':
    main()

