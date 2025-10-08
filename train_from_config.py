import argparse
import os
import yaml
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


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model from Config')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run a quick test with 1 batch')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded configuration from {args.config}")
    
    # Set seed for reproducibility
    pl.seed_everything(config['seed'], workers=True)
    
    # Initialize data module
    datamodule = ActionRecognitionDataModule(
        root_dir=config['data']['root_dir'],
        dataset_type=config['data']['dataset_type'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size']),
        n_sampled_frames=config['data']['n_sampled_frames'],
        stack_frames=config['data']['stack_frames'],
        augment=config['data']['augment'],
    )
    
    # Initialize model
    model = ActionRecognitionModel(
        model_type=config['model']['model_type'],
        num_classes=config['model']['num_classes'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        dropout=config['model']['dropout'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
    )
    
    # Create experiment directory
    experiment_dir = os.path.join(
        config['logging']['log_dir'],
        config['logging']['experiment_name']
    )
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(
                config['logging']['checkpoint_dir'],
                config['logging']['experiment_name']
            ),
            filename='{epoch:02d}-{val/loss:.3f}-{val/acc:.3f}',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=config['training']['patience'],
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar(),
    ]
    
    # Loggers
    loggers = [
        TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name=config['logging']['experiment_name'],
            version=None,
        ),
        CSVLogger(
            save_dir=config['logging']['log_dir'],
            name=config['logging']['experiment_name'],
        ),
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
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
    print(yaml.dump(config, default_flow_style=False))
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
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 10))
            class_labels = [config['classes'][i] for i in range(config['model']['num_classes'])]
            sns.heatmap(
                confusion_matrix.cpu().numpy(),
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_dir, 'confusion_matrix.png'))
            print(f"\nConfusion matrix saved to {os.path.join(experiment_dir, 'confusion_matrix.png')}")
        except ImportError:
            print("\nMatplotlib or seaborn not available. Skipping confusion matrix visualization.")


if __name__ == '__main__':
    main()

