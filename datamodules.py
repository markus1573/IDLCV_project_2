import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameImageDataset, FrameVideoDataset


class ActionRecognitionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for action recognition datasets.
    
    Args:
        root_dir: Path to the dataset root directory
        dataset_type: Type of dataset ('frame_image' or 'frame_video')
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        image_size: Size to resize images to (height, width)
        n_sampled_frames: Number of frames to sample from each video (only for frame_video)
        stack_frames: Whether to stack frames in video dataset
        augment: Whether to use data augmentation
    """
    
    def __init__(
        self,
        root_dir: str = 'data/ufc10',
        dataset_type: str = 'frame_video',
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: tuple = (112, 112),
        n_sampled_frames: int = 10,
        stack_frames: bool = True,
        augment: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.n_sampled_frames = n_sampled_frames
        self.stack_frames = stack_frames
        self.augment = augment
        
        # Save hyperparameters
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        """Setup datasets for each stage (fit, test, predict)."""
        
        # Define transforms
        train_transform = self._get_train_transforms() if self.augment else self._get_val_transforms()
        val_transform = self._get_val_transforms()
        
        # Initialize datasets based on dataset_type
        if self.dataset_type == 'frame_image':
            self.train_dataset = FrameImageDataset(
                root_dir=self.root_dir,
                split='train',
                transform=train_transform
            )
            self.val_dataset = FrameImageDataset(
                root_dir=self.root_dir,
                split='val',
                transform=val_transform
            )
            self.test_dataset = FrameImageDataset(
                root_dir=self.root_dir,
                split='test',
                transform=val_transform
            )
        elif self.dataset_type == 'frame_video':
            self.train_dataset = FrameVideoDataset(
                root_dir=self.root_dir,
                split='train',
                transform=train_transform,
                stack_frames=self.stack_frames
            )
            self.val_dataset = FrameVideoDataset(
                root_dir=self.root_dir,
                split='val',
                transform=val_transform,
                stack_frames=self.stack_frames
            )
            self.test_dataset = FrameVideoDataset(
                root_dir=self.root_dir,
                split='test',
                transform=val_transform,
                stack_frames=self.stack_frames
            )
            # Update n_sampled_frames for all datasets
            self.train_dataset.n_sampled_frames = self.n_sampled_frames
            self.val_dataset.n_sampled_frames = self.n_sampled_frames
            self.test_dataset.n_sampled_frames = self.n_sampled_frames
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
    
    def _get_train_transforms(self):
        """Get training data augmentation transforms."""
        return T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self):
        """Get validation/test transforms (no augmentation)."""
        return T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


if __name__ == '__main__':
    """Test the DataModule to ensure it works correctly."""
    
    print("="*60)
    print("Testing ActionRecognitionDataModule")
    print("="*60)
    
    # Test with frame_video dataset (default)
    print("\n1. Testing with frame_video dataset (default)")
    print("-"*60)
    
    datamodule = ActionRecognitionDataModule(
        root_dir='data/ufc10',
        dataset_type='frame_video',
        batch_size=4,
        num_workers=2,
        image_size=(112, 112),
        n_sampled_frames=10,
        stack_frames=True,
        augment=True,
    )
    
    # Setup the datamodule
    print("Setting up datamodule...")
    datamodule.setup()
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    frames, labels = train_batch
    print(f"  Train batch shapes:")
    print(f"    Frames: {frames.shape}")  # Expected: (batch, channels, time, height, width)
    print(f"    Labels: {labels.shape}")  # Expected: (batch,)
    print(f"    Frames dtype: {frames.dtype}")
    print(f"    Labels dtype: {labels.dtype}")
    print(f"    Frames min/max: {frames.min():.3f} / {frames.max():.3f}")
    print(f"    Unique labels: {labels.unique().tolist()}")
    
    # Test val dataloader
    print("\nTesting val dataloader...")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    frames, labels = val_batch
    print(f"  Val batch shapes:")
    print(f"    Frames: {frames.shape}")
    print(f"    Labels: {labels.shape}")
    
    # Test test dataloader
    print("\nTesting test dataloader...")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    frames, labels = test_batch
    print(f"  Test batch shapes:")
    print(f"    Frames: {frames.shape}")
    print(f"    Labels: {labels.shape}")
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"  Train: {len(datamodule.train_dataset)} videos")
    print(f"  Val:   {len(datamodule.val_dataset)} videos")
    print(f"  Test:  {len(datamodule.test_dataset)} videos")
    
    # Test with frame_image dataset
    print("\n" + "="*60)
    print("2. Testing with frame_image dataset")
    print("-"*60)
    
    datamodule_image = ActionRecognitionDataModule(
        root_dir='data/ufc10',
        dataset_type='frame_image',
        batch_size=8,
        num_workers=2,
        image_size=(112, 112),
        augment=False,
    )
    
    datamodule_image.setup()
    
    # Test train dataloader
    print("\nTesting train dataloader...")
    train_loader_image = datamodule_image.train_dataloader()
    train_batch_image = next(iter(train_loader_image))
    images, labels = train_batch_image
    print(f"  Train batch shapes:")
    print(f"    Images: {images.shape}")  # Expected: (batch, channels, height, width)
    print(f"    Labels: {labels.shape}")  # Expected: (batch,)
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"  Train: {len(datamodule_image.train_dataset)} frames")
    print(f"  Val:   {len(datamodule_image.val_dataset)} frames")
    print(f"  Test:  {len(datamodule_image.test_dataset)} frames")
    
    print("\n" + "="*60)
    print("✓ DataModule test completed successfully!")
    print("="*60)
    print("\nYou can now use this DataModule for training with:")
    print("  python train.py --dataset_type frame_video")
    print("  python train.py --dataset_type frame_image")
    print("="*60)