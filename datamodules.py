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

