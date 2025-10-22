"""
Test script to verify all dataset classes load data correctly
"""
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import FrameImageDataset, FrameVideoDataset, FrameVideoFlowDataset


def test_frame_image_dataset():
    """Test FrameImageDataset class"""
    print("\n" + "="*80)
    print("Testing FrameImageDataset")
    print("="*80)
    
    root_dir = "data/ufc10"
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    try:
        # Test on train split
        train_dataset = FrameImageDataset(root_dir=root_dir, split="train", transform=transform)
        print(f"✓ Train dataset created successfully")
        print(f"  - Number of frames: {len(train_dataset)}")
        
        # Load a sample
        frame, label = train_dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  - Frame shape: {frame.shape}")
        print(f"  - Label: {label}")
        print(f"  - Label type: {type(label)}")
        
        # Test dataloader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        batch_frames, batch_labels = next(iter(train_loader))
        print(f"✓ Successfully loaded batch")
        print(f"  - Batch frames shape: {batch_frames.shape}")
        print(f"  - Batch labels shape: {batch_labels.shape}")
        
        # Test validation split
        val_dataset = FrameImageDataset(root_dir=root_dir, split="val", transform=transform)
        print(f"✓ Val dataset created successfully")
        print(f"  - Number of frames: {len(val_dataset)}")
        
        # Test test split
        test_dataset = FrameImageDataset(root_dir=root_dir, split="test", transform=transform)
        print(f"✓ Test dataset created successfully")
        print(f"  - Number of frames: {len(test_dataset)}")
        
        print("\n✅ FrameImageDataset: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FrameImageDataset: FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_video_dataset():
    """Test FrameVideoDataset class"""
    print("\n" + "="*80)
    print("Testing FrameVideoDataset")
    print("="*80)
    
    root_dir = "data/ufc10"
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    
    try:
        # Test with stack_frames=True
        print("\n--- Testing with stack_frames=True ---")
        train_dataset_stacked = FrameVideoDataset(
            root_dir=root_dir, split="train", transform=transform, stack_frames=True
        )
        print(f"✓ Train dataset (stacked) created successfully")
        print(f"  - Number of videos: {len(train_dataset_stacked)}")
        
        # Load a sample
        frames, label = train_dataset_stacked[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  - Frames shape: {frames.shape} (expected: [C, T, H, W])")
        print(f"  - Label: {label}")
        
        # Test dataloader
        train_loader = DataLoader(train_dataset_stacked, batch_size=4, shuffle=False)
        batch_frames, batch_labels = next(iter(train_loader))
        print(f"✓ Successfully loaded batch")
        print(f"  - Batch frames shape: {batch_frames.shape} (expected: [B, C, T, H, W])")
        print(f"  - Batch labels shape: {batch_labels.shape}")
        
        # Test with stack_frames=False
        print("\n--- Testing with stack_frames=False ---")
        train_dataset_list = FrameVideoDataset(
            root_dir=root_dir, split="train", transform=transform, stack_frames=False
        )
        print(f"✓ Train dataset (list) created successfully")
        print(f"  - Number of videos: {len(train_dataset_list)}")
        
        # Load a sample
        frames_list, label = train_dataset_list[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  - Type: {type(frames_list)}")
        print(f"  - Number of frames: {len(frames_list)}")
        if len(frames_list) > 0:
            print(f"  - First frame shape: {frames_list[0].shape}")
        print(f"  - Label: {label}")
        
        # Test validation split
        val_dataset = FrameVideoDataset(root_dir=root_dir, split="val", transform=transform, stack_frames=True)
        print(f"✓ Val dataset created successfully")
        print(f"  - Number of videos: {len(val_dataset)}")
        
        # Test test split
        test_dataset = FrameVideoDataset(root_dir=root_dir, split="test", transform=transform, stack_frames=True)
        print(f"✓ Test dataset created successfully")
        print(f"  - Number of videos: {len(test_dataset)}")
        
        print("\n✅ FrameVideoDataset: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FrameVideoDataset: FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_video_flow_dataset():
    """Test FrameVideoFlowDataset class"""
    print("\n" + "="*80)
    print("Testing FrameVideoFlowDataset")
    print("="*80)
    
    root_dir = "data/ucf101_noleakage"
    image_transform = T.Compose([T.Resize((112, 112)), T.ToTensor()])
    
    try:
        # Check if metadata exists
        metadata_dir = os.path.join(root_dir, "metadata")
        if not os.path.exists(metadata_dir):
            print(f"⚠️  Warning: Metadata directory not found at {metadata_dir}")
            print(f"   The dataset requires CSV metadata files for each split")
            print(f"   Expected structure: {root_dir}/metadata/{{train,val,test}}.csv")
            
            # Check if we can create metadata from directory structure
            print(f"\n   Attempting to infer metadata from directory structure...")
            
            # Check available splits
            frames_dir = os.path.join(root_dir, "frames")
            if os.path.exists(frames_dir):
                splits = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
                print(f"   Available splits: {splits}")
                
                if len(splits) > 0:
                    print(f"\n   ⚠️  ISSUE FOUND: FrameVideoFlowDataset requires metadata CSV files")
                    print(f"   The ucf101_noleakage dataset is missing metadata CSV files.")
                    print(f"   These files should contain: video_name, action, label columns")
                    return False
            else:
                print(f"   ❌ Frames directory not found at {frames_dir}")
                return False
        
        # Test on train split
        print("\n--- Testing train split ---")
        train_dataset = FrameVideoFlowDataset(
            root_dir=root_dir, 
            split="train", 
            image_transform=image_transform,
            n_sampled_frames=10
        )
        print(f"✓ Train dataset created successfully")
        print(f"  - Number of videos: {len(train_dataset)}")
        
        # Load a sample
        (image, flow_stack), label = train_dataset[0]
        print(f"✓ Successfully loaded sample 0")
        print(f"  - Image shape: {image.shape} (expected: [3, H, W])")
        print(f"  - Flow stack shape: {flow_stack.shape} (expected: [2*(T-1), H, W])")
        print(f"  - Label: {label}")
        
        # Verify dimensions match
        expected_flow_channels = 2 * (10 - 1)  # 2 channels per flow, 9 flows for 10 frames
        assert flow_stack.shape[0] == expected_flow_channels, \
            f"Flow channels mismatch: got {flow_stack.shape[0]}, expected {expected_flow_channels}"
        assert image.shape[1:] == flow_stack.shape[1:], \
            f"Spatial dimensions mismatch: image {image.shape[1:]}, flow {flow_stack.shape[1:]}"
        print(f"✓ Dimensions validated correctly")
        
        # Test dataloader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=0)
        (batch_images, batch_flows), batch_labels = next(iter(train_loader))
        print(f"✓ Successfully loaded batch")
        print(f"  - Batch images shape: {batch_images.shape}")
        print(f"  - Batch flows shape: {batch_flows.shape}")
        print(f"  - Batch labels shape: {batch_labels.shape}")
        
        # Test validation split
        print("\n--- Testing val split ---")
        val_dataset = FrameVideoFlowDataset(
            root_dir=root_dir, 
            split="val", 
            image_transform=image_transform,
            n_sampled_frames=10
        )
        print(f"✓ Val dataset created successfully")
        print(f"  - Number of videos: {len(val_dataset)}")
        
        # Test test split
        print("\n--- Testing test split ---")
        test_dataset = FrameVideoFlowDataset(
            root_dir=root_dir, 
            split="test", 
            image_transform=image_transform,
            n_sampled_frames=10
        )
        print(f"✓ Test dataset created successfully")
        print(f"  - Number of videos: {len(test_dataset)}")
        
        print("\n✅ FrameVideoFlowDataset: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ FrameVideoFlowDataset: FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DATASET VERIFICATION TEST SUITE")
    print("="*80)
    
    results = {
        "FrameImageDataset": test_frame_image_dataset(),
        "FrameVideoDataset": test_frame_video_dataset(),
        "FrameVideoFlowDataset": test_frame_video_flow_dataset(),
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for dataset_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{dataset_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All datasets are working correctly!")
    else:
        print("\n⚠️  Some datasets have issues that need to be addressed.")
    
    return all_passed


if __name__ == "__main__":
    main()

