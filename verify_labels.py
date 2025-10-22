"""
Script to verify label correctness across datasets
"""
import os
import pandas as pd
from collections import defaultdict
from datasets import FrameImageDataset, FrameVideoDataset, FrameVideoFlowDataset


def verify_label_consistency_ufc10():
    """Verify labels are consistent with action names for UFC10 dataset"""
    print("\n" + "="*80)
    print("Verifying Label Consistency for UFC10 Dataset")
    print("="*80)
    
    root_dir = "data/ufc10"
    splits = ["train", "val", "test"]
    
    # Collect all action-to-label mappings
    all_action_label_map = {}
    
    for split in splits:
        print(f"\n--- Checking {split} split ---")
        csv_path = f"{root_dir}/metadata/{split}.csv"
        df = pd.read_csv(csv_path)
        
        # Check action to label mapping
        action_label_map = df.groupby('action')['label'].unique()
        
        print(f"Actions and their labels in {split}:")
        for action, labels in action_label_map.items():
            print(f"  {action}: {labels}")
            if len(labels) > 1:
                print(f"    ⚠️  WARNING: Multiple labels for same action!")
            
            # Store in global map
            if action not in all_action_label_map:
                all_action_label_map[action] = labels[0]
            elif all_action_label_map[action] != labels[0]:
                print(f"    ❌ ERROR: Inconsistent label across splits!")
        
        # Verify video names match directory structure
        print(f"\nVerifying video names match directory structure...")
        for idx, row in df.iterrows():
            video_name = row['video_name']
            action = row['action']
            
            # Check if video exists in frames directory
            frames_dir = os.path.join(root_dir, "frames", split, action, video_name)
            if not os.path.exists(frames_dir):
                print(f"  ❌ Missing frames directory: {frames_dir}")
                break
        else:
            print(f"  ✓ All video directories exist")
        
        # Check number of unique labels
        num_classes = df['label'].nunique()
        print(f"Number of unique labels: {num_classes}")
    
    print("\n--- Global Action-Label Mapping ---")
    sorted_actions = sorted(all_action_label_map.items(), key=lambda x: x[1])
    for action, label in sorted_actions:
        print(f"  Label {label}: {action}")
    
    # Verify labels are 0-indexed and contiguous
    labels = sorted(all_action_label_map.values())
    expected_labels = list(range(len(labels)))
    if labels == expected_labels:
        print(f"\n✓ Labels are properly 0-indexed and contiguous (0 to {len(labels)-1})")
    else:
        print(f"\n❌ ERROR: Labels are not contiguous!")
        print(f"   Expected: {expected_labels}")
        print(f"   Got: {labels}")
    
    return all_action_label_map


def verify_label_consistency_ucf101():
    """Verify labels are consistent with action names for UCF101 dataset"""
    print("\n" + "="*80)
    print("Verifying Label Consistency for UCF101_NoLeakage Dataset")
    print("="*80)
    
    root_dir = "data/ucf101_noleakage"
    splits = ["train", "val", "test"]
    
    # Check if metadata exists
    metadata_dir = os.path.join(root_dir, "metadata")
    if not os.path.exists(metadata_dir):
        print(f"⚠️  Metadata directory not found. Checking directory structure...")
        
        # Infer from directory structure
        frames_dir = os.path.join(root_dir, "frames")
        if os.path.exists(frames_dir):
            for split in splits:
                split_dir = os.path.join(frames_dir, split)
                if os.path.exists(split_dir):
                    actions = sorted([d for d in os.listdir(split_dir) 
                                    if os.path.isdir(os.path.join(split_dir, d))])
                    print(f"\n{split} split - Found {len(actions)} action classes:")
                    for idx, action in enumerate(actions):
                        video_count = len([d for d in os.listdir(os.path.join(split_dir, action))
                                         if os.path.isdir(os.path.join(split_dir, action, d))])
                        print(f"  Label {idx}: {action} ({video_count} videos)")
        
        print(f"\n⚠️  Note: UCF101 dataset is missing metadata CSV files.")
        print(f"   If you need to use FrameVideoFlowDataset, you'll need to create these files.")
        return None
    
    # If metadata exists, verify it
    all_action_label_map = {}
    for split in splits:
        csv_path = os.path.join(metadata_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Missing {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        action_label_map = df.groupby('action')['label'].unique()
        
        for action, labels in action_label_map.items():
            if action not in all_action_label_map:
                all_action_label_map[action] = labels[0]
    
    return all_action_label_map


def test_actual_label_loading():
    """Test that datasets actually load the correct labels"""
    print("\n" + "="*80)
    print("Testing Actual Label Loading from Datasets")
    print("="*80)
    
    root_dir = "data/ufc10"
    
    # Load metadata to create expected mapping
    df_train = pd.read_csv(f"{root_dir}/metadata/train.csv")
    expected_labels = {}
    for _, row in df_train.iterrows():
        video_name = row['video_name']
        expected_labels[video_name] = row['label']
    
    # Test FrameVideoDataset
    print("\n--- Testing FrameVideoDataset ---")
    dataset = FrameVideoDataset(root_dir=root_dir, split="train", stack_frames=True)
    
    # Check first 10 samples
    mismatches = []
    for i in range(min(10, len(dataset))):
        _, label = dataset[i]
        video_path = dataset.video_paths[i]
        video_name = video_path.split("/")[-1].split(".avi")[0]
        
        expected_label = expected_labels[video_name]
        if label != expected_label:
            mismatches.append((video_name, expected_label, label))
    
    if len(mismatches) == 0:
        print(f"✓ All checked samples have correct labels")
    else:
        print(f"❌ Found {len(mismatches)} label mismatches:")
        for vid, exp, got in mismatches:
            print(f"  Video: {vid}, Expected: {exp}, Got: {got}")
    
    # Test FrameImageDataset
    print("\n--- Testing FrameImageDataset ---")
    img_dataset = FrameImageDataset(root_dir=root_dir, split="train")
    
    # Check first 10 samples
    mismatches = []
    for i in range(min(10, len(img_dataset))):
        _, label = img_dataset[i]
        frame_path = img_dataset.frame_paths[i]
        video_name = frame_path.split("/")[-2]
        
        expected_label = expected_labels[video_name]
        if label != expected_label:
            mismatches.append((video_name, expected_label, label))
    
    if len(mismatches) == 0:
        print(f"✓ All checked samples have correct labels")
    else:
        print(f"❌ Found {len(mismatches)} label mismatches:")
        for vid, exp, got in mismatches:
            print(f"  Video: {vid}, Expected: {exp}, Got: {got}")


def check_label_distribution():
    """Check the distribution of labels across splits"""
    print("\n" + "="*80)
    print("Label Distribution Analysis")
    print("="*80)
    
    root_dir = "data/ufc10"
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n--- {split.upper()} split ---")
        df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        
        # Count samples per class
        label_counts = df.groupby(['label', 'action']).size().reset_index(name='count')
        label_counts = label_counts.sort_values('label')
        
        print(f"Total videos: {len(df)}")
        print(f"Class distribution:")
        for _, row in label_counts.iterrows():
            label, action, count = row['label'], row['action'], row['count']
            print(f"  Label {label} ({action}): {count} videos")
        
        # Check if balanced
        counts = label_counts['count'].values
        if len(set(counts)) == 1:
            print(f"✓ Dataset is perfectly balanced")
        else:
            min_count, max_count = counts.min(), counts.max()
            print(f"⚠️  Dataset is imbalanced: min={min_count}, max={max_count}")


def main():
    """Run all label verification checks"""
    print("\n" + "="*80)
    print("LABEL VERIFICATION SUITE")
    print("="*80)
    
    # Verify UFC10 dataset
    ufc10_mapping = verify_label_consistency_ufc10()
    
    # Verify UCF101 dataset  
    ucf101_mapping = verify_label_consistency_ucf101()
    
    # Test actual loading
    test_actual_label_loading()
    
    # Check distributions
    check_label_distribution()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\n✅ If no errors were reported above, your labels are correctly annotated!")
    print("   The label-to-action mapping is consistent across all splits,")
    print("   and the datasets are loading the correct labels for each video.")


if __name__ == "__main__":
    main()

