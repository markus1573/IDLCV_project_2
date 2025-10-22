from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
# Stack all optical flows between successive frames using .npy files
import numpy as np
import torch.nn.functional as F


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/ufc10", split="train", transform=None):
        self.frame_paths = sorted(glob(f"{root_dir}/frames/{split}/*/*/*.jpg"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame_path = frame_path.replace("\\", "/")
        video_name = frame_path.split("/")[-2]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir="data/ufc10", split="train", transform=None, stack_frames=True
    ):

        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_path = video_path.replace("\\", "/")
        video_name = video_path.split("/")[-1].split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        video_frames_dir = (
            self.video_paths[idx].split(".avi")[0].replace("videos", "frames")
        )
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            if os.path.exists(frame_file):
                frame = Image.open(frame_file).convert("RGB")
                frames.append(frame)
            else:
                # If frame doesn't exist, use the last available frame
                if frames:
                    frames.append(frames[-1])  # Duplicate the last frame
                else:
                    # If no frames exist at all, this is a serious data issue
                    raise FileNotFoundError(
                        f"No frames found in {frames_dir}. Expected at least frame_1.jpg"
                    )

        return frames


class FrameVideoFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="data/ucf101_noleakage",
        split="train",
        image_transform=None,
        image_size=(112, 112),
        n_sampled_frames=10,
        seed: int = 42,
    ):
        # Metadata lists all videos for the split
        self.df = pd.read_csv(os.path.join(root_dir, "metadata", f"{split}.csv"))
        self.root_dir = root_dir
        self.split = split
        self.image_transform = image_transform
        self.image_size = image_size
        self.n_sampled_frames = n_sampled_frames
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = row["video_name"]
        action = row["action"]
        label = int(row["label"])

        # Directories
        frames_dir = os.path.join(
            self.root_dir, "frames", self.split, action, video_name
        )
        flows_dir = os.path.join(self.root_dir, "flows", self.split, action, video_name)

        # Select a single RGB frame (center frame by default)
        center_index = (self.n_sampled_frames + 1) // 2
        rgb_path = os.path.join(frames_dir, f"frame_{center_index}.jpg")
        if not os.path.exists(rgb_path):
            # Fall back to png if present
            rgb_path = os.path.join(frames_dir, f"frame_{center_index}.png")

        image = Image.open(rgb_path).convert("RGB")
        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = T.Compose([T.Resize(self.image_size), T.ToTensor()])(image)


        flow_tensors = []
        for i in range(1, self.n_sampled_frames):
            flow_path_npy = os.path.join(flows_dir, f"flow_{i}_{i+1}.npy")
            if not os.path.exists(flow_path_npy):
                raise FileNotFoundError(f"Missing flow: {flow_path_npy}")

            flow = np.load(flow_path_npy)
            # Ensure channel-first [2, H, W]
            if flow.ndim == 3 and flow.shape[0] in (2, 3):
                chw = flow
            elif flow.ndim == 3 and flow.shape[-1] in (2, 3):
                chw = np.transpose(flow, (2, 0, 1))
            else:
                raise ValueError(
                    f"Unexpected flow shape {flow.shape} at {flow_path_npy}"
                )
            flow = torch.from_numpy(chw).float()  # [2, H, W]
            # Robust resize via interpolate on 4D tensor to avoid channel mishandling
            flow = F.interpolate(
                flow.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            flow_tensors.append(flow)

        # Concatenate along channel dimension -> [2*(T-1), H, W]
        flow_stack = torch.cat(flow_tensors, dim=0).contiguous()

        # Sanity checks
        expected_c = 2 * (self.n_sampled_frames - 1)
        assert (
            flow_stack.shape[0] == expected_c
        ), f"Flow stack channels {flow_stack.shape[0]} != expected {expected_c}"
        assert (
            flow_stack.shape[1:] == image_tensor.shape[1:]
        ), f"Flow size {flow_stack.shape[1:]} must match image size {image_tensor.shape[1:]}"

        return (image_tensor, flow_stack), label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root_dir = "data/ufc10"

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    frameimage_dataset = FrameImageDataset(
        root_dir=root_dir, split="val", transform=transform
    )
    framevideostack_dataset = FrameVideoDataset(
        root_dir=root_dir, split="val", transform=transform, stack_frames=True
    )
    framevideolist_dataset = FrameVideoDataset(
        root_dir=root_dir, split="val", transform=transform, stack_frames=False
    )

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(
        framevideostack_dataset, batch_size=8, shuffle=False
    )
    framevideolist_loader = DataLoader(
        framevideolist_dataset, batch_size=8, shuffle=False
    )

    # for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    # for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]

    for video_frames, labels in framevideostack_loader:
        print(
            video_frames.shape, labels.shape
        )  # [batch, channels, number of frames, height, width]
