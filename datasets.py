from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T


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
        root_dir="data/ufc10",
        split="train",
        transform=None,
        stack_frames=True,
        n_sampled_frames=10,
    ):
        self.rgb_paths = sorted(glob(f"{root_dir}/frames/{split}/*/*/*.jpg"))
        self.flow_root = root_dir.replace("frames", "flows")
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return len(self.rgb_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        rgb_frame_path = self.rgb_paths[idx]
        rgb_frame_path = rgb_frame_path.replace("\\", "/")
        video_name = rgb_frame_path.split("/")[-2]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        # Derive frame directory and load both RGB and Flow
        video_frames_dir = rgb_frame_path.split("frame_")[0]
        rgb_frames = self.load_frames(video_frames_dir, mode="frames")
        flow_frames = self.load_frames(video_frames_dir.replace("frames", "flows"), mode="flows")

        if self.transform:
            rgb_frames = [self.transform(f) for f in rgb_frames]
            flow_frames = [self.transform(f) for f in flow_frames]
        else:
            rgb_frames = [T.ToTensor()(f) for f in rgb_frames]
            flow_frames = [T.ToTensor()(f) for f in flow_frames]

        # Stack frames
        if self.stack_frames:
            rgb_tensor = torch.stack(rgb_frames).permute(1, 0, 2, 3)   # [3, T, H, W]
            flow_tensor = torch.stack(flow_frames).permute(1, 0, 2, 3) # [2, T, H, W]
            combined = torch.cat([rgb_tensor, flow_tensor], dim=0)     # [5, T, H, W]
        else:
            combined = list(zip(rgb_frames, flow_frames))

        return combined, label

    def load_frames(self, frames_dir, mode="frames"):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg") if mode == "frames" else os.path.join(frames_dir, f"flow_{i}.jpg")
            if os.path.exists(frame_file):
                frame = Image.open(frame_file).convert("RGB")
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])  # duplicate last frame
                else:
                    raise FileNotFoundError(f"No {mode} frames found in {frames_dir}")
        return frames


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
