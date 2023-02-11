import os
import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms
from dataset.dataset import MultiModalDataset
from dataset.transforms import transform

try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
except:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

transforms = transform()

train_dataset = MultiModalDataset(ann_file='data/wlasl10/train_annotations.txt',
                            root_dir='data/wlasl10/rawframes',
                            clip_len=32,
                            resolution=224,
                            transforms = transforms,
                            frame_interval=1,
                            num_clips=1
                            )

test_dataset = MultiModalDataset(ann_file='data/wlasl10/test_annotations.txt',
                            root_dir='data/wlasl10/rawframes',
                            clip_len=32,
                            resolution=224,
                            transforms = transforms,
                            frame_interval=1,
                            num_clips=1
                            )

# Setting up dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True)


for i, (rgb, _, face, left_hand, right_hand, depth, flow, pose, targets) in enumerate(train_loader):
        rgb, face, left_hand, right_hand, depth, flow, pose, targets = rgb.to(device), face.to(device), left_hand.to(device), right_hand.to(device), depth.to(device), flow.to(device), pose.to(device), targets.to(device)