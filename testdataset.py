from dataset.dataset import MultiModalDataset
from dataset.transforms import transform
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw
import numpy as np
import cv2

transforms = transform()
dataset = MultiModalDataset(ann_file='data/rawframes/annotations.txt',
                            root_dir='data/rawframes/test',
                            clip_len=32,
                            resolution=224,
                            transforms = transforms,
                            frame_interval=1,
                            num_clips=1
                            )

dataset.visualise(key = 'head')

test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

crop, label = next(iter(test_loader))   
print(crop.shape, label)