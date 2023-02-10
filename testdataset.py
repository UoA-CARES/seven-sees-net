from dataset.dataset import MultiModalDataset

dataset = MultiModalDataset(ann_file='data/rawframes/annotations.txt',
                            root_dir='data/rawframes/test',
                            clip_len=32,
                            frame_interval=2,
                            num_clips=1)

results = dataset.load_video(idx=0)
print(results)