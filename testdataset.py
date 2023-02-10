from dataset.dataset import MultiModalDataset
from dataset.transforms import transform
from PIL import Image, ImageDraw
import numpy as np
import cv2
dataset = MultiModalDataset(ann_file='data/rawframes/annotations.txt',
                            root_dir='data/rawframes/test',
                            clip_len=32,
                            frame_interval=1,
                            num_clips=1)

results = dataset.load_video(idx=2)
#results = transform(results)  
def visualise(results):
    imgs= []
    for i in range(len(results['rgb'])): 
        img = results['rgb'][i]
        img =  np.array(img)[:, :, ::-1].copy() 
        #crop
        bodybbox = results['pose'][i]['body_bbox']
        img = img[int(bodybbox[0]):int(bodybbox[2]), int(bodybbox[1]):int(bodybbox[3])]
    

        keypoints = results['pose'][i]['keypoints']

        for j in keypoints:
            img = cv2.circle(img, (int(keypoints[j]['x']), int(keypoints[j]['y'])), radius=1, color=(0, 0, 255), thickness=1)
    

        cv2.imshow("", img)
        cv2.waitKey(0)



visualise(results)