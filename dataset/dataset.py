from torch.utils.data import Dataset
import torchvision.transforms as transform
import torch

from .pipelines.sampleframes import SampleFrames
from .pipelines.readpose import ReadPose
import os.path as osp
from PIL import Image
import numpy as np
import cv2

class MultiModalDataset(Dataset):
    """Samples frames using MMAction's SampleFrames and handles multimodal 
    rawframes data.

    Example of a annotation file:
    .. code-block:: txt
        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Required keys are "ann_file", "root_dir" and "clip_len".
    Args:
        ann_file (str): Path to annotation file.
        root_dir (str): Root directory of the rawframes.
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        rgb_prefix (str): File format for rgb image files.
        flow_prefix (str): File format for flow image files.
        depth_prefix (str): File format for depth image files.
    """

    def __init__(self,                
                ann_file,
                root_dir,
                clip_len,
                resolution,
                transforms=None,
                frame_interval=1,
                num_clips=1,
                rgb_prefix =  'img_{:05}.jpg',
                flow_prefix = 'flow_{:05}.jpg',
                depth_prefix = 'depth_{:05}.jpg'):

        self.ann_file = ann_file
        self.root_dir = root_dir
        self.rgb_prefix = rgb_prefix
        self.flow_prefix = flow_prefix
        self.depth_prefix = depth_prefix
        self.transforms = transforms
        self.resolution = resolution

        self.video_infos = self.load_annotations()
        self.read_pose = ReadPose()
        self.sample_frames = SampleFrames(clip_len=clip_len,
                                        frame_interval=frame_interval,
                                        num_clips=num_clips)

        self.img2tensorTransforms = transform.Compose(
                                                [
                                                    transform.Resize((self.resolution,self.resolution)),
                                                    transform.ToTensor(),
                                                ]
                                            )

    def __len__(self):
        return self.numvideos 
    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for i, line in enumerate(fin):
                line_split = line.strip().split()

                video_info = dict()
                video_info['video_path'] = osp.join(self.root_dir, line_split[0])
                video_info['start_index'] = 1
                video_info['total_frames'] = int(line_split[1])
                video_info['label'] = int(line_split[2])
                video_infos.append(video_info)
            self.numvideos = i + 1
        return video_infos

    def load_pose(self, video_path):
        """Load pose file under each video to get pose information."""
        pose_frames = dict()
        with open(osp.join(video_path, 'pose.txt'), 'r') as fin:
            for line in fin:
                pose_values, head, lhand, rhand, bodybbox, imgpath = self.read_pose(line)
                pose_frames[int(imgpath[4:9])] = dict(keypoints=pose_values,
                                        head=head,
                                        left_hand=lhand,
                                        right_hand=rhand,
                                        body_bbox=bodybbox,
                                        )
                

        return pose_frames

    def load_video(self, idx):
        """Load a video at a particular index and return rgb, flow, depth and 
        pose data in a dictionary.
        
        Args: 
            idx (int): The index position in the annotation file
            corresponding to a video.
        Returns:
            results (dict): The dictionary containing all the video data.
        """
        video_info = self.video_infos[idx]
        results = dict()
        results.update(video_info)
        
        self.sample_frames(results)
        frame_indices = results['frame_inds']
        video_path = results['video_path']

        pose_data = self.load_pose(video_path)

        rgb_frames = []
        flow_frames = []
        depth_frames = []
        pose_frames = []

        cache = dict()

        for frame in frame_indices:
            if frame not in cache:
                rgb_frame = Image.open(osp.join(video_path, self.rgb_prefix.format(frame)))
                depth_frame = Image.open(osp.join(video_path, self.depth_prefix.format(frame)))
                flow_frame = Image.open(osp.join(video_path, self.flow_prefix.format(frame)))
                pose_frame = pose_data[frame]

                # Add frames to cache
                cache[frame] = dict(rgb_frame=rgb_frame,
                                depth_frame=depth_frame,
                                flow_frame=flow_frame,
                                pose_frame=pose_frame)
                
                rgb_frames.append(rgb_frame)
                depth_frames.append(depth_frame)
                flow_frames.append(flow_frame)
                pose_frames.append(pose_frame)
                
            else:
                rgb_frames.append(cache[frame]['rgb_frame'])
                depth_frames.append(cache[frame]['depth_frame'])
                flow_frames.append(cache[frame]['flow_frame'])
                pose_frames.append(cache[frame]['pose_frame'])

        results['rgb'] = rgb_frames
        results['flow'] = flow_frames
        results['depth'] = depth_frames
        results['pose'] = pose_frames

        return results
        


    def visualise(self, idx=0, key = 'body_bbox'):
        results = self.load_video(idx=idx)
        results = self.transforms(results)  
        for i in range(len(results[key])): 
            img = results[key][i]
            img =  np.array(img)[:, :, ::-1].copy() 
            if(key=='body_bbox'):
                keypoints = results['pose'][i]['keypoints']
                for j in keypoints:
                    img = cv2.circle(img, (int(keypoints[j]['x']), int(keypoints[j]['y'])), radius=1, color=(0, 0, 255), thickness=1)
            cv2.imshow("", img)
            cv2.waitKey(0)

    def to_3dtensor(self, images):
        image_tensors = []
        for img in images:
            image_tensors.append(self.img2tensorTransforms(img).unsqueeze(dim=1))
        tensor = torch.cat(image_tensors, dim = 1)
        return tensor
    def pose2tensor(self, pose):
        points =[]
        for posen in pose:
            keypoints = posen['keypoints']           
            for i, point in enumerate(keypoints):
                points.append(keypoints[point]['x'])
                points.append(keypoints[point]['y'])
                points.append(keypoints[point]['confidence'])
        
        tensor = torch.tensor(points)
        return tensor
    def __getitem__(self, idx):
        #['rgb','depth', 'flow', 'pose', 'body_bbox', 'head', 'right_hand','left_hand']
        results = self.load_video(idx=idx)
        if(self.transforms != None):
            results = self.transforms(results)
            
        rgb = self.to_3dtensor(results['rgb'])
        body_bbox = self.to_3dtensor(results['body_bbox'])
        head = self.to_3dtensor(results['head'])
        left_hand = self.to_3dtensor(results['left_hand'])
        right_hand = self.to_3dtensor(results['right_hand'])
        depth  =  self.to_3dtensor(results['depth'])
        flow  =  self.to_3dtensor(results['flow'])
        
        pose =self.pose2tensor(results['pose'])
        
        label = torch.tensor(results['label'])
    
        return rgb, body_bbox, head, left_hand, right_hand, depth, flow,  pose, label

        