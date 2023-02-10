import os
import glob
import torch
import cv2
import torchvision.transforms.functional as TF
import random
from PIL import Image
import numpy as np
class transform:
    def __init__(self):
        pass
    def rotation(self,frames, rotation, modalities):
        angle = random.randint(-rotation, rotation)
        for modality in modalities:
            if(modality in frames ):
                rgb = frames[modality]
                for i, frame in enumerate(rgb):
                    frame = TF.rotate(frame, angle)
                    frames[modality][i] = frame

                
        return frames
    def flip(self,frames, modalities):
        flip = random.random() > 0.5
        if(flip):
            for modality in modalities:
                if(modality in frames and modality!= 'pose'):
                    rgb = frames[modality]
                    for i, frame in enumerate(rgb):
                            frame = TF.hflip(frame)
                            frames[modality][i] = frame
                elif(modality == 'pose' and len(frames['pose'])>0 ):

                    width = frames['pose'][0]['body_bbox'][2] -frames['pose'][0]['body_bbox'][0] 
                    print(width) 
                    poseframes= frames['pose']
                    for idx,frame in enumerate(poseframes):
                        keypoints = frame['keypoints']
                        for keypoint in keypoints:
                            x = keypoints[keypoint]['x']
                            newx = width - x -1
                            frames['pose'][idx]['keypoints'][keypoint]['x'] = newx

                    


        return frames

    def rgbtransforms(self,frames, modalities):
        blur = int(random.randrange(10,50)/10)
        if(blur%2==0 and blur!= 1):
            blur+=1
        sharpness = random.randrange(10,20)/10
        brightness = (random.randrange(5,15)/10)
        hue = random.randrange(0,5)/100
        saturation = (random.randrange(5,15)/10)
        for modality in modalities:
            if(modality in frames):
                rgb = frames[modality]
                for i, frame in enumerate(rgb):
                    frame = TF.gaussian_blur(frame, blur)
                    frame = TF.adjust_sharpness(frame, sharpness)
                    frame = TF.adjust_brightness(frame, brightness)
                    frame = TF.adjust_hue(frame, hue)
                    frame = TF.adjust_saturation(frame, saturation)
                    frames[modality][i] = frame
        return frames

    def depthtransforms(self,frames, modalities):
        blur = int(random.randrange(10,50)/10)
        if(blur%2==0 and blur!= 1):
            blur+=1
        sharpness = random.randrange(10,20)/10
        brightness = (random.randrange(5,20)/10)
        hue = random.randrange(0,5)/100
        saturation = (random.randrange(5,20)/10)
        for modality in modalities:
            if(modality in frames):
                rgb = frames[modality]
                for i, frame in enumerate(rgb):
                    frame = TF.gaussian_blur(frame, blur)
                    frame = TF.adjust_sharpness(frame, sharpness)
                    frame = TF.adjust_brightness(frame, brightness)
                    frame = TF.adjust_hue(frame, hue)
                    frame = TF.adjust_saturation(frame, saturation)
                    frames[modality][i] = frame
        return frames

    def flowtransforms(self,frames, modalities):
        blur = int(random.randrange(10,150)/10)
        if(blur%2==0 and blur!= 1):
            blur+=1
        sharpness = random.randrange(10,15)/10
        for modality in modalities:
            if(modality in frames):
                rgb = frames[modality]
                for i, frame in enumerate(rgb):
                    frame = TF.gaussian_blur(frame, blur)
                    frame = TF.adjust_sharpness(frame, sharpness)               
                    frames[modality][i] = frame
        return frames
    def crop(self,frames):
        rgbcrop =[]
        rgb = frames['rgb']
        for i, frame in enumerate(rgb):
            img =  np.array(frame).copy() 
            bodybbox = frames['pose'][i]['body_bbox']
            img = img[int(bodybbox[0]):int(bodybbox[2]), int(bodybbox[1]):int(bodybbox[3])]
            rgbcrop.append(Image.fromarray(img))
            
        frames['rgbcrop']= rgbcrop
        return frames
    def __call__(self, frames):
        #crop human bounding box
        frames = self.crop(frames)
        #global transforms apllied identically to all modalities (rgb, depth, flow)
        #frames = self.rotation(frames, 30, modalities)  #NEEDS POSE ROTATION
        
        frames = self.flip(frames, ['rgb','depth', 'flow', 'pose', 'rgbcrop'])
        #rgb only transforms
        frames = self.rgbtransforms(frames, ['rgb','rgbcrop'])
        frames = self.depthtransforms(frames, ['depth'])
        frames = self.flowtransforms(frames, ['flow'])
        return frames
