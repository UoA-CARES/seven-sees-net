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
    def crop(self,frames, cropname, posekey):
        rgbcrop =[]
        rgb = frames['rgb']
        w,h = rgb[0].size
        print(w,h)
        for i, frame in enumerate(rgb):
            img =  np.array(frame).copy() 
            bodybbox = frames['pose'][i][posekey]
            pad = 0
            print(posekey,bodybbox, i)
            x0 = bodybbox[0]
            if(x0<0):
                x0 = 0
                pad = 1
            x1 = bodybbox[2]
            if(x1>w):
                x1 = w
                pad = 1
            y0 = bodybbox[1]
            if(y0<0):
                y0=0
                pad = 1
            y1 = bodybbox[3]
            if(y1>h):
                y1=h
                pad = 1

            img = img[int(x0):int(x1), int(y0):int(y1)]
            if(pad):
                padimage = np.zeros((h,w,3), np.uint8)
                print(padimage.shape)
                padimage[x0:x1,y0:y1] = img
                
                img = padimage
            print(img.shape)
            rgbcrop.append(Image.fromarray(img))
            
        frames[cropname]= rgbcrop
        return frames
    def __call__(self, frames):
        #crop human bounding box
        frames = self.crop(frames, 'rgbcrop', 'body_bbox')
        frames = self.crop(frames, 'head', 'head')
        frames = self.crop(frames, 'right_hand', 'right_hand')
        frames = self.crop(frames, 'left_hand', 'left_hand')
        #global transforms apllied identically to all modalities (rgb, depth, flow)
        #frames = self.rotation(frames, 30, modalities)  #NEEDS POSE ROTATION
        
        frames = self.flip(frames, ['rgb','depth', 'flow', 'pose', 'rgbcrop', 'head', 'right_hand','left_hand'])
        #rgb only transforms
        frames = self.rgbtransforms(frames, ['rgb','rgbcrop'])
        frames = self.depthtransforms(frames, ['depth'])
        frames = self.flowtransforms(frames, ['flow'])
        return frames
