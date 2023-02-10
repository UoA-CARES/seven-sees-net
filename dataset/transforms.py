import os
import glob
import torch
import cv2
import torchvision.transforms.functional as TF
import random
from PIL import Image
'''
a = Image.open("a.jpg")
b = Image.open("b.jpg")
d = Image.open("d.jpeg").convert('L')
o = Image.open("o.jpeg")
frames= {'rgb':[a,b], 'depth':[d], 'opticalflow':[o], 'pose':[]}
'''

def rotation(frames, rotation, modalities):
    angle = random.randint(-rotation, rotation)
    for modality in modalities:
        if(modality in frames ):
            rgb = frames[modality]
            for i, frame in enumerate(rgb):
                frame = TF.rotate(frame, angle)
                frames[modality][i] = frame

            
    return frames
def flip(frames, modalities):
    flip = 1#random.random() > 0.5
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

def rgbtransforms(frames, modalities):
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

def depthtransforms(frames, modalities):
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

def flowtransforms(frames, modalities):
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

def transform(frames):
   #global transforms apllied identically to all modalities (rgb, depth, flow)
   #frames = rotation(frames, 30, modalities)  #NEEDS POSE ROTATION
   frames = flip(frames, ['rgb','depth', 'flow', 'pose'])
   #rgb only transforms
   frames = rgbtransforms(frames, ['rgb'])
   frames = depthtransforms(frames, ['depth'])
   frames = flowtransforms(frames, ['flow'])
   return frames
'''
frames = transform(frames)


modalities = ['rgb','depth', 'opticalflow']
for m in modalities:
    frame = frames[m]
    for f in frame:
        f.show()
'''