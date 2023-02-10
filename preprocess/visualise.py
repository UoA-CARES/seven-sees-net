import os
import glob
import torch
import cv2

def decode(line):
    line = [l for l in line.replace(',',"").split(' ') if l != '' and l != '\n']
    imgpath = line[0]
    line = line[1:]
    line = [int(float(l)) for l in line]


    #line [imgpath [0], [y,x, conf][1], ...[4], ... [52], headlb[54], headrt,lhandlb[58], lhandrt, rhandlb[62], rhandrt, bboxlb[66], bboxrt[68] ]

    
    posepoints = line[0:51]
    head = line[51:55]
    lhand =line[55:59]
    rhand = line[59:63]
    bodybbox = line[63:]
    return posepoints, head, lhand, rhand, bodybbox, imgpath
paths = ['val']
for path in paths: #loop through parent folders ie [train, val, test]
    folders = next(os.walk(path))[1]
    for folder in folders:
        os.makedirs( path+ "_visualise"+os.sep + folder , exist_ok=True)
        with open(path+ os.sep + folder + os.sep + 'pose.txt') as f:
            lines = f.readlines()
        
        for line in lines:
            posepoints, head, lhand, rhand, bodybbox, imgpath = decode(line)
            try:
                img = cv2.imread(path + os.sep + folder + os.sep + imgpath)
           
                if(len(line)>1):
                    crop = img[bodybbox[0]:bodybbox[2], bodybbox[1]:bodybbox[3]]
                    crop = cv2.rectangle(crop, [head[0], head[1]], [head[2], head[3]] , (100,40,20))
                    crop = cv2.rectangle(crop, [lhand[0], lhand[1]], [lhand[2], lhand[3]] , (100,40,20))
                    crop = cv2.rectangle(crop, [rhand[0], rhand[1]], [rhand[2], rhand[3]] , (100,40,20))
                    for i in range(0,51,3):
                        crop = cv2.circle(crop, (posepoints[i+1],posepoints[i]), radius=1, color=(0, 0, 255), thickness=1)
                    cv2.imwrite(path+ "_visualise"+os.sep + folder +os.sep + imgpath,crop)
            except Exception as e: print(e)

    

   