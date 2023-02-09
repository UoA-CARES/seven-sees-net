import os
from pathlib import Path
import glob 
import cv2
path = 'val'

folderss = next(os.walk(path))[1]
folders = []
for f in folderss:
    if("_viz" not in f):
        folders.append(f)

for folder in folders:
    print(folder)
    Path(path + os.sep + folder + "_viz").mkdir(parents=True, exist_ok=True)
    
    with open(path + os.sep + folder+ os.sep + 'bodyboxes.txt') as f:
        bodylines = f.readlines()
        bodyboxes = []
        for l in bodylines:
            bodyboxes.append(l.split(' '))
    f.close()
    
    with open(path + os.sep + folder+ os.sep + 'handboxes.txt') as f:
        handlines = f.readlines()
        handboxes = []
        for l in handlines:
            handboxes.append(l.split(' '))
    f.close()
    
    with open(path + os.sep + folder+ os.sep + 'faceboxes.txt') as f:
        facelines = f.readlines()
        faceboxes = []
        for l in facelines:
            faceboxes.append(l.split(' '))
    f.close()
    #print(len(faceboxes))

    allboxes = faceboxes + handboxes + bodyboxes
    #print(len(allboxes))
    imgs = glob.glob(path + os.sep + folder + os.sep + "*.jpg")
   # print(imgs)
    for img in imgs:
        boxes = []
        for b in allboxes:
            
            #print(b, img.split(os.sep)[-1])
            if(img.split(os.sep)[-1] == b[0]):

                boxes.append(b[1:5])

        image= cv2.imread(img)
        h,w,c = image.shape

        for b in boxes:
 
            start = [int(float(b[0])*w), int(float(b[1])*h)]
            end = [int(float(b[2])*w), int(float(b[3])*h)]
            image = cv2.rectangle(image, start, end, (100,100,100), 2)
        cv2.imwrite(path + os.sep + folder + "_viz" + os.sep + img.split(os.sep)[-1], image)