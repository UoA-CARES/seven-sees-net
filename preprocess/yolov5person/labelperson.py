import torch
import os
import glob
import cv2

def labelpersons(path, classes):
    

    
    cwd = os.getcwd()
    path = cwd + os.sep + path 
    print(path)
    os.chdir('yolov5person') 

    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
    folders = next(os.walk(path))[1]
    
    for folder in folders:
        imgs = glob.glob(path + os.sep + folder + os.sep + "*.jpg")
        print(folder,len(imgs))
        images = []
        for i in imgs:
            img = cv2.imread(i)
            h,w,c = img.shape
        
            preds = model(img) 

            highestconf = 0
            highestbbox = None
            for p in preds.xyxy[0].cpu().tolist():

                #only keep highest confidence and class
                if(p[4]>highestconf and p[5] in classes):
                    highestconf = p[4]
                    highestbbox = p[0:4]
            if(highestbbox!= None):    
                file = open( i.rsplit(os.sep,1)[0]+ os.sep + "bodyboxes.txt","a")#append mode
                file.write(i.split(os.sep)[-1] + ' ' + str(highestbbox[0]/w)+ ' ' + str(highestbbox[1]/h) + ' ' + str(highestbbox[2]/w) + ' ' + str(highestbbox[3]/h) +" \n")
    os.chdir(cwd)

if __name__ == "__main__":
    paths = ['val']
    classes = [0]
    for folder in paths:
        labelpersons(folder, classes)

