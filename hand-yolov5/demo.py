import os
import os.path as osp
import sys

import cv2
import handdet


def detect_demo(imgs, savepath):
    for img in imgs:
        im = cv2.imread(img)
        res = handdet.detect(im)
        res = highestconf(res)
        print(res)
        rim = handdet.visualize(im, res)
        cv2.imwrite(savepath + os.sep + f"saved{osp.basename(img)}", rim)
    print("Done!")

def detect_text(imgs, savepath, ):

    for img in imgs:
        im = cv2.imread(img)
        res, preds= handdet.detect(im)
        res = highestconf(res)

        file = open(os.getcwd()+ os.sep+savepath + os.sep + "handboxes.txt","a")#append mode
        h,w,c = im.shape
        for det in res:
            bbox = det['bbox']
            left, top = bbox[0]/w, bbox[1]/h
            right, bottom = bbox[2]/w, bbox[3]/h
            
            file.write(img.split(os.sep)[-1] + ' ' + str(left)+ ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) +" \n")
    
        #im = cv2.rectangle(im, (int(left*w),int(top*h)), (int(right*w),int(bottom*h)), (0,234,242), 2)
        #cv2.imwrite("im.jpg", im)
        file.close()


    print("Done!")

def highestconf(dets):
    highestconf = 0
    highestconf1 = 0 
    highestconfs  = [0,0]
    for det in dets:
        print(det)
        conf = det['conf']
        print(conf)
        if(conf>highestconf):
            highestconf = conf
            highestconfs[1] = highestconfs[0]
            highestconfs[0] = det
        elif(conf>highestconf1):
            highestconf1 = conf
            highestconfs[1] = det       
    largest2= []
    for i in highestconfs:
        if(i!=0):
            largest2.append(i)
    return largest2


def crop_demo(img):
    im = cv2.imread(img)
    res = handdet.detect(im)
    crops = handdet.crop(im, res)
    for (i, crop) in enumerate(crops):
        cv2.imwrite(f"crop_{i}.png", crop)

if __name__ == '__main__':
    # guard
    
    # detect
    target = sys.argv[1]
    savepath = sys.argv[2]
    if osp.isdir(target):
        imgs = [osp.join(target, f) for f in sorted(os.listdir(target))]
    elif osp.isfile(target):
        imgs = [target]

    #detect_demo(imgs, savepath)
    detect_text(imgs, savepath)
    #crop_demo(imgs[0])