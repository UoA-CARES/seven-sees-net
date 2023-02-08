import os
from handyolov5.labelhands import labelhands
from yolov5face.labelfaces import labelfaces
from yolov5person.labelperson import labelpersons
paths = ['val']
for folder in paths:
    labelhands(folder)


    labelpersons(folder, classes = [0])

    labelfaces(folder)