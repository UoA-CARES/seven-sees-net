import os
def labelfaces(path):
    cwd = os.getcwd()
    path = cwd + os.sep + path
    os.chdir('yolov5face')
    folders = next(os.walk(path))[1]
    
    for folder in folders:
        print(folder)

        cmd = 'python detect_face.py --weights yolov5s-face.pt --source ' +path + os.sep + folder 
        os.system(cmd)

if __name__ == "__main__":
    paths = ['val']
    for folder in paths:
        labelfaces(folder)
