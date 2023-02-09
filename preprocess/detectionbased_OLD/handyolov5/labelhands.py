import os
def labelhands(path):
    folders = next(os.walk(path))[1]
    
    for folder in folders:
        print(folder)
        cmd = 'python handyolov5/demo.py ' + path + os.sep + folder + ' ' +  path + os.sep + folder
        os.system(cmd)
if __name__ == "__main__":
    paths = ['val']
    for folder in paths:
        labelhands(folder)
