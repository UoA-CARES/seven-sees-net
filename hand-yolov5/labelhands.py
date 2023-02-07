import os
def labelfolder(path):
    folders = next(os.walk(path))[1]
    
    for folder in folders:
        print(folder)
        cmd = 'python demo.py ' + path + os.sep + folder + ' ' +  path + os.sep + folder
        os.system(cmd)
paths = ['val']
for folder in paths:
    labelfolder(folder)
