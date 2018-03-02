
import os
import fnmatch

# Ex: 
# in: '/home/rafael/thesis/simulations/data1/test_data/000001.jpg'
# out: '/home/rafael/thesis/simulations/data1/test_data/', '000001.jpg'
def splitPathFile(fileDataPath):
    idx = fileDataPath.rfind('/')
    p = fileDataPath[:idx+1] #path
    f = fileDataPath[idx+1:] #file
    return p,f

def getAllFilesRecursively(filePath, extension="*"):
    files = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(filePath)
        for f in fnmatch.filter(files, '*.'+extension)]
    return files
