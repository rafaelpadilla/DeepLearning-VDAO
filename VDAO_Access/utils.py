import os
import fnmatch
import cv2
import numpy as np
import itertools
import random
import math

# Ex: 
# in: '/home/rafael/thesis/simulations/data1/test_data/000001.jpg'
# out: '/home/rafael/thesis/simulations/data1/test_data/', '000001.jpg'
def splitPathFile(fileDataPath):
    idx = fileDataPath.rfind('/')
    p = fileDataPath[:idx+1] #path
    f = fileDataPath[idx+1:] #file
    return p,f

# Ex: 
# in: '/home/rafael/thesis/simulations/data1/test_data/'
# out: '{ 'home', 'rafael', 'thesis', 'simulations', 'data1', 'test_data' }
def splitPaths(path):
    folders = []
    indexes = [i for i, letter in enumerate(path) if letter == '/']
    for i in range(len(indexes)):
        if i+1 < len(indexes):
            item = path[indexes[i]:indexes[i+1]]
        else:
            item = path[indexes[i]:]
        item = item.replace('/','')
        if item != '':
            folders.append(item)
    return folders

def getAllFilesRecursively(filePath, extension="*"):
    files = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(filePath)
        for f in fnmatch.filter(files, '*.'+extension)]
    return files

# boxA = (Ax1,Ay1,Ax2,Ay2)
# boxB = (Bx1,By1,Bx2,By2)
def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]: 
        return False # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False # boxA is below boxB
    return True

# box = (Ax1,Ay1,Ax2,Ay2)
def getArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def getNonOverlappedBoxes(boxes):
    if len(boxes) == 1 or boxes == []:
        return boxes, [0]
    nonOverlappedBoxes = []
    nonOverlappedIdx = []
    # Get combination among all boxes
    combinations = list(itertools.combinations(boxes,2))
    # Loop through the pairs
    for combination in combinations:
        # If boxes do not intersect
        if boxesIntersect(combination[0], combination[1]) == False:
            if combination[0] not in nonOverlappedBoxes:
                nonOverlappedBoxes.append(combination[0])
                nonOverlappedIdx.append(boxes.index(combination[0]))
            if combination[1] not in nonOverlappedBoxes:
                nonOverlappedBoxes.append(combination[1])
                nonOverlappedIdx.append(boxes.index(combination[1]))
    return nonOverlappedBoxes, nonOverlappedIdx

def getOverlappedBoxes(boxes):
    if len(boxes) == 1 or boxes == []:
        return [], []
    overlappedBoxes = []
    overlappedIdx = []
    # Get combination among all boxes
    combinations = list(itertools.combinations(boxes,2))
    # Loop through the pairs
    for combination in combinations:
        # If boxes do not intersect
        if boxesIntersect(combination[0], combination[1]) == True:
            if combination[0] not in overlappedBoxes:
                overlappedBoxes.append(combination[0])
                overlappedIdx.append(boxes.index(combination[0]))
            if combination[1] not in overlappedBoxes:
                overlappedBoxes.append(combination[1])
                overlappedIdx.append(boxes.index(combination[1]))
    return overlappedBoxes, overlappedIdx

def removeIdxList(myList, indexesToRemove):
    newList = []
    for idx in range(len(myList)):
        # index should not be removed
        if idx not in indexesToRemove:
            newList.append(myList[idx])
    return newList

# bgShape = (height, width)
# boxSize = (height, width)
# scaleFator = (minFactor, maxFactor)
# rotationFator = (minAngle, maxAngle)
def getUniqueBoundingBoxes(bgShape, amountBoxes, boxSize, scaleFator=(100,100)):
    # Get background dimension
    bgHeight = bgShape[0]
    bgWidth = bgShape[1]
    # Get box original dimension
    boxHeight = boxSize[0]
    boxWidth = boxSize[1]

    boxes = []
    scales = []
    # for i in range(amountBoxes):
    while len(boxes) < amountBoxes:
        # Random scale
        scale = float(random.randint(scaleFator[0], scaleFator[1]))/100
        scales.append(scale)
        # Apply random scale
        boxHeight = boxSize[0]*scale
        boxWidth = boxSize[1]*scale
        # Apply random Xint Yint
        xPos = random.randint(0,bgWidth-boxWidth)  
        yPos = random.randint(0,bgHeight-boxHeight)
        # Define transformation matrix (rotation and scale)
        # transfMatriz = cv2.getRotationMatrix2D(, angle, 1.0)
        boxes.append([xPos, yPos, int(xPos+boxWidth), int(yPos+boxHeight)])
        combinations, overlappedIdx = getOverlappedBoxes(boxes)
        # remove overlapped
        boxes = removeIdxList(boxes, overlappedIdx)
        scales = removeIdxList(scales, overlappedIdx)
    ## Just display results graphically ##
    # img = np.zeros((bgHeight, bgWidth, 3), np.uint8)
    # for box in boxes:
    #     img = add_bb_into_image(img, box, (255,0,0), 1)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    return boxes, scales

def add_bb_into_image(image, boundingBox, color, thickness, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1
    safetyPixels = 0

    xIn = boundingBox[0]
    yIn = boundingBox[1]
    cv2.rectangle(image,(boundingBox[0], boundingBox[1]),(boundingBox[2], boundingBox[3]),(b,g,r), thickness)
    # Add label
    if label != None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (xIn+thickness, yIn-th+int(12.5*fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0: # if outside the image
            yin_bb = yIn+th # put it inside the bb
        r_Xin = xIn-int(thickness/2) 
        r_Yin = yin_bb-th-int(thickness/2) 
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image,(r_Xin,r_Yin-thickness), (r_Xin+tw+thickness*3,r_Yin+th+int(12.5*fontScale)), (b,g,r), -1)
        cv2.putText(image,label, (xin_bb, yin_bb), font, fontScale, (0,0,0), fontThickness, cv2.LINE_AA)
    return image

# Source: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# The lower the value, the more blurier the image is
def blur_measurement(image):
    if type(image) is str:
        if os.path.isfile(image):
            image = cv2.imread(image)
        else:
            raise IOError('It was not possible to load image %s' % image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channelR = image[:,:,2]
    channelG = image[:,:,1]
    channelB = image[:,:,0]
    try:
        grayVar =  cv2.Laplacian(gray, cv2.CV_64F)
        grayVar = grayVar.var()
        RVar = cv2.Laplacian(channelR, cv2.CV_64F).var()
        GVar = cv2.Laplacian(channelG, cv2.CV_64F).var()
        BVar = cv2.Laplacian(channelB, cv2.CV_64F).var()
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    return [RVar, GVar, BVar, grayVar]

def enlargeMask(mask, iterations):
    inv_mask = 255 - mask
    se = cv2.getStructuringElement(shape=cv2.MORPH_RECT,ksize=(3,3))
    enlargedMask = cv2.erode(src=inv_mask, kernel=se, iterations=iterations)
    enlargedMask_bin = enlargedMask/255
    diffMask = np.add(enlargedMask, mask)
    diffMask_bin = diffMask/255
    return enlargedMask, enlargedMask_bin.astype(np.uint8), diffMask, diffMask_bin.astype(np.uint8)
    
def euclideanDistance(list1, list2):
    # dist = 0
    # for i in range(len(vect1)):
    #     dist = dist + pow(vect1[i]-vect2[i],2)
    # return math.sqrt(dist)
    # OR
    return np.linalg.norm(np.asarray(list1)-np.asarray(list2))


def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def secsToMinSecMs(seconds):
    frac,whole = math.modf(round(seconds/60,9))
    _min = str(whole).replace('.0','') #minutes
    frac,whole = math.modf(frac*60)
    _sec = str(whole).replace('.0','') #seconds
    _ms = str(round(frac*1000,2)) #milliseconds
    return '%s min %s sec %s ms' % (_min, _sec, _ms)

