###########################################################################################
# Here you can find demo codes showing how to access and manipulate VDAO database.        #
#
# Developed by: Rafael Padilla
#        SMT - Signal Multimedia and Telecommunications Lab
#        COPPE - Universidade Federal do Rio de Janeiro
#        Last modification: Dec 17th 2017
############################################################################################
import os
import cv2
from VDAOVideo import VDAOVideo 
from VDAOHelper import VideoType, ImageExtension
from ObjectHelper import ObjectDatabase

####################################################################### 
# Example 00: Create the object VDAOVideo
####################################################################### 
# Creating a VDAOVideo object using a video path.
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Creating a VDAOVideo object identifying it as a reference video (it contains no lost objects)
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi", VideoType.Reference)

####################################################################### 
# Example 01: Obtain different information of a video
####################################################################### 
# Get the number of frames
numberOfFrames = myVideo.videoInfo.getNumberOfFrames()
print("Number of frames: "+ str(numberOfFrames))
# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print("Frame rate: "+ frameRate)


####################################################################### 
# Example 02: Obtain all information of a video
####################################################################### 
# Print all information about the video  
myVideo.videoInfo.printAllInformation()

####################################################################### 
# Example 03: Obtaining and saving specific frames within the video
####################################################################### 
# Get every 2 frames starting from 1 to 15 and save them
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=15, framesToSkip=2, outputFolder='/home/user/VDAO/outputs', filePrefix='myFrame_', showInfo=True)

####################################################################### 
# Example 04: Play a VDAO video
####################################################################### 
# Play video showing information about the video
myVideo.PlayVideo(showInfo=True)
# Play video showing information about the video
myVideo.PlayVideo(showInfo=False)

# Set the folder where the annotation file (.txt) and the video (.avi) are
folder = '/home/user/VDAO'
# Set the paths for the video and its annotation file
videoPath = os.path.join(folder, 'obj-sing-amb-part01-video02.avi')
annotationPath = os.path.join(folder, 'obj-sing-amb-part01-video02.txt')
# Create VDAOvideo object
vdao = VDAOVideo(videoPath, annotationFilePath=annotationPath)
# Play the video setting the parameter showBoundingBoxes to True
vdao.PlayVideo(showBoundingBoxes=True)


####################################################################### 
# Example 05: Get the bounding box of a mask
####################################################################### 
# Read image from the ALOI database
mask = cv2.imread('/home/user/ALOI/mask4/1/1_c1.png')
# Use static method to get the ROI of the mask
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(mask)
# Draw rectangle representing the ROI and show it
cv2.rectangle(mask,(min_x,min_y),(max_x,max_y),(0,255,0),1)
# Show image
cv2.imshow('ROI', mask)
cv2.waitKey(0)

####################################################################### 
# Example 06: Merge an object with its mask
####################################################################### 
# Read image from the ALOI database
myObjectPath = '/home/user/ALOI/png4/259/259_c.png'
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
blendedImage = ObjectDatabase.blendImageAndMask(objPath = myObjectPath, maskPath=myMaskPath)
cv2.imshow('Blended image', blendedImage)
cv2.waitKey(0)

####################################################################### 
# Example 07: Get random ALOI object and merge it with its mask 
####################################################################### 
# Creating ObjectDatabase passing the images and masks paths
aloi = ObjectDatabase(imagesPath='/home/user/ALOI/png4', \
                      masksPath='/home/user/ALOI/mask4')
# Get a random object and its mask
[mergedImage, (min_x, min_y, max_x, max_y)] = aloi.getRandomObject()
# Draw a green rectangle around the merged image
cv2.rectangle(mergedImage,(min_x,min_y),(max_x,max_y),(0,255,0),1)
cv2.imshow('image', mergedImage)
cv2.waitKey(0)

####################################################################### 
# FAQ
####################################################################### 

# Q: How do I slice a video from frame 1 to 1001 skipping every 100 frames and save it in JPEG format?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Save videos skipping every 100 frames and save it in the output folder
# The jpegQuality can go from 0 to 100 (the higher, the better). Default is 95.
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=1001, framesToSkip=100, \
outputFolder='/home/user/VDAO/outputs', \
extension=ImageExtension.JPG, jpegQuality=100)

# Q: How do I slice a video from frame 1 to 1001 skipping every 100 frames and save it in PNG format?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Save videos skipping every 100 frames and save it in the output folder
# The compressionLevel can go from 0 to 9 (higher value means a smaller size
# and longer compression time). Default is 3.
myVideo.SkipAndSaveFrames(startingFrame=1, endingFrame=1001, framesToSkip=100, \
outputFolder='/home/user/VDAO/outputs', \
extension=ImageExtension.PNG, compressionLevel=1)

#Q: How do I visualize a specific frame within a video?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frame number 530
ret, frame, _ = myVideo.GetFrame(530)
# Check if frame was successfully retrieved 
if ret:
    # Show frame
    cv2.imshow('frame', frame)
    cv2.waitKey(0)


#Q: How to know which compression method was used in a video?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get codec info
codec = myVideo.videoInfo.getCodecLongType()
print('Codec: %s' % codec)

#Q: How do I get the frame rate of a video?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print('Frame rate: %s' % frameRate)

#Q: How do I get the total number of frame in a video?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Get frames
totalFrames = myVideo.videoInfo.getNumberOfFrames()
print('Total frames: %s' % totalFrames)

#Q: How to merge an image and its mask in a single image?
# Get image and mask paths
myObjectPath = '/home/user/ALOI/png4/259/259_c.png'
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
# Merge image with its mask
blendedImage = ObjectDatabase.blendImageAndMask(objPath = myObjectPath, maskPath=myMaskPath)
# Show result
cv2.imshow('Blended image', blendedImage)
cv2.waitKey(0)

#Q: How to obtain a bounding box of a mask?
# Get mask path
myMaskPath = '/home/user/ALOI/mask4/259/259_c.png'
# Get bounding box
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(myMaskPath)
print("Upper left point: (%s, %s)" % (min_x, min_y))
print("Width: %s" % (max_x-min_x))
print("Height: %s" %  (max_y-min_y))

#Q: How do I play a VDAO video?
# Create myVideo object
myVideo = VDAOVideo("/home/user/VDAO/ref-mult-ext-part02-video01.avi")
# Play
myVideo.PlayVideo(showInfo=False)

#Q: How do I play a VDAO video showing the bounding boxes in the annotation file?
# Create myVideo object
myVideo = VDAOVideo(videoPath="/home/user/VDAO/obj-sing-amb-part01-video02.avi", annotationFilePath='/home/user/VDAO/obj-sing-amb-part01-video02.txt')
# Play
myVideo.PlayVideo(showInfo=False, showBoundingBoxes=True)

#Q: How do I add an object in an image making a smooth transition in the border regions?
objectPath = '/home/user/ALOI/png4/259/259_c.png'
maskPath = '/home/user/ALOI/mask4/259/259_c.png'
backgroundPath = '/home/user/Backgrounds/bg_1.jpg'
# Set some parameters for the image
scale = 1
angle = 30
# Blend mask, image and background
resultImage, _ = ObjectDatabase.blendImageAndBackground_2(objectPath, maskPath, backgroundPath, xIni=20, yIni=70, scaleFactor=scale, rotAngle=angle)
# Show result
cv2.imshow('final image', resultImage/255)
cv2.waitKey(0)