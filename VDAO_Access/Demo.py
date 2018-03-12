###########################################################################################
# Here you can find demo codes showing how to access and manipulate VDAO database.        #
#
# Developed by: Rafael Padilla
#        SMT - Signal Multimedia and Telecommunications Lab
#        COPPE - Universidade Federal do Rio de Janeiro
#        Last modification: Dec 17th 2017
############################################################################################

import cv2
from VDAOVideo import VDAOVideo 
from VDAOHelper import VideoType
from ObjectHelper import ObjectDatabase

# Change here the path to your video. 
videoPath = "/media/rafael/Databases/databases/VDAO/obj-mult-ext-part02-video01.avi"

# Create a VDAOVideo object informing if it its a reference or if it contains objects
myVideo = VDAOVideo(videoPath)

####################################################################### 
# Example 01: Obtaining different information of a video
####################################################################### 

# Get number of frames
numberOfFrames = myVideo.videoInfo.getNumberOfFrames()
print("Number of frames: "+ numberOfFrames)

# Get frame rate
frameRate = myVideo.videoInfo.getFrameRate()
print("Frame rate: "+ frameRate)

# Print all information about the video
myVideo.videoInfo.printAllInformation()

####################################################################### 
# Example 02: Showing a specific frame
####################################################################### 
frameNumber = 300
res, frame = myVideo.GetFrame(frameNumber, True) # Getting 180th frame
# Check if frame was retrieved 
# if res == True:
    # # Show frame
    # cv2.imshow('VDAO', frame)
    # cv2.waitKey(0)
    # # Save the frame
    # cv2.imwrite('/media/rafael/Databases/frame%d.jpg'%frameNumber,frame)

####################################################################### 
# Example 03: Saving every xth frame
####################################################################### 
# Get every 2 frames starting from 1 to 14 and save them
# You can set the last parameter to True if you want to show details of 
# the frame, otherwise set it to False or don't pass anything.
# myVideo.SkipAndSaveFrames(1, 14, 2, '/media/rafael/Databases', True)

####################################################################### 
# Example 04: Play a video
####################################################################### 
# Play video showing information about the video
# myVideo.PlayVideo(True)
# # Play video showing without showing information about the video
# myVideo.PlayVideo(False)

####################################################################### 
# Example 05: Play a video adding bounding box
####################################################################### 
# Given a VDAO video and its annotation file, play video showing
# bouding boxes around the objects
# folder = '/media/rafael/Databases/databases/VDAO/VDAO/Table_1-Shoe_Position_1'
# get video and annotation file
# video = os.path.join(folder, 'obj-sing-amb-part01-video01.avi')
# annotation = os.path.join(folder, 'obj-sing-amb-part01-video01.txt')
# Create vdao video object
# vdao = VDAOVideo(video, annotationFilePath=annotation)
# Play video
# vdao.PlayVideo(True, True)


####################################################################### 
# Example 06: Crop an object given its mask
####################################################################### 
# Read image from the ALOI database
mask = cv2.imread('/media/rafael/Databases/databases/ALOI/mask4/1/1_c1.png')
# Use static method to get the ROI of the mask
[min_x, min_y, max_x, max_y] = ObjectDatabase.getBoundingBoxMask(mask)
# Draw rectangle representing the ROI and show it
cv2.rectangle(mask,(min_x,min_y),(max_x,max_y),(0,255,0),1)
# cv2.imshow('ROI', mask)
# cv2.waitKey(0)

####################################################################### 
# Example 07: Merge object and its mask
####################################################################### 
# Create ObjectDatabase passing the images and masks paths
aloi = ObjectDatabase('/media/rafael/Databases/databases/ALOI/png4', \
                      '/media/rafael/Databases/databases/ALOI/mask4')
# Get a random object and its mask
[mergedImage, (min_x, min_y, max_x, max_y)] = aloi.getRandomObject()
# [mergedImage, (min_x, min_y, max_x, max_y)] = aloi.getRandomCroppedObject_ZeroDegrees(2)
cv2.rectangle(mergedImage,(min_x,min_y),(max_x,max_y),(0,255,0),1)
cv2.imshow('ROI', mergedImage)
cv2.waitKey(0)