# -*- coding: utf-8 -*-
import cv2
import os
import sys
import numpy as np
import utils
# To get video information
from VDAOHelper import VDAOInfo, VideoType

class VDAOVideo:
    """
	VDAOVideo class contains important information all important methods and tools to access VDAO videos.

        Developed by: Rafael Padilla
        SMT - Signal Multimedia and Telecommunications Lab
        COPPE - Universidade Federal do Rio de Janeiro
        Last modification: Dec 9th 2017
    """
    
    def __init__(self, videoPath, videoType=None, annotationFile=None):
        # Defines if the video is reference or with objects by the name of the file
        if videoType == None:
            _, fileName = utils.splitPathFile(videoPath)
            if fileName.startswith('ref-'):
                videoType = VideoType.Reference
            elif fileName.startswith('obj-'):
                videoType = VideoType.WithObjects
        else:
            self.videoType = VideoType
    
        self.videoPath = videoPath
        self.videoInfo = VDAOInfo(self.videoPath)

        if annotationFile != None:
            self.annotationFile = annotationFile
    
    def PlayVideo(self, showFrame=True, showBoundingBoxes=False):
        cap = cv2.VideoCapture(self.videoPath)
        fps = self.videoInfo.getFramesPerSecond() #or cap.get(cv2.CAP_PROP_FPS)
        waitFraction = int(1000/fps)
        # Parameters to display video info
        width, height = self.videoInfo.getWidthHeight()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thicknessFont = 1
        scaleText = .7
        colorText = (255,255,255) # G,B,R
        spaceBtwLines = 10
        outsideMargin = 10
        numberOfLines = 4 # number of text lines
        # Based on the textSize, define the new frame size
        textSize = cv2.getTextSize(str(self.videoInfo.getNumberOfFrames()), font, scaleText, thicknessFont)[0]
        frameHeight = (textSize[1]*numberOfLines)+( ((numberOfLines-1)*spaceBtwLines) + (2*outsideMargin) )
        framedImageHeight = height+frameHeight
        framedImage = np.zeros((framedImageHeight, width, 3), np.uint8) # VDAO videos have 3 channels
        # Define positions of the texts to appear (bottom-left is the reference)
        originText1 = (outsideMargin, height+outsideMargin+textSize[1])
        originText2 = (outsideMargin, height+outsideMargin+(textSize[1]*2)+spaceBtwLines)
        originText3 = (outsideMargin, height+outsideMargin+(textSize[1]*3)+(spaceBtwLines*2))
        originText4 = (outsideMargin, height+outsideMargin+(textSize[1]*4)+(spaceBtwLines*3))
        # Define texts
        firstLine = "VDAO: Video Database of Abandoned Objects"
        secondLine = "File: " + self.videoInfo.getFileName()
        thirdLine = "Frame rate: "+ self.videoInfo.getFrameRate()
        fourthLine = "Frame: %d/"+str(self.videoInfo._numberOfFrames)
        # Define and resize logo to fit on the screen
        logo = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/images/logo.png')
        hlogo,wlogo = logo.shape[0], logo.shape[1]
        logo = cv2.resize(logo, (0,0), fx=(frameHeight-(2*outsideMargin))/hlogo, fy=(frameHeight-(2*outsideMargin))/hlogo, interpolation = cv2.INTER_CUBIC) 
        hlogo,wlogo = logo.shape[0], logo.shape[1]
        logoPosition = (height+int(frameHeight/2)-int(hlogo/2), height+int(frameHeight/2)-int(hlogo/2)+hlogo, width-outsideMargin-wlogo, width-outsideMargin)

        # Start reading -> OpenCV counts frames from 0 to nrFrames-1. We will count from 1 to nframes
        ret,frame = cap.read()
        frameCount = 1
        ret = True
        while(ret == True):
            
            if showFrame:
                # VDAO videos have 3 channels
                framedImage = np.zeros((framedImageHeight, width, 3), np.uint8)
                framedImage[0:height, 0:width, : ] = frame
                # Add logo
                framedImage[logoPosition[0]:logoPosition[1], logoPosition[2]:logoPosition[3], : ] = logo
                # Add text into
                cv2.putText(framedImage, firstLine, org=originText1, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
                cv2.putText(framedImage, secondLine, org=originText2, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
                cv2.putText(framedImage, thirdLine, org=originText3, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
                cv2.putText(framedImage, fourthLine%frameCount, org=originText4, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)

            else:
                framedImage = frame
                
            # TODO: Falta implementar o parser do txt e marcar os frames com um retângulo
            # Usar a classe VDAOAnnotatedFile
            # if frameCount == 2902:
            #     cv2.rectangle(framedImage,(0,430),(2,453),(0,255,0),3)

            # Show framedImage
            cv2.imshow('VDAO', framedImage)
            cv2.waitKey(waitFraction)

            ret,frame = cap.read()        
            frameCount = frameCount+1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release video resources
        cap.release()
        cv2.destroyAllWindows()

    def GetFrame(self, frameNumber, withInfo=False):
        """Frame count starts from 1 to max frames -> self.infoVideo.getNumberOfFrames()"""

        maxNumberFrames = self.videoInfo.getNumberOfFrames()
        if  maxNumberFrames == None:
            raise IOError('It was not possible to detect the number of frames in the file')
            
        # Check if frame exist within the video
        if frameNumber < 1 or frameNumber > int(maxNumberFrames):
            raise IOError('Frame number must be between 1 and '+self.videoInfo.getNumberOfFrames())

        cap = cv2.VideoCapture(self.videoPath)
        fr = int(self.videoInfo.getFramesPerSecond())
        frameTime = 1000 * (frameNumber-1)/fr
        cap.set(cv2.CAP_PROP_POS_MSEC, frameTime)
        ret,frame = cap.read()
        cap.release()
        if ret & withInfo:
            frame = self.AddInfoToFrame(frame, frameNumber)
        return ret,frame

    def AddInfoToFrame(self, frame, frameNumber):
      # Parameters to display video info
        height, width= frame.shape[0:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thicknessFont = 1
        scaleText = .7
        colorText = (255,255,255) # G,B,R
        spaceBtwLines = 10
        outsideMargin = 10
        numberOfLines = 4 # number of text lines
        # Based on the textSize, define the new frame size
        textSize = cv2.getTextSize(str(frameNumber), font, scaleText, thicknessFont)[0]
        frameHeight = (textSize[1]*numberOfLines)+( ((numberOfLines-1)*spaceBtwLines) + (2*outsideMargin) )
        framedImageHeight = height+frameHeight
        framedImage = np.zeros((framedImageHeight, width, 3), np.uint8) # VDAO videos have 3 channels
        # Define positions of the texts to appear (bottom-left is the reference)
        originText1 = (outsideMargin, height+outsideMargin+textSize[1])
        originText2 = (outsideMargin, height+outsideMargin+(textSize[1]*2)+spaceBtwLines)
        originText3 = (outsideMargin, height+outsideMargin+(textSize[1]*3)+(spaceBtwLines*2))
        originText4 = (outsideMargin, height+outsideMargin+(textSize[1]*4)+(spaceBtwLines*3))
        # Define texts
        firstLine = "VDAO: Video Database of Abandoned Objects"
        secondLine = "File: " + self.videoInfo.getFileName()
        thirdLine = "Frame rate: "+ self.videoInfo.getFrameRate()
        fourthLine = "Frame: %d/"+str(self.videoInfo._numberOfFrames)
        # Define and resize logo to fit on the screen
        logo = cv2.imread(os.path.dirname(os.path.abspath(__file__))+'/images/logo.png')
        hlogo,wlogo = logo.shape[0], logo.shape[1]
        logo = cv2.resize(logo, (0,0), fx=float((frameHeight-(2*outsideMargin)))/hlogo, fy=float((frameHeight-(2*outsideMargin)))/hlogo, interpolation = cv2.INTER_CUBIC) 
        hlogo,wlogo = logo.shape[0], logo.shape[1]
        logoPosition = (height+int(frameHeight/2)-int(hlogo/2), height+int(frameHeight/2)-int(hlogo/2)+hlogo, width-outsideMargin-wlogo, width-outsideMargin)
        # VDAO videos have 3 channels
        framedImage = np.zeros((framedImageHeight, width, 3), np.uint8)
        # framedImage = np.zeros((width, framedImageHeight, 3), np.uint8)
        framedImage[0:height, 0:width, : ] = frame
        # Add logo
        framedImage[logoPosition[0]:logoPosition[1], logoPosition[2]:logoPosition[3], : ] = logo
        # Add text into
        cv2.putText(framedImage, firstLine, org=originText1, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
        cv2.putText(framedImage, secondLine, org=originText2, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
        cv2.putText(framedImage, thirdLine, org=originText3, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
        cv2.putText(framedImage, fourthLine%frameNumber, org=originText4, fontFace=font, color=colorText, thickness=thicknessFont, fontScale=scaleText)
        # Return frame with information
        return framedImage
    
    def SkipAndSaveFrames(self, startingFrame, endingFrame, framesToSkip, outputFolder, showInfo=False):
        for i in range(startingFrame,endingFrame,framesToSkip):
            # Get the ith frame
            res,frame = self.GetFrame(i, showInfo)
            # Check if frame was successfully retrieved and save it
            if res: 
                foldeAndFile = outputFolder+'/frame%d.jpg'%i
                cv2.imwrite(foldeAndFile, frame)
                print("File sucessfully saved: %s" % foldeAndFile)
            else:
                print("Error opening the frame %d" % i)



    # def CropROI(self, outputPath):
