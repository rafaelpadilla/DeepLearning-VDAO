# -*- coding: utf-8 -*-
import cv2
import os
import sys
import numpy as np
import utils
import time
# To get video information
from VDAOHelper import VDAOInfo, VideoType, ImageExtension
from Annotation import Annotation

class VDAOVideo:
    """
	VDAOVideo class contains important information all important methods and tools to access VDAO videos.

        Developed by: Rafael Padilla
        SMT - Signal Multimedia and Telecommunications Lab
        COPPE - Universidade Federal do Rio de Janeiro
        Last modification: Dec 9th 2017
    """
    
    def __init__(self, videoPath, videoType=None, annotationFilePath=None):
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

        # self._annotation = annotationFilePath
        self._annotation = Annotation(annotationFilePath=annotationFilePath, totalFrames=self.videoInfo.getNumberOfFrames())

    def ParseAnnotation(self):
        return self._annotation._parseFile()

    def GetAnnotations(self):
        if self._annotation.parsed == False:
            self.ParseAnnotation()
        return self._annotation

    def SetAnnotation(self, newAnnotation):
        self._annotation = newAnnotation
        self._annotation.parsed = True
        self._annotation.error = False

    
    def PlayVideo(self, showInfo=True, showBoundingBoxes=False):
        # Parse annotations
        if showBoundingBoxes and self._annotation.parsed == False:
            # if somehow there was an error while parsing, do not showBoundingBoxes
            showBoundingBoxes = self.ParseAnnotation()

        cap = cv2.VideoCapture(self.videoPath)
        fps = self.videoInfo.getFrameRateFloat() #or cap.get(cv2.CAP_PROP_FPS)
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
        start_time = time.time()
        maxTime = 0
        while(ret == True):
            deltaTime = time.time() - start_time
            if deltaTime > maxTime and frameCount > 50:
                maxTime = deltaTime
                print('Frame: %d (%f)' % (frameCount, maxTime*10))
            start_time = time.time()
            
            if showInfo:
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
            
            # if there is annotation to show
            if showBoundingBoxes == True: 
                # Annotation object's listAnnotation has frames+1 positions
                # listAnnotation[0] is not taken into account by the VDAOVideo.PlayVideo() when needed to draw bb
                # But the VDAOVideo.PlayVideo() plays the first frame :p
                # listAnnotation's last element is the last frame of the video
                fr = self._annotation.listAnnotation[frameCount]
                for b in range(len(fr)):
                    # label = fr[b][0]
                    # box = fr[b][1]
                    # framedImage = utils.add_bb_into_image(framedImage,box, (0,255,0), 3, label)
                    framedImage = utils.add_bb_into_image(framedImage, fr[b][1], (0,255,0), 3, fr[b][0])

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
            raise IOError('Frame number must be between 1 and '+str(self.videoInfo.getNumberOfFrames()))

        cap = cv2.VideoCapture(self.videoPath)
        fr = self.videoInfo.getFrameRateFloat() 
        # we make frameNumber-1, because for this API, our frames go from 1 to max
        frameTime = 1000 * (frameNumber-1)/fr 
        cap.set(cv2.CAP_PROP_POS_MSEC, frameTime)

        ret,frame = cap.read()
        cap.release()
        sizeImg = None
        if ret:
            sizeImg = frame.shape
            if withInfo:
                frame = self.AddInfoToFrame(frame, frameNumber)
        return ret,frame, sizeImg

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
    
    def SkipAndSaveFrames(self, startingFrame, endingFrame, framesToSkip, outputFolder, \
        extension=ImageExtension.JPG, \
        # For JPEG format, set the quality from 0 to 100 (the higher, the better)
        jpegQuality=95, 
        # For PNG, set the compression level from 0 to 9 (higher value means a smaller 
        # size and longer compression time)
        compressionLevel=3, \
        # For PPM, PGM, and PBM formats set the binary format
        binaryFormat = 1, \
        filePrefix='frame_', showInfo=False):

        for i in range(startingFrame,endingFrame+1,framesToSkip):
            # Get the ith frame
            res,frame = self.GetFrame(i, showInfo)
            # Check if frame was successfully retrieved and save it
            if res: 
                # Save image based on the extension
                if extension == ImageExtension.JPG:
                    ext = "jpg"
                    # For JPEG, it can be a quality ( CV_IMWRITE_JPEG_QUALITY ) 
                    # from 0 to 100 (the higher is the better). 
                    # Default value is 95.
                    folderAndFile = outputFolder+'/%s%d.%s'%(filePrefix,i,ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_JPEG_QUALITY, jpegQuality])
                elif extension == ImageExtension.PNG:
                    ext = "png"
                    # For PNG, it can be the compression level ( CV_IMWRITE_PNG_COMPRESSION ) 
                    # from 0 to 9. A higher value means a smaller size and longer compression time. 
                    # Default value is 3.
                    folderAndFile = outputFolder+'/%s%d.%s'%(filePrefix,i,ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PNG_COMPRESSION, compressionLevel])

                # For PPM, PGM, or PBM, it can be a binary format flag ( CV_IMWRITE_PXM_BINARY ), 0 or 1. 
                # Default value is 1.
                elif extension == ImageExtension.PPM:
                    ext = "ppm"
                    folderAndFile = outputFolder+'/%s%d.%s'%(filePrefix,i,ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                elif extension == ImageExtension.PGM:
                    ext = "pgm"
                    folderAndFile = outputFolder+'/%s%d.%s'%(filePrefix,i,ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                elif extension == ImageExtension.PBM:
                    ext = "pbm"
                    folderAndFile = outputFolder+'/%s%d.%s'%(filePrefix,i,ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                if os.path.isfile(folderAndFile):
                    print("File sucessfully saved: %s" % folderAndFile)
                else:
                    print("Error saving file saved: %s" % folderAndFile)
            else:
                print("Error opening the frame %d" % i)