# -*- coding: utf-8 -*-
import os
import sys
import time

import cv2
import imageio
import numpy as np

import utils

# To get video information
if __name__ == '__main__':
    from .Annotation import Annotation
    from .utils import splitPathFile
    from .VDAOHelper import ImageExtension, VDAOInfo, VideoType
    from .YoloTrainingHelper import YOLOHelper
else:
    from Annotation import Annotation
    from utils import splitPathFile
    from VDAOHelper import ImageExtension, VDAOInfo, VideoType
    from YoloTrainingHelper import YOLOHelper


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
        if videoType is None:
            _, fileName = splitPathFile(videoPath)
            if fileName.startswith('ref-'):
                self.videoType = VideoType.Reference
            elif fileName.startswith('obj-'):
                self.videoType = VideoType.WithObjects
        else:
            self.videoType = VideoType

        self.videoPath = videoPath
        self.videoInfo = VDAOInfo(self.videoPath)

        # self._annotation = annotationFilePath
        self._annotation = Annotation(annotationFilePath=annotationFilePath,
                                      totalFrames=self.videoInfo.getNumberOfFrames())

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

    def GetVideoType(self):
        return self.videoType

    @staticmethod
    def PlayFrameToFrame(listImages, dirImages, showBoundingBoxes=False):
        i = 0
        while i < len(listImages):
            if showBoundingBoxes:
                imagem = YOLOHelper.get_image_with_bb(listImages[i], dirImages)
                cv2.putText(imagem,
                            listImages[i],
                            org=(30, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=(255, 255, 255),
                            thickness=1,
                            fontScale=0.7)
            else:
                imagem = cv2.imread(os.path.join(dirImages, listImages[i]))
            cv2.imshow('Frame', imagem)
            wkey = cv2.waitKey(0)
            key = chr(wkey % 256)
            if key == 'a':
                i = i - 1
            elif key == 's':
                i = i + 1
            elif key == 'q':
                cv2.destroyAllWindows()
                return

    @staticmethod
    def GenerateVideosFromImages(listImages, outputFilePath, quality=6, fps=24):
        # Using imageio to generate output video
        writer = imageio.get_writer(outputFilePath, fps=fps, quality=quality, codec='libx264')
        for i in listImages:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)
        writer.close()

    @staticmethod
    def GenerateNewVideoFromFramesList(videoSource, listFrames, outputFilePath, quality=6, fps=24):
        vdao = VDAOVideo(videoSource)
        # Using imageio to generate output video
        writer = imageio.get_writer(outputFilePath, fps=fps, quality=quality, codec='libx264')
        for i in listFrames:
            frameInfo = vdao.GetFrame(i)
            if frameInfo[0] is False:
                raise Exception("Error reading frame %d" % i)
            image = frameInfo[1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            writer.append_data(image)
        writer.close()

    def PlayVideo(self, showInfo=True, showBoundingBoxes=False, frameCallback=None):
        # Parse annotations
        if showBoundingBoxes and self._annotation.parsed == False:
            # if somehow there was an error while parsing, do not showBoundingBoxes
            showBoundingBoxes = self.ParseAnnotation()

        cap = cv2.VideoCapture(self.videoPath)
        fps = self.videoInfo.getFrameRateFloat()  #or cap.get(cv2.CAP_PROP_FPS)
        waitFraction = int(
            770 / fps)  #Ajusta-se o fator 770 para tentar fazer o vídeo tocar no fps original
        # waitFraction = int(1000/fps) #Ajusta-se o fator 770 para tentar fazer o vídeo tocar no fps original
        # Parameters to display video info
        width, height = self.videoInfo.getWidthHeight()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thicknessFont = 1
        scaleText = .7
        colorText = (255, 255, 255)  # G,B,R
        spaceBtwLines = 10
        outsideMargin = 10
        numberOfLines = 4  # number of text lines
        # Based on the textSize, define the new frame size
        textSize = cv2.getTextSize(str(self.videoInfo.getNumberOfFrames()), font, scaleText,
                                   thicknessFont)[0]
        frameHeight = (textSize[1] * numberOfLines) + (((numberOfLines - 1) * spaceBtwLines) +
                                                       (2 * outsideMargin))
        framedImageHeight = height + frameHeight
        framedImage = np.zeros((framedImageHeight, width, 3),
                               np.uint8)  # VDAO videos have 3 channels
        # Define positions of the texts to appear (bottom-left is the reference)
        originText1 = (outsideMargin, height + outsideMargin + textSize[1])
        originText2 = (outsideMargin, height + outsideMargin + (textSize[1] * 2) + spaceBtwLines)
        originText3 = (outsideMargin,
                       height + outsideMargin + (textSize[1] * 3) + (spaceBtwLines * 2))
        originText4 = (outsideMargin,
                       height + outsideMargin + (textSize[1] * 4) + (spaceBtwLines * 3))
        # Define texts
        firstLine = "VDAO: Video Database of Abandoned Objects"
        secondLine = "File: " + self.videoInfo.getFileName()
        thirdLine = "Frame rate: " + self.videoInfo.getFrameRate()
        fourthLine = "Frame: %d/" + str(self.videoInfo._numberOfFrames)
        # Define and resize logo to fit on the screen
        logo = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/images/logo.png')
        hlogo, wlogo = logo.shape[0], logo.shape[1]
        logo = cv2.resize(logo, (0, 0),
                          fx=(frameHeight - (2 * outsideMargin)) / hlogo,
                          fy=(frameHeight - (2 * outsideMargin)) / hlogo,
                          interpolation=cv2.INTER_CUBIC)
        hlogo, wlogo = logo.shape[0], logo.shape[1]
        logoPosition = (height + int(frameHeight / 2) - int(hlogo / 2),
                        height + int(frameHeight / 2) - int(hlogo / 2) + hlogo,
                        width - outsideMargin - wlogo, width - outsideMargin)

        # Start reading -> OpenCV counts frames from 0 to nrFrames-1. We will count from 1 to nframes
        ret, frame = cap.read()
        frameCount = 1
        ret = True
        # start_time = time.time()
        maxTime = 0

        while (ret == True):
            start_time = time.time()
            # deltaTime = time.time() - start_time
            # if deltaTime > maxTime and frameCount > 50:
            #     maxTime = deltaTime
            #     print('Frame: %d (%f)' % (frameCount, maxTime*10))
            # start_time = time.time()

            if showInfo:
                # VDAO videos have 3 channels
                framedImage = np.zeros((framedImageHeight, width, 3), np.uint8)
                framedImage[0:height, 0:width, :] = frame
                # Add logo
                framedImage[logoPosition[0]:logoPosition[1], logoPosition[2]:
                            logoPosition[3], :] = logo
                # Add text into
                cv2.putText(framedImage,
                            firstLine,
                            org=originText1,
                            fontFace=font,
                            color=colorText,
                            thickness=thicknessFont,
                            fontScale=scaleText)
                cv2.putText(framedImage,
                            secondLine,
                            org=originText2,
                            fontFace=font,
                            color=colorText,
                            thickness=thicknessFont,
                            fontScale=scaleText)
                cv2.putText(framedImage,
                            thirdLine,
                            org=originText3,
                            fontFace=font,
                            color=colorText,
                            thickness=thicknessFont,
                            fontScale=scaleText)
                cv2.putText(framedImage,
                            fourthLine % frameCount,
                            org=originText4,
                            fontFace=font,
                            color=colorText,
                            thickness=thicknessFont,
                            fontScale=scaleText)
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
                    framedImage = utils.add_bb_into_image(framedImage, fr[b][1], (0, 255, 0), 3,
                                                          fr[b][0])

            deltaTime = (time.time() - start_time) * 1000  # secs to ms
            waitMs = waitFraction - deltaTime
            # If there is a callback, return the frame
            if frameCallback != None:
                frameCallback(framedImage, waitMs)
            else:
                # Show framedImage
                cv2.imshow('VDAO', framedImage)
                cv2.waitKey(int(waitMs))  #in miliseconds

            # Read next frame
            ret, frame = cap.read()
            frameCount = frameCount + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release video resources
        cap.release()
        cv2.destroyAllWindows()

    def GetFrame(self, frameNumber, withInfo=False, raiseException=True):
        """Frame count starts from 1 to max frames -> self.infoVideo.getNumberOfFrames()"""

        maxNumberFrames = self.videoInfo.getNumberOfFrames()
        if maxNumberFrames == None:
            raise IOError('It was not possible to detect the number of frames in the file')

        # Check if frame exist within the video
        if frameNumber < 1 or frameNumber > int(maxNumberFrames):
            if raiseException == True:
                raise IOError('Frame number must be between 1 and %s. Required frame=%d.' %
                              (str(self.videoInfo.getNumberOfFrames()), frameNumber))
            else:
                print('Error: Frame number must be between 1 and %s. Required frame=%d.' %
                      (str(self.videoInfo.getNumberOfFrames()), frameNumber))
                return None, None, None

        cap = cv2.VideoCapture(self.videoPath)

        # We make frameNumber-1, because for this API, our frames go from 1 to max;
        # openCV frame count is 0-based
        # Reference: https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

        # Approach #1: Slower
        # fr = self.videoInfo.getFrameRateFloat()
        # frameTime = 1000 * (frameNumber-1)/fr
        # cap.set(cv2.CAP_PROP_POS_MSEC, frameTime)
        # Approach #2: Faster -> We immediately get to the frame we want
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber - 1)

        ret, frame = cap.read()
        cap.release()
        sizeImg = None
        if ret:
            sizeImg = frame.shape
            if withInfo:
                frame = self.AddInfoToFrame(frame, frameNumber)
        return ret, frame, sizeImg

    def GetFrames(self, framesNumbers=[], raiseException=True, flatten=False):
        """Frame count starts from 1 to max frames -> self.infoVideo.getNumberOfFrames()"""

        maxPossibleNumberFrames = self.videoInfo.getNumberOfFrames()
        if maxPossibleNumberFrames == None:
            raise IOError('It was not possible to detect the number of frames in the file')

        if framesNumbers == []:
            raise IOError('Pass a valid array of frames')

        maxRequiredFrames = max(framesNumbers)

        # Check if frame exist within the video
        if maxRequiredFrames < 1 or maxRequiredFrames > int(maxPossibleNumberFrames):
            if raiseException == True:
                raise IOError('Frame number must be between 1 and %s. Required frame=%d.' %
                              (str(self.videoInfo.getNumberOfFrames()), maxRequiredFrames))
            else:
                print('Error: Frame number must be between 1 and %s. Required frame=%d.' %
                      (str(self.videoInfo.getNumberOfFrames()), maxRequiredFrames))
                return None, None, None

        cap = cv2.VideoCapture(self.videoPath)
        # Get first frame to define the size of the returning array
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret == False:
            if raiseException == True:
                raise IOError('Error reading frame=0.')
            else:
                print('Error reading frame=0.')
                return None, None, None

        # Create output vector namely returning_array
        if flatten == True:
            # Based on the amount of pixels ([len(framesNumber),width*height*channels])
            total_pixels = np.prod(frame.shape)
            returning_array = np.zeros((len(framesNumbers), total_pixels), dtype=np.uint8)
        else:
            returning_array = np.zeros(
                (len(framesNumbers), frame.shape[0], frame.shape[1], frame.shape[2]),
                dtype=np.uint8)

        # We make frameNumber-1, because for this API, our frames go from 1 to max;
        # openCV frame count is 0-based
        # Reference: https://docs.opencv.org/3.0-beta/modules/videoio/doc/reading_and_writing_video.html

        # Approach #1: Slower
        # fr = self.videoInfo.getFrameRateFloat()
        # frameTime = 1000 * (frameNumber-1)/fr
        # cap.set(cv2.CAP_PROP_POS_MSEC, frameTime)
        # Approach #2: Faster -> We immediately get to the frame we want
        count = 0
        for i in framesNumbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i - 1)
            ret, frame = cap.read()
            if ret == False:
                if raiseException == True:
                    raise IOError('Error reading frame=%d.' % i)
                else:
                    print('Error reading frame=%d.' % i)
                    return None, None, None
            # Adding flattened frame
            if flatten:
                returning_array[count] = frame.flatten().astype(np.uint8)
            # Adding frame
            else:
                returning_array[count] = frame
            count += 1
        cap.release()
        return returning_array

    def AddInfoToFrame(self, frame, frameNumber):
        # Parameters to display video info
        height, width = frame.shape[0:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thicknessFont = 1
        scaleText = .7
        colorText = (255, 255, 255)  # G,B,R
        spaceBtwLines = 10
        outsideMargin = 10
        numberOfLines = 4  # number of text lines
        # Based on the textSize, define the new frame size
        textSize = cv2.getTextSize(str(frameNumber), font, scaleText, thicknessFont)[0]
        frameHeight = (textSize[1] * numberOfLines) + (((numberOfLines - 1) * spaceBtwLines) +
                                                       (2 * outsideMargin))
        framedImageHeight = height + frameHeight
        framedImage = np.zeros((framedImageHeight, width, 3),
                               np.uint8)  # VDAO videos have 3 channels
        # Define positions of the texts to appear (bottom-left is the reference)
        originText1 = (outsideMargin, height + outsideMargin + textSize[1])
        originText2 = (outsideMargin, height + outsideMargin + (textSize[1] * 2) + spaceBtwLines)
        originText3 = (outsideMargin,
                       height + outsideMargin + (textSize[1] * 3) + (spaceBtwLines * 2))
        originText4 = (outsideMargin,
                       height + outsideMargin + (textSize[1] * 4) + (spaceBtwLines * 3))
        # Define texts
        firstLine = "VDAO: Video Database of Abandoned Objects"
        secondLine = "File: " + self.videoInfo.getFileName()
        thirdLine = "Frame rate: " + self.videoInfo.getFrameRate()
        fourthLine = "Frame: %d/" + str(self.videoInfo._numberOfFrames)
        # Define and resize logo to fit on the screen
        logo = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '/images/logo.png')
        hlogo, wlogo = logo.shape[0], logo.shape[1]
        logo = cv2.resize(logo, (0, 0),
                          fx=float((frameHeight - (2 * outsideMargin))) / hlogo,
                          fy=float((frameHeight - (2 * outsideMargin))) / hlogo,
                          interpolation=cv2.INTER_CUBIC)
        hlogo, wlogo = logo.shape[0], logo.shape[1]
        logoPosition = (height + int(frameHeight / 2) - int(hlogo / 2),
                        height + int(frameHeight / 2) - int(hlogo / 2) + hlogo,
                        width - outsideMargin - wlogo, width - outsideMargin)
        # VDAO videos have 3 channels
        framedImage = np.zeros((framedImageHeight, width, 3), np.uint8)
        # framedImage = np.zeros((width, framedImageHeight, 3), np.uint8)
        framedImage[0:height, 0:width, :] = frame
        # Add logo
        framedImage[logoPosition[0]:logoPosition[1], logoPosition[2]:logoPosition[3], :] = logo
        # Add text into
        cv2.putText(framedImage,
                    firstLine,
                    org=originText1,
                    fontFace=font,
                    color=colorText,
                    thickness=thicknessFont,
                    fontScale=scaleText)
        cv2.putText(framedImage,
                    secondLine,
                    org=originText2,
                    fontFace=font,
                    color=colorText,
                    thickness=thicknessFont,
                    fontScale=scaleText)
        cv2.putText(framedImage,
                    thirdLine,
                    org=originText3,
                    fontFace=font,
                    color=colorText,
                    thickness=thicknessFont,
                    fontScale=scaleText)
        cv2.putText(framedImage,
                    fourthLine % frameNumber,
                    org=originText4,
                    fontFace=font,
                    color=colorText,
                    thickness=thicknessFont,
                    fontScale=scaleText)
        # Return frame with information
        return framedImage

    # For JPEG format, set the quality from 0 to 100 (the higher, the better)
    # For PNG, set the compression level from 0 to 9 (higher value means a smaller size and longer compression time)
    # For PPM, PGM, and PBM formats set the binary format
    def SkipAndSaveFrames(self,
                          startingFrame,
                          endingFrame,
                          framesToSkip,
                          outputFolder,
                          extension=ImageExtension.JPG,
                          jpegQuality=95,
                          compressionLevel=3,
                          binaryFormat=1,
                          filePrefix='frame_',
                          showInfo=False):

        for i in range(startingFrame, endingFrame + 1, framesToSkip):
            # Get the ith frame
            res, frame, _ = self.GetFrame(i, showInfo)
            # Check if frame was successfully retrieved and save it
            if res:
                # Save image based on the extension
                if extension == ImageExtension.JPG:
                    ext = "jpg"
                    # For JPEG, it can be a quality ( CV_IMWRITE_JPEG_QUALITY )
                    # from 0 to 100 (the higher is the better).
                    # Default value is 95.
                    folderAndFile = outputFolder + '/%s%d.%s' % (filePrefix, i, ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_JPEG_QUALITY, jpegQuality])
                elif extension == ImageExtension.PNG:
                    ext = "png"
                    # For PNG, it can be the compression level ( CV_IMWRITE_PNG_COMPRESSION )
                    # from 0 to 9. A higher value means a smaller size and longer compression time.
                    # Default value is 3.
                    folderAndFile = outputFolder + '/%s%d.%s' % (filePrefix, i, ext)
                    cv2.imwrite(folderAndFile, frame,
                                [cv2.IMWRITE_PNG_COMPRESSION, compressionLevel])

                # For PPM, PGM, or PBM, it can be a binary format flag ( CV_IMWRITE_PXM_BINARY ), 0 or 1.
                # Default value is 1.
                elif extension == ImageExtension.PPM:
                    ext = "ppm"
                    folderAndFile = outputFolder + '/%s%d.%s' % (filePrefix, i, ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                elif extension == ImageExtension.PGM:
                    ext = "pgm"
                    folderAndFile = outputFolder + '/%s%d.%s' % (filePrefix, i, ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                elif extension == ImageExtension.PBM:
                    ext = "pbm"
                    folderAndFile = outputFolder + '/%s%d.%s' % (filePrefix, i, ext)
                    cv2.imwrite(folderAndFile, frame, [cv2.IMWRITE_PXM_BINARY, binaryFormat])
                if os.path.isfile(folderAndFile):
                    print("File sucessfully saved: %s" % folderAndFile)
                else:
                    print("Error saving file saved: %s" % folderAndFile)
            else:
                print("Error opening the frame %d" % i)
