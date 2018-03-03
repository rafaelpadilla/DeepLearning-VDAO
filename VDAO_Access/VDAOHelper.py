# -*- coding: utf-8 -*-
from enum import Enum
class VideoType(Enum):
    """
	Class representing enums that identify a VDAO video:
    * Reference for videos containing only background (no lost objects)"
    * WithObjects for videos that contain lost objects

        Developed by: Rafael Padilla
        SMT - Signal Multimedia and Telecommunications Lab
        COPPE - Universidade Federal do Rio de Janeiro
        Last modification: Dec 9th 2017
    """
    Reference = 1
    WithObjects = 2

import subprocess
import shlex
import json
import pprint
import os

class VDAOInfo:
    """
	VDAOInfo brings all important information about a video from VDAO database.

        Developed by: Rafael Padilla
        COPPE
        Universidade Federal do Rio de Janeiro
        Last modification: Dec 9th 2017
    """
    def __init__(self, video_file):
        self._filePath=video_file
        # Inicializa variáveis
        self._idxVideoInfo=None
        self._idxAudioInfo=None
        self._idxSubtitleInfo=None
        self._fileName=None
        self._format=None
        self._formatLong=None
        self._size=None #in bytes
        self._codec=None
        self._codecLong=None
        self._width=None
        self._height=None
        self._widthHeight=None
        self._sampleAspectRatio=None
        self._displayAspectRatio=None
        self._pixelFormat=None
        self._frameRate=None
        self._framesPerSecond=None
        self._durationTS=None
        self._duration=None
        self._durationReal=None
        self._bitRate=None
        self._numberOfFrames=None
        self._createdOn=None
        self._enconder=None
        # self.streams=[]
        # self.video=[]
        # self.audio=[]
        try:
            with open(os.devnull, 'w') as tempf:
                subprocess.check_call(["ffprobe","-h"],stdout=tempf,stderr=tempf)
        except:
            raise IOError('ffprobe not found.')
        if os.path.isfile(video_file):
            cmd = "ffprobe -v error -print_format json -show_streams -show_format"
            args = shlex.split(cmd)
            args.append(video_file)
            # Running ffprobe process and loads it in a json structure
            ffoutput = subprocess.check_output(args).decode('utf-8')
            ffoutput = json.loads(ffoutput)
            # Check available information on the file
            for i in range(len(ffoutput['streams'])):
                if ffoutput['streams'][i]['codec_type'] == 'video':
                    self._idxVideoInfo = i
                elif ffoutput['streams'][i]['codec_type'] == 'audio':
                    self._idxAudioInfo = i
                elif ffoutput['streams'][i]['codec_type'] == 'subtitle':
                    self._idxSubtitleInfo = i
            # Set properties related to the file itself
            self._fileName = ffoutput['format']['filename']
            self._fileName = self._fileName[self._fileName.rfind('/')+1:]
            self._format = ffoutput['format']['format_name']
            self._formatLong = ffoutput['format']['format_long_name']
            self._size = ffoutput['format']['size'] 
            if 'creation_time' in ffoutput['format']['tags']:
                self._createdOn = ffoutput['format']['tags']['creation_time']
            self._encoder = ffoutput['format']['tags']['encoder']
            # Set properties related to the video
            if self.isVideo():
                self._codec = ffoutput['streams'][self._idxVideoInfo]['codec_name'] 
                self._codecLong = ffoutput['streams'][self._idxVideoInfo]['codec_long_name'] 
                self._width = ffoutput['streams'][self._idxVideoInfo]['width'] 
                self._height = ffoutput['streams'][self._idxVideoInfo]['height'] 
                self._widthHeight = [self._width,self._height]
                self._sampleAspectRatio = ffoutput['streams'][self._idxVideoInfo]['sample_aspect_ratio'] 
                self._displayAspectRatio = ffoutput['streams'][self._idxVideoInfo]['display_aspect_ratio'] 
                self._pixelFormat = ffoutput['streams'][self._idxVideoInfo]['pix_fmt'] 
                self._frameRate = ffoutput['streams'][self._idxVideoInfo]['r_frame_rate'] 
                self._framesPerSecond = int(self._frameRate[:self._frameRate.index('/')])
                self._durationTS = ffoutput['streams'][self._idxVideoInfo]['duration_ts'] 
                self._duration = ffoutput['streams'][self._idxVideoInfo]['duration'] 
                # self._durationReal = ffoutput['streams'][self._idxVideoInfo]['duration'] 
                self._bitRate = ffoutput['streams'][self._idxVideoInfo]['bit_rate'] 
                self._numberOfFrames = ffoutput['streams'][self._idxVideoInfo]['nb_frames']       
        else:
            raise IOError('This is not a valid media file '+video_file)

    def isVideo(self):
        """Returns true if the file is a valid video extension"""
        val = False
        if self._idxVideoInfo is not None:
            val = True
        return val

    def hasAudio(self):
        """Returns true if the file provides audio information"""
        val = False
        if self._idxAudioInfo is not None:
            val = True
        return val

    def hasSubtitles(self):
        """Returns true if the file makes subtitle data available"""
        val = False
        if self._idxSubtitleInfo is not None:
            val = True
        return val

    def getFilePath(self):
        """Gets full file path"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._filePath
        return val

    def getFileName(self):
        """Gets the name of the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._fileName
        return val

    def getFormat(self):
        """Gets format of the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._format
        return val

    def getFormatLong(self):
        """Gets full format description"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._formatLong
        return val

    def getSize(self):
        """Gets the size of the file in bytes"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._size
        return val

    def getCreationDate(self):
        """Gets the creation date and time"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._createdOn
        return val

    def getEnconderType(self):
        """Gets the encoder used to generate the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._encoder
        return val

    def getCodecType(self):
        """Gets the codec for the file"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._codec
        return val

    def getCodecLongType(self):
        """Gets the full description of the codec"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._codecLong
        return val

    def getWidth(self):
        """Gets the width (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._width
        return val

    def getHeight(self):
        """Gets the height (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._height
        return val

    def getWidthHeight(self):
        """Gets the width and height (in pixels) of the frames"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._widthHeight
        return val

    def getSampleAspectRatio(self):
        """Gets width by height ratio of the pixels with respect to the original source"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._sampleAspectRatio
        return val

    def getDisplayAspectRatio(self):
        """Gets width by height ratio of the data as it is supposed to be displayed"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._displayAspectRatio
        return val

    def getPixelFormat(self):
        """Gets the raw representation of the pixel.
           For reference see: http://blog.arrozcru.org/?p=234"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._pixelFormat
        return val

    def getFrameRateFloat(self):
        """Gets number of frames that are displayed per second in float format"""
        val = self.getFrameRate()
        if val is not None:
            idx = val.find('/')
            if idx == -1:
                return None
            num = float(val[:idx])
            den = float(val[idx+1:])            
            return num/den
        return val

    def getFrameRate(self):
        """Gets number of frames that are displayed per second in the format X/1"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._frameRate
        return val

    def getFramesPerSecond(self): #WRONG!
        return None # Make it useless
        """Gets number of frames that are displayed per second ????? TO REVIEW!"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._framesPerSecond
        return val

    def getDurationTs(self):
        """Gets the duration whole video in frames ?????"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._durationTS
        return val

    def getRealDuration(self):
        """Gets the full duration of the video in seconds"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._durationReal
        return val

    def getBitRate(self):
        """Gets the number of bits used to represent each second of the video"""
        val = None
        if self._idxVideoInfo is not None:
            val = self._bitRate
        return val

    def getNumberOfFrames(self):
        """Gets the number of frames of the whole video ????"""
        val = None
        if self._idxVideoInfo is not None:
            val = int(self._numberOfFrames)
        return val

    def printAllInformation(self):
        print('\n*************************************')
        print('********** VDAO video info **********')
        print(' ')
        print('File path: ' + str(self._filePath))
        print('File name: ' + str(self._fileName))
        print('File extension: ' + str(self._format) + ' (' +str(self._formatLong)+')')
        print('Created on: ' + str(self._createdOn))
        print('Encoder: ' + str(self._encoder))
        print('File size: ' + str(self._size))
        print('Codec: ' + str(self._codec) + ' (' +str(self._codecLong)+')')
        print('Width: ' + str(self._width))
        print('Height: ' + str(self._height))
        print('Width x Height: ' + str(self._widthHeight))
        print('Sample aspect ratio: ' + str(self._sampleAspectRatio))
        print('Display aspect ratio: ' + str(self._displayAspectRatio))
        print('Pixel format: ' + str(self._pixelFormat))
        print('Frame rate: ' + str(self._frameRate))
        print('Duration ts: ' + str(self._durationTS))
        print('Duration: ' + str(self._duration))
        # print('Real douration: ' + str(self._real))
        print('Bit rate: ' + str(self._bitRate))
        print('Number of frames: ' + str(self._numberOfFrames))
        print(' ')
        print('*************************************')

class VDAOAnnotatedFile:
     def __init__(self, annotated_file):
         self.annotated_file = annotated_file
         
