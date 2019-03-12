#coding=utf-8
import os
import sys
import threading

import numpy as np
from PIL import Image, ImageTk

import _init_paths
import cv2
import utils
from Annotation import Annotation
from InputWindow import InputWindow
from MyEnums import StatusPlayer

if sys.version_info[0] <= 2:  # add tkinker depending on the
    import Tkinter as tk
    import tkFont
    import ttk
else:
    import tkinter as tk
    import tkinter.font as tkFont
    import tkinter.ttk as ttk


class VideoPlayer:
    def ChangeNavButtonsStatus(self, enable):
        # Se o vídeo é processado, não permite botões de tocar
        if self.videoIsProcessed == True:
            return
        if enable:
            self.btnBackwardsBeg.config(state="normal")
            self.btnBackwards1.config(state="normal")
            self.btnBackwards5.config(state="normal")
            self.btnBackwards10.config(state="normal")
            self.btnForwardEnd.config(state="normal")
            self.btnForward1.config(state="normal")
            self.btnForward5.config(state="normal")
            self.btnForward10.config(state="normal")
            self.btnSelectFrame.config(state="normal")
        else:
            self.btnBackwardsBeg.config(state="disabled")
            self.btnBackwards1.config(state="disabled")
            self.btnBackwards5.config(state="disabled")
            self.btnBackwards10.config(state="disabled")
            self.btnForwardEnd.config(state="disabled")
            self.btnForward1.config(state="disabled")
            self.btnForward5.config(state="disabled")
            self.btnForward10.config(state="disabled")
            self.btnSelectFrame.config(state="disabled")

    def CheckBoxAnnotaion_Changed(self):
        self.showBoundingBox = not self.showBoundingBox

    def RenderFrame(self, cvimagem, skippingFrame):
        imagem = ImageTk.PhotoImage(image=Image.fromarray(cvimagem))
        # Render image
        self.lblImageVideo.configure(image=imagem)
        self.lblImageVideo.image = imagem
        # Trigger the callback
        if self.callback_FrameUpdated != None:
            self.callback_FrameUpdated(cvimagem, self.currentFrameNbr, skippingFrame)

    def ResizeAndRenderFrame(self, cvFrame, skippingFrame):
        # Resize image
        imagem = cv2.cvtColor(cvFrame, cv2.COLOR_BGR2RGB)
        if self.showBoundingBox == True:
            # Annotation object's listAnnotation has frames+1 positions
            # listAnnotation[0] is not taken into account by the VDAOVideo.PlayVideo() when needed to draw bb
            # But the VDAOVideo.PlayVideo() plays the first frame :p
            # listAnnotation's last element is the last frame of the video
            fr = self.annotations.listAnnotation[self.currentFrameNbr]
            for b in range(len(fr)):
                # label = fr[b][0]
                # box = fr[b][1]
                imagem = utils.add_bb_into_image(imagem, fr[b][1], (0, 255, 0), 3, fr[b][0])
        scale_percent = 0.29  # percent of original size
        width = int(imagem.shape[1] * scale_percent)
        height = int(imagem.shape[0] * scale_percent)
        dim = (width, height)
        cvimagem = cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)
        self.RenderFrame(cvimagem, skippingFrame)

    def PlayOn(self):
        while self.cvVideo.isOpened():
            # If pause event is set to True
            if self.eventPause.isSet():
                self.eventPause.clear(
                )  # Set it to False, so we can wait (if it is not cleared, it is True, and it wont be able to wait)
                self.eventPause.wait()  # wait unitl it is set again
                self.eventPause.clear()
            # Increment frame counter
            self.currentFrameNbr += 1
            # Read next frame
            ret, cvFrame = self.cvVideo.read()
            if ret == False:
                break
            # Resize the frame and render it to the screen
            # skippingFrame is False when it is played by using the button Play
            # skippingFrame is True when it is played 1x, 5x or 10x buttons
            self.ResizeAndRenderFrame(cvFrame, skippingFrame=False)
            # Update label with current frame number
            self.lblFrameNumber.configure(
                text="Frame: %d/%d" % (self.currentFrameNbr, self.totalFrames))
        # Release video resource
        self.cvVideo.release()
        self.statusPlayer = StatusPlayer.NOT_STARTED

    def btnPlayPause_Clicked(self):
        if self.callBack_EquateFrames != None:
            self.callBack_EquateFrames()
        # Define thread to play
        if not hasattr(self, 'threadPlayOn'):
            self.eventPause = threading.Event()
            self.threadPlayOn = threading.Thread(
                target=self.PlayOn, args=[])  # Play video from current frame on
            self.eventPause.set()
            self.threadPlayOn.start()
        # Depending on the action...
        if self.statusPlayer == StatusPlayer.NOT_STARTED:  # it hasnt started yet
            self.callBack_PlayPauseBtn_Clicked(
                False, StatusPlayer.PLAYING)  # disable buttons on Player and pass new action
            self.ChangeNavButtonsStatus(False)  # disable navigation buttons
            self.statusPlayer = StatusPlayer.PLAYING  # let's play it
            self.btnPlayPause.config(image=self.imgPause)  # image shows it can be paused
            # Define starting frames
            self.currentFrameNbr = 1
            self.cvVideo.set(cv2.CAP_PROP_POS_FRAMES, self.currentFrameNbr - 1)
            self.lblFrameNumber.configure(
                text="Frame: %d/%d" % (self.currentFrameNbr, self.totalFrames))
            # Play thread
            self.eventPause.set()  # set event to release it
        elif self.statusPlayer == StatusPlayer.PLAYING:  # it is playing
            self.callBack_PlayPauseBtn_Clicked(
                True, StatusPlayer.PAUSED)  # enable buttons on Player and pass new action
            self.ChangeNavButtonsStatus(True)  # enable navigation buttons
            self.statusPlayer = StatusPlayer.PAUSED  # let's pause it
            self.eventPause.set()  # set event to pause it
            self.btnPlayPause.config(image=self.imgPlay)  # image shows it can be played
        elif self.statusPlayer == StatusPlayer.PAUSED:  # it is paused
            self.callBack_PlayPauseBtn_Clicked(
                False, StatusPlayer.PLAYING)  # disable buttons on Player and pass new action
            self.ChangeNavButtonsStatus(False)  # disable navigation buttons
            self.statusPlayer = StatusPlayer.PLAYING  # let's play it
            self.btnPlayPause.config(image=self.imgPause)  # image shows it can be paused
            self.eventPause.set()  # set event again to release it

    def btnBackwardsBeg_Clicked(self):
        self.Rewind(None)

    def btnBackwards10_Clicked(self):
        self.Rewind(10)

    def btnBackwards5_Clicked(self):
        self.Rewind(5)

    def btnBackwards1_Clicked(self):
        self.Rewind(1)

    def btnForward1_Clicked(self):
        self.MoveForward(1)

    def btnForward5_Clicked(self):
        self.MoveForward(5)

    def btnForward10_Clicked(self):
        self.MoveForward(10)

    def btnForwardEnd_Clicked(self):
        self.MoveForward(None)

    def btnSelectFrame_Clicked(self):
        self.inputWindow = tk.Toplevel(self.parent)
        inputWindow = InputWindow(
            parent=self.inputWindow,
            eventClose=self.event_SetFrame,
            title="asd",
            minValue=1,
            maxValue=self.totalFrames)
        inputWindow.Center(self.parent)

    def event_SetFrame(self, frameNumber):
        if frameNumber != None:
            self.GoToFrame(frameNumber)

    def GoToFrame(self, frameNumber):
        if self.callBack_EquateFrames != None:
            self.callBack_EquateFrames()

        self.currentFrameNbr = frameNumber
        self.cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frameNumber - 1)
        self.lblFrameNumber.configure(
            text="Frame: %d/%d" % (self.currentFrameNbr, self.totalFrames))
        if self.cvVideo.isOpened():
            ret, frame = self.cvVideo.read()
            if ret:
                self.ResizeAndRenderFrame(frame, skippingFrame=True)

    def Rewind(self, amountFrames):
        if amountFrames == None:
            self.GoToFrame(1)
        else:
            self.GoToFrame(self.currentFrameNbr - amountFrames)

    def MoveForward(self, amountFrames):
        if amountFrames == None:
            self.statusPlayer = StatusPlayer.NOT_STARTED  # as we are navigating to the last frame
            self.GoToFrame(self.totalFrames)
        else:
            self.GoToFrame(self.currentFrameNbr + amountFrames)

    def UpdateVideoDetails(self, fileName, videoFilePath, annotationFilePath, currentFrameNbr):
        if self.videoIsProcessed == False:
            self.lblFileName.config(text="File: " + fileName)
        else:
            self.lblFileName.config(text=" ")
        self.videoFilePath = videoFilePath
        self.cvVideo = cv2.VideoCapture(videoFilePath)
        self.annotationFilePath = annotationFilePath
        self.totalFrames = int(self.cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.totalFrames > 0:
            self.loadedFrames = True
        self.currentFrameNbr = currentFrameNbr
        self.annotations = Annotation(self.annotationFilePath, self.totalFrames)
        hasAnnotations = self.annotations.IsValid()
        # Habilita ou desabilita checkboxe para mostrar bounding boxes caso tenha annotation file
        if hasAnnotations:
            self.chkAnnotations.configure(state='normal')
        else:
            self.chkAnnotations.configure(state='disabled')
        # Define o status do player
        if self.totalFrames == 0:  # Se vídeo não tem frames para tocar
            self.statusPlayer = StatusPlayer.FAILED
            self.btnPlayPause.config(state="disabled")
        else:  # Se possuir frames para tocar
            self.statusPlayer = StatusPlayer.NOT_STARTED
            self.btnPlayPause.config(state="normal")
            # Cria evento e mostra o primeiro frame
            self.eventPause = threading.Event()
            self.threadPlayOn = threading.Thread(
                target=self.PlayOn, args=[])  # Play video from current frame on
            self.eventPause.set()
            self.threadPlayOn.start()
            self.callBack_PlayPauseBtn_Clicked(
                True, StatusPlayer.NOT_STARTED)  # disable buttons on Player and pass new action
            self.ChangeNavButtonsStatus(True)  # enable navigation buttons
            # Define starting frames
            self.GoToFrame(1)

    def CreateEmptyFrame(self, width, height):
        emptyCVFrame = np.zeros((height, width, 3), np.uint8)
        self.ResizeAndRenderFrame(emptyCVFrame, True)

    def __init__(self,
                 parent,
                 titleVideo,
                 fileName,
                 videoFilePath,
                 annotationFilePath,
                 currentFrameNbr,
                 callback_FrameUpdated=None,
                 callBack_PlayPauseBtn_Clicked=None,
                 videoIsProcessed=False,
                 callBack_EquateFrames=None):
        # Define variáveis
        self.updatedFrames = False
        self.eventPause = None
        self.callback_FrameUpdated = None  # Later it will receive the argument value
        self.videoIsProcessed = videoIsProcessed
        self.callBack_PlayPauseBtn_Clicked = callBack_PlayPauseBtn_Clicked
        self.callBack_EquateFrames = callBack_EquateFrames  # Evento chamado quando algum botão de atualização de frame é acionado (não é chamado quando o video está tocando)
        self.parent = parent
        self.videoFilePath = videoFilePath
        self.annotationFilePath = annotationFilePath
        self.startingFrame = 0
        self.currentFrameNbr = 0
        self.cvVideo = cv2.VideoCapture(videoFilePath)
        self.totalFrames = int(self.cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.annotations = Annotation(self.annotationFilePath, self.totalFrames)
        hasAnnotations = self.annotations.IsValid()
        self.showBoundingBox = tk.BooleanVar()
        # Create UI structure
        self.pnlPrincipal = tk.PanedWindow(parent, orient=tk.VERTICAL)
        self.pnlPrincipal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Label Frame
        self.lfVideo = tk.LabelFrame(self.pnlPrincipal)
        self.lfVideo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Label with the title of the video
        lblTitle = tk.Label(
            self.lfVideo,
            text=titleVideo,
            font=("TkDefaultFont", 10, "bold", "underline"),
            justify=tk.CENTER,
            anchor=tk.W)
        lblTitle.pack()
        # Label with the file name
        pnlFileName = tk.PanedWindow(self.lfVideo, orient=tk.HORIZONTAL)
        pnlFileName.pack(anchor=tk.W, fill=tk.BOTH)
        if self.videoIsProcessed == False:
            self.lblFileName = tk.Label(
                pnlFileName, text="File: " + fileName, justify=tk.LEFT, anchor=tk.W)
        else:
            self.lblFileName = tk.Label(pnlFileName, text=" ", justify=tk.LEFT, anchor=tk.W)
        pnlFileName.add(self.lblFileName)
        # Label to add the video image
        pnlVideoImage = tk.PanedWindow(self.lfVideo)
        pnlVideoImage.pack()
        # LABEL
        # self.lblImageVideo = tk.Label(pnlVideoImage, justify=tk.CENTER, anchor=tk.CENTER, width=54, height=20)
        self.lblImageVideo = tk.Label(pnlVideoImage, justify=tk.CENTER, anchor=tk.CENTER)
        pnlVideoImage.add(self.lblImageVideo)
        self.currentFrameNbr = 0
        self.CreateEmptyFrame(width=1280, height=720)
        # Define callback saying frame was updated
        self.callback_FrameUpdated = callback_FrameUpdated
        # CANVAS
        # self.lblImageVideo = tk.Canvas(self.lfVideo, width=200, height=200)
        # self.lblImageVideo.pack(anchor=tk.W, fill=tk.BOTH)
        # self.lblImageVideo.pack()
        # Label to visualize the current frame number
        pnlFrameNumber = tk.PanedWindow(self.lfVideo, orient=tk.VERTICAL)
        pnlFrameNumber.pack()
        if self.videoIsProcessed == False:
            # Checkbox showing if annotation is possible
            self.chkAnnotations = tk.Checkbutton(
                pnlFrameNumber,
                text='show bounding boxes',
                variable=self.showBoundingBox,
                justify=tk.LEFT,
                anchor=tk.W,
                onvalue=1,
                offvalue=0,
                command=self.CheckBoxAnnotaion_Changed)
            pnlFrameNumber.add(self.chkAnnotations)
            self.showBoundingBox = False
            if not hasAnnotations:
                self.chkAnnotations.configure(state='disabled')
            self.lblFrameNumber = tk.Label(
                pnlFrameNumber,
                text="Frame: %d/%d" % (self.currentFrameNbr, self.totalFrames),
                justify=tk.LEFT,
                anchor=tk.W)
            pnlFrameNumber.add(self.lblFrameNumber)
            # Buttons
            pnlButtons = tk.PanedWindow(self.lfVideo)
            pnlButtons.pack()
            currentPath = os.path.dirname(os.path.realpath(__file__))
            # Load images
            self.imgBackwardsBeg = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'rewind_beg.png'))
            self.imgBackwards1 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'rewind_1.png'))
            self.imgBackwards5 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'rewind_5.png'))
            self.imgBackwards10 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'rewind_10.png'))
            self.imgPlay = tk.PhotoImage(file=os.path.join(currentPath, 'aux_images', 'play.png'))
            self.imgPause = tk.PhotoImage(file=os.path.join(currentPath, 'aux_images', 'pause.png'))
            self.imgForwardEnd = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'forward_end.png'))
            self.imgForward1 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'forward_1.png'))
            self.imgForward5 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'forward_5.png'))
            self.imgForward10 = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'forward_10.png'))
            self.imgSelectFrame = tk.PhotoImage(
                file=os.path.join(currentPath, 'aux_images', 'select_frame.png'))
            # # Create and add buttons
            self.btnBackwardsBeg = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgBackwardsBeg,
                state=tk.NORMAL,
                command=self.btnBackwardsBeg_Clicked)
            self.btnBackwardsBeg.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnBackwards10 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgBackwards10,
                state=tk.NORMAL,
                command=self.btnBackwards10_Clicked)
            self.btnBackwards10.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnBackwards5 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgBackwards5,
                state=tk.NORMAL,
                command=self.btnBackwards5_Clicked)
            self.btnBackwards5.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnBackwards1 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgBackwards1,
                state=tk.NORMAL,
                command=self.btnBackwards1_Clicked)
            self.btnBackwards1.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnPlayPause = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgPlay,
                state=tk.NORMAL,
                command=self.btnPlayPause_Clicked)
            self.btnPlayPause.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnForward1 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgForward1,
                state=tk.NORMAL,
                command=self.btnForward1_Clicked)
            self.btnForward1.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnForward5 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgForward5,
                state=tk.NORMAL,
                command=self.btnForward5_Clicked)
            self.btnForward5.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnForward10 = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgForward10,
                state=tk.NORMAL,
                command=self.btnForward10_Clicked)
            self.btnForward10.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnForwardEnd = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgForwardEnd,
                state=tk.NORMAL,
                command=self.btnForwardEnd_Clicked)
            self.btnForwardEnd.pack(side=tk.LEFT, padx=2, pady=2)
            self.btnSelectFrame = tk.Button(
                pnlButtons,
                width=24,
                height=24,
                image=self.imgSelectFrame,
                state=tk.NORMAL,
                command=self.btnSelectFrame_Clicked)
            self.btnSelectFrame.pack(side=tk.LEFT, padx=2, pady=2)
            # Só permite tocar o play (se vídeo tiver frames para tocar)
            self.ChangeNavButtonsStatus(False)
