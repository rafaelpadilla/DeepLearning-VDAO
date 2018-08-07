import sys
import os
import threading
import _init_paths
import utils
import numpy as np
from VDAOVideo import VDAOVideo
from VDAOHelper import VideoType
from VideoPlayer import VideoPlayer
from InputWindow import InputWindow
from MyEnums import StatusPlayer
from PIL import Image, ImageTk
import time
import cv2
if sys.version_info[0] < 2: # add tkinker depending on the 
    import Tkinter as tk
    import Tkinter.ttk as ttk
    import Tkinter.tkFont as tkFont
else:
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.font as tkFont

class Player:
        
    def btnPlayPause_Clicked(self):
        # Make sure it is showing the correct frame
        self.videoPlayer1.updatedFrames = self.videoPlayer2.updatedFrames

        # Define thread to play
        if not hasattr(self, 'threadPlayOn'):
            self.eventPause = threading.Event()
            self.threadPlayOn = threading.Thread(target=self.PlayOn, args=[]) # Play video from current frame on
            self.eventPause.set()
            # Frames could have changed before first play together
            self.videoPlayer1.cvVideo.set(cv2.CAP_PROP_POS_FRAMES, self.videoPlayer1.currentFrameNbr)
            self.videoPlayer2.cvVideo.set(cv2.CAP_PROP_POS_FRAMES, self.videoPlayer2.currentFrameNbr)
            self.threadPlayOn.start()
        if self.statusPlayer == StatusPlayer.NOT_STARTED: # it hasnt started yet
            self.ChangeNavButtonsStatus(False) # disable navigation buttons
            self.statusPlayer = StatusPlayer.PLAYING # let's play it
            # Update status to each video
            self.videoPlayer1.statusPlayer = StatusPlayer.PLAYING
            self.videoPlayer2.statusPlayer = StatusPlayer.PLAYING
            self.btnPlayPause.config(image=self.imgPause) # image shows it can be paused
            # Enable/Disable buttons
            self.videoPlayer1.ChangeNavButtonsStatus(False) #disable buttons on video1
            self.videoPlayer2.ChangeNavButtonsStatus(False) #disable buttons on video2
            self.videoPlayer1.btnPlayPause.config(state="disabled")
            self.videoPlayer2.btnPlayPause.config(state="disabled")
            self.videoPlayer1.btnPlayPause.config(image=self.imgPause)
            self.videoPlayer2.btnPlayPause.config(image=self.imgPause)
            self.videoPlayer1.lblFrameNumber.configure(text="Frame: %d/%d" % (self.videoPlayer1.currentFrameNbr,self.videoPlayer1.totalFrames))
            self.videoPlayer2.lblFrameNumber.configure(text="Frame: %d/%d" % (self.videoPlayer2.currentFrameNbr,self.videoPlayer2.totalFrames))
            # Define thread to play
            self.eventPause.set() # set event to release it
        elif self.statusPlayer == StatusPlayer.PLAYING: # it is playing
            # Enable/Disable buttons
            self.videoPlayer1.ChangeNavButtonsStatus(True) #enable buttons on video1
            self.videoPlayer2.ChangeNavButtonsStatus(True) #enable buttons on video2
            self.videoPlayer1.btnPlayPause.config(state="normal")
            self.videoPlayer2.btnPlayPause.config(state="normal")
            self.videoPlayer1.btnPlayPause.config(image=self.imgPlay)
            self.videoPlayer2.btnPlayPause.config(image=self.imgPlay)
            self.ChangeNavButtonsStatus(True) # enable navigation buttons
            self.statusPlayer = StatusPlayer.PAUSED # let's pause it
            # Update status to each video
            self.videoPlayer1.statusPlayer = StatusPlayer.PAUSED
            self.videoPlayer2.statusPlayer = StatusPlayer.PAUSED
            self.eventPause.set() # set event to pause it
            self.btnPlayPause.config(image=self.imgPlay) # image shows it can be played
        elif self.statusPlayer == StatusPlayer.PAUSED: # it is paused
             # Enable/Disable buttons
            self.videoPlayer1.ChangeNavButtonsStatus(False) #disable buttons on video1
            self.videoPlayer2.ChangeNavButtonsStatus(False) #disable buttons on video2
            self.videoPlayer1.btnPlayPause.config(state="disabled")
            self.videoPlayer2.btnPlayPause.config(state="disabled")
            self.videoPlayer1.btnPlayPause.config(image=self.imgPause)
            self.videoPlayer2.btnPlayPause.config(image=self.imgPause)
            self.ChangeNavButtonsStatus(False) # disable navigation buttons
            self.statusPlayer = StatusPlayer.PLAYING # let's play it 
            # Update status to each video
            self.videoPlayer1.statusPlayer = StatusPlayer.PLAYING
            self.videoPlayer2.statusPlayer = StatusPlayer.PLAYING
            self.btnPlayPause.config(image=self.imgPause) # image shows it can be paused
            self.eventPause.set() # set event again to release it
    
    def callback_PlayPause(self, enableButtons, newStatus):
        # if (self.loadedFrame1 == True and self.loadedFrame2 == True):
        if self.videoPlayer1.updatedFrames == True and self.videoPlayer2.updatedFrames == True:
            self.statusPlayer = newStatus
            self.ChangeNavButtonsStatus(enableButtons)
            if enableButtons:
                self.btnPlayPause.config(state="normal")
            else:
                self.btnPlayPause.config(state="disabled")

    def ChangeNavButtonsStatus(self, enable):
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

    def PlayOn(self):
        # Plays until both videos have frames to play
        while self.videoPlayer1.cvVideo.isOpened() and self.videoPlayer2.cvVideo.isOpened():
            # If pause event is set to True
            if self.eventPause.isSet():
                self.eventPause.clear() # Set it to False, so we can wait (if it is not cleared, it is True, and it wont be able to wait)
                self.eventPause.wait() # wait unitl it is set again
                self.eventPause.clear()
            ret1,frame1 = self.videoPlayer1.cvVideo.read()
            ret2,frame2 = self.videoPlayer2.cvVideo.read()
            if ret1 and ret2:
                self.videoPlayer1.currentFrameNbr += 1
                self.videoPlayer2.currentFrameNbr += 1
                self.videoPlayer1.ResizeAndRenderFrame(frame1, skippingFrame=False)
                self.videoPlayer2.ResizeAndRenderFrame(frame2, skippingFrame=False)
                self.videoPlayer1.lblFrameNumber.configure(text="Frame: %d/%d" % (self.videoPlayer1.currentFrameNbr,self.videoPlayer1.totalFrames))
                self.videoPlayer2.lblFrameNumber.configure(text="Frame: %d/%d" % (self.videoPlayer2.currentFrameNbr,self.videoPlayer2.totalFrames))
        
    def btnBackwardsBeg_Clicked(self):
        self.videoPlayer1.btnBackwardsBeg_Clicked()
        self.videoPlayer2.btnBackwardsBeg_Clicked()

    def btnBackwards10_Clicked(self):
        self.videoPlayer1.btnBackwards10_Clicked()
        self.videoPlayer2.btnBackwards10_Clicked()

    def btnBackwards5_Clicked(self):
        self.videoPlayer1.btnBackwards5_Clicked()
        self.videoPlayer2.btnBackwards5_Clicked()

    def btnBackwards1_Clicked(self):
        self.videoPlayer1.btnBackwards1_Clicked()
        self.videoPlayer2.btnBackwards1_Clicked()

    def btnForward1_Clicked(self):
        self.videoPlayer1.btnForward1_Clicked()
        self.videoPlayer2.btnForward1_Clicked()

    def btnForward5_Clicked(self):
        self.videoPlayer1.btnForward5_Clicked()
        self.videoPlayer2.btnForward5_Clicked()

    def btnForward10_Clicked(self):
        self.videoPlayer1.btnForward10_Clicked()
        self.videoPlayer2.btnForward10_Clicked()
    
    def btnForwardEnd_Clicked(self):
        self.videoPlayer1.btnForwardEnd_Clicked()
        self.videoPlayer2.btnForwardEnd_Clicked()

    def event_SetFrame(self, frameNumber):
        if frameNumber != None:
            self.videoPlayer1.GoToFrame(frameNumber)
            self.videoPlayer2.GoToFrame(frameNumber)

    def btnSelectFrame_Clicked(self):
        self.inputWindow = tk.Toplevel(self.root)
        inputWindow = InputWindow(parent=self.inputWindow, eventClose=self.event_SetFrame, title="", minValue=1, maxValue=np.min([self.videoPlayer1.totalFrames, self.videoPlayer2.totalFrames]))
        inputWindow.Center(self.inputWindow)

    def PlayDiff(self, skippingFrame):
        if self.videoPlayer1.updatedFrames == self.videoPlayer2.updatedFrames or skippingFrame:
            grayA = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(self.frame2, cv2.COLOR_BGR2GRAY)
            diff = cv2.subtract(grayA, grayB)
            self.videoPlayerDifference.RenderFrame(diff,skippingFrame=False)

    def callBack_EquateFrames(self):
        self.videoPlayer1.updatedFrames = self.videoPlayer2.updatedFrames = True

    def callback_PlayFrame1(self, cvImage, frameNumber, skippingFrame):
        self.frame1 = cvImage
        self.videoPlayer1.updatedFrames = not self.videoPlayer1.updatedFrames
        if self.videoPlayer1.currentFrameNbr != 0 and self.videoPlayer2.currentFrameNbr != 0:
            self.PlayDiff(skippingFrame)

    def callback_PlayFrame2(self, cvImage, frameNumber, skippingFrame):
        self.frame2 = cvImage
        self.videoPlayer2.updatedFrames = not self.videoPlayer2.updatedFrames
        if self.videoPlayer1.currentFrameNbr != 0 and self.videoPlayer2.currentFrameNbr != 0:
            self.PlayDiff(skippingFrame)

    def AddVideo1(self, videoFilePath, annotationFilePath, currentFrameNbr):
        _,fileName = utils.splitPathFile(videoFilePath)
        self.videoPlayer1.UpdateVideoDetails(fileName, videoFilePath, annotationFilePath, currentFrameNbr)

    def AddVideo2(self, videoFilePath, annotationFilePath, currentFrameNbr):
        _,fileName = utils.splitPathFile(videoFilePath)
        self.videoPlayer2.UpdateVideoDetails(fileName, videoFilePath, annotationFilePath, currentFrameNbr)
        # Depois de adicionar o segundo vídeo, habilita botão para tocar os dois vídeos
        self.btnPlayPause.config(state="normal")
        self.ChangeNavButtonsStatus(True)
        # Seta o status como NOT_STARTED para que o vídeo seja tocado
        self.statusPlayer = StatusPlayer.NOT_STARTED
        self.videoPlayer2.loadedFrames = True

    def __init__(self, parent, video1FilePath=None, annotation1FilePath=None, video2FilePath=None, annotation2FilePath=None):
        self.root = parent
        # Set status of the player
        self.statusPlayer = StatusPlayer.NOT_STARTED
        self.eventPause = threading.Event()
        # current path
        currentPath = os.path.dirname(os.path.realpath(__file__))
        self.root.title("VDAO - Videos visualization")
        self.pnlMain = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.pnlMain.pack(fill=tk.BOTH, expand=True)
        # Define video player 1 
        self.videoPlayer1 = VideoPlayer(parent=self.pnlMain, titleVideo='Video #1', fileName='', videoFilePath='', annotationFilePath='', currentFrameNbr=0, callback_FrameUpdated=self.callback_PlayFrame1, callBack_PlayPauseBtn_Clicked=self.callback_PlayPause, callBack_EquateFrames=self.callBack_EquateFrames)
        # Define video Difference
        self.videoPlayerDifference = VideoPlayer(self.pnlMain, 'Video Difference', "", "", None, 1, None, None, videoIsProcessed=True)
        # Define video player 2        
        self.videoPlayer2 = VideoPlayer(parent=self.pnlMain, titleVideo='Video #2', fileName='', videoFilePath='', annotationFilePath='', currentFrameNbr=0, callback_FrameUpdated=self.callback_PlayFrame2, callBack_PlayPauseBtn_Clicked=self.callback_PlayPause, callBack_EquateFrames=self.callBack_EquateFrames)
       # Buttons
        pnlButtons = tk.PanedWindow(self.root)
        pnlButtons.pack()
        # Load images
        self.imgBackwardsBeg = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','rewind_beg.png'))
        self.imgBackwards1 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','rewind_1.png'))
        self.imgBackwards5 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','rewind_5.png'))
        self.imgBackwards10 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','rewind_10.png'))
        self.imgPlay = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','play.png'))
        self.imgPause = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','pause.png'))
        self.imgForwardEnd = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','forward_end.png'))
        self.imgForward1 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','forward_1.png'))
        self.imgForward5 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','forward_5.png'))
        self.imgForward10 = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','forward_10.png'))
        self.imgSelectFrame = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','select_frame.png'))
        # Create and add buttons
        self.btnBackwardsBeg = tk.Button(pnlButtons, width=24, height=24, image=self.imgBackwardsBeg, state=tk.NORMAL, command=self.btnBackwardsBeg_Clicked)
        self.btnBackwardsBeg.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnBackwards10 = tk.Button(pnlButtons, width=24, height=24, image=self.imgBackwards10, state=tk.NORMAL, command=self.btnBackwards10_Clicked)
        self.btnBackwards10.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnBackwards5 = tk.Button(pnlButtons, width=24, height=24, image=self.imgBackwards5, state=tk.NORMAL, command=self.btnBackwards5_Clicked)
        self.btnBackwards5.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnBackwards1 = tk.Button(pnlButtons, width=24, height=24, image=self.imgBackwards1, state=tk.NORMAL, command=self.btnBackwards1_Clicked)
        self.btnBackwards1.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnPlayPause = tk.Button(pnlButtons, width=24, height=24, image=self.imgPlay, state=tk.NORMAL, command=self.btnPlayPause_Clicked)
        self.btnPlayPause.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnForward1 = tk.Button(pnlButtons, width=24, height=24, image=self.imgForward1, state=tk.NORMAL, command=self.btnForward1_Clicked)
        self.btnForward1.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnForward5 = tk.Button(pnlButtons, width=24, height=24, image=self.imgForward5, state=tk.NORMAL, command=self.btnForward5_Clicked)
        self.btnForward5.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnForward10 = tk.Button(pnlButtons, width=24, height=24, image=self.imgForward10, state=tk.NORMAL, command=self.btnForward10_Clicked)
        self.btnForward10.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnForwardEnd = tk.Button(pnlButtons, width=24, height=24, image=self.imgForwardEnd, state=tk.NORMAL, command=self.btnForwardEnd_Clicked)
        self.btnForwardEnd.pack(side=tk.LEFT, padx=2, pady=2)
        self.btnSelectFrame = tk.Button(pnlButtons, width=24, height=24, image=self.imgSelectFrame, state=tk.NORMAL, command=self.btnSelectFrame_Clicked)
        self.btnSelectFrame.pack(side=tk.LEFT, padx=2, pady=2)
        # Desabilita botoes
        self.ChangeNavButtonsStatus(False)
        # Se tiver não frames para tocar em ambos vídeos
        if self.videoPlayer1.totalFrames == 0 or self.videoPlayer2.totalFrames == 0: # Se vídeo não tem frames para tocar
            self.statusPlayer = StatusPlayer.FAILED
            # self.loadedFrame1 = self.loadedFrame2 = False
            self.videoPlayer1.loadedFrames = self.videoPlayer2.loadedFrames = False
            # Desabilita botão geral para tocar vídeos
            self.btnPlayPause.config(state="disabled")
        else: # Tem frames para tocar em ambos videos
            self.statusPlayer = StatusPlayer.NOT_STARTED 
            # self.loadedFrame1 = self.loadedFrame2 = True
            self.videoPlayer1.loadedFrames = self.videoPlayer2.loadedFrames = True
            # Habilita botão geral para tocar vídeos
            self.btnPlayPause.config(state="normal")
        # Desabilita botões de navegação entre para os 2 vídeos
        self.videoPlayer1.ChangeNavButtonsStatus(False)
        self.videoPlayer2.ChangeNavButtonsStatus(False)
        # Se videoPlayer1 não tiver frames para tocar, desabilita o play
        if self.videoPlayer1.totalFrames == 0:
            self.videoPlayer1.btnPlayPause.config(state="disabled")
        else: # habilita o play
            self.videoPlayer1.btnPlayPause.config(state="normal")
        # Se videoPlayer2 não tiver frames para tocar, desabilita o play
        if self.videoPlayer2.totalFrames == 0:
            self.videoPlayer2.btnPlayPause.config(state="disabled")
        else: # habilita o play
            self.videoPlayer2.btnPlayPause.config(state="normal")
        # identifies last video that was created (0: None, 1: First video, 2: Second video)
        self.lastVideoCreated = 0
        # Creates empty frames
        self.frame1 = self.frame2 = None

videosFolder = '/media/rafael/Databases/databases/VDAO/references/table_01/'
video1FilePath = os.path.join(videosFolder,'Table_01-Reference_01','_1_ref-sing-amb-part01-video01.avi')
# video1FilePath = os.path.join(videosFolder,'Table_01-Reference_01','_2_ref-sing-amb-part01-video01.avi')
# video1FilePath = os.path.join(videosFolder,'Table_01-Object_03','obj-sing-amb-part01-video03.avi')
video2FilePath = os.path.join(videosFolder,'Table_01-Object_01','align__1_obj-sing-amb-part01-video01.avi')
# video2FilePath = os.path.join(videosFolder,'Table_01-Object_03','_2_obj-sing-amb-part01-video03.avi')
v = VDAOVideo(video2FilePath)

root = tk.Tk()
player = Player(root, video1FilePath,None,video2FilePath,None)
player.AddVideo1(video1FilePath, None, currentFrameNbr=0)
player.AddVideo2(video2FilePath, None, currentFrameNbr=0)
root.mainloop()