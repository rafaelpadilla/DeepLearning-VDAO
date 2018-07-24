import os
import sys
if sys.version_info[0] < 2: # add tkinker depending on the 
    import Tkinter as tk
    import Tkinter.font as tkFont
else:
    import tkinter as tk
    import tkinter.font as tkFont

class PanelDetails:
    
    def __init__(self, parent=None, descriptionsVideo=[], detailsVideos=[], descriptionsAnnotat=[], detailsAnnot=[],maxRow=2, eventAddVideoPlayer=None, eventRemoveVideoPlayer=None):
        # Set event if a video is added to player
        self.eventAddVideoPlayer = eventAddVideoPlayer
        self.eventRemoveVideoPlayer = eventRemoveVideoPlayer
        # Create UI structure
        self.pnlPrincipal = tk.PanedWindow(parent, orient=tk.VERTICAL)
        self.pnlPrincipal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pnlButtons = tk.PanedWindow(self.pnlPrincipal, orient=tk.HORIZONTAL)
        self.pnlButtons.pack(fill=tk.BOTH)
        self.pnlInfo = tk.PanedWindow(self.pnlPrincipal, orient=tk.HORIZONTAL) # Will contain both LabelFrames (video and annotation)
        self.pnlInfo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lfVideo = tk.LabelFrame(self.pnlInfo)
        self.lfAnnotation = tk.LabelFrame(self.pnlInfo)
        self.lfVideo.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lfAnnotation.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Title labels
        lblVideoInfo = tk.Label(self.lfVideo, text='Video Information', font =("TkDefaultFont", 10, "bold", "underline"), justify=tk.CENTER, anchor=tk.W)
        lblVideoInfo.pack()
        lblAnnotationInfo = tk.Label(self.lfAnnotation, text='Annotation Information', font =("TkDefaultFont", 10, "bold", "underline"), justify=tk.CENTER, anchor=tk.W)
        lblAnnotationInfo.pack()
        # Icons add and remove
        currentPath = os.path.dirname(os.path.realpath(__file__))
        self.imgAdd = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','add.png'))
        self.btnAdd = tk.Button(self.pnlButtons, width=16, height=16, image=self.imgAdd, state=tk.DISABLED, command=self.btnAdd_Clicked)
        self.btnAdd.pack(side=tk.LEFT, padx=2, pady=2)
        self.imgRemove = tk.PhotoImage(file=os.path.join(currentPath,'aux_images','remove.png'))
        self.btnRemove = tk.Button(self.pnlButtons, compound=tk.TOP, width=16, height=16, image=self.imgRemove, state=tk.DISABLED, command=self.btnRemove_Clicked)
        self.btnRemove.pack(side=tk.LEFT, padx=2, pady=2)
        # Data
        self.parent = parent
        self.labelsInformationVideos = {}
        self.labelsInformationAnnot = {}
        self.descriptionsVideo = descriptionsVideo
        self.detailsVideos = detailsVideos
        self.descriptionsAnnotat = descriptionsAnnotat
        self.detailsAnnot = detailsAnnot
        self.maxRow = maxRow
        self.CreateInfoStructure()

    def RemoveAllInformation(self):
        for l in self.labelsInformationVideos:
            self.labelsInformationVideos[l].set('')
        for l in self.labelsInformationAnnot:
            self.labelsInformationAnnot[l].set('')
        self.btnRemove['state'] = 'disabled'
        self.btnAdd['state'] = 'disabled'

    # Click on the button to add item to the player        
    def btnAdd_Clicked(self):
        self.eventAddVideoPlayer(self.detailsVideos, self.detailsAnnot)

    # Click on the button to remove item from the player
    def btnRemove_Clicked(self):
        self.eventRemoveVideoPlayer(self.detailsVideos, self.detailsAnnot)

    def CreateInfoStructure(self):
        for idx in range(len(self.descriptionsVideo)):
            pnlItemVideo = tk.PanedWindow(self.lfVideo, orient=tk.HORIZONTAL)
            # Information label
            lbl1 = tk.Label(pnlItemVideo, text='%s:'%self.descriptionsVideo[idx], justify=tk.LEFT)
            pnlItemVideo.add(lbl1)
            # Item without information
            textVar = tk.StringVar()
            lbl2 = tk.Label(pnlItemVideo, textvariable=textVar, justify=tk.LEFT, anchor=tk.W)
            pnlItemVideo.add(lbl2)
            self.labelsInformationVideos['%s_v' % self.descriptionsVideo[idx]] = textVar
            # Pack it
            pnlItemVideo.pack(anchor=tk.W, fill=tk.BOTH)
        for idx in range(len(self.descriptionsAnnotat)):
            pnlItemAnnot = tk.PanedWindow(self.lfAnnotation, orient=tk.HORIZONTAL)
            # Information label
            lbl1 = tk.Label(pnlItemAnnot, text='%s:'%self.descriptionsAnnotat[idx], justify=tk.LEFT)
            pnlItemAnnot.add(lbl1)
            # Item without information
            textVar = tk.StringVar()
            lbl2 = tk.Label(pnlItemAnnot, textvariable=textVar, justify=tk.LEFT, anchor=tk.W)
            pnlItemAnnot.add(lbl2)
            self.labelsInformationAnnot['%s_a' % self.descriptionsAnnotat[idx]] = textVar
            # Pack it
            pnlItemAnnot.pack(anchor=tk.W, fill=tk.BOTH)

    def AddDetails(self, detailsVideos, detailsAnnot):
        self.RemoveAllInformation()
        self.detailsVideos = detailsVideos
        self.detailsAnnot = detailsAnnot
        if len(detailsVideos) != 0:
            for k,v in self.labelsInformationVideos.items():
                if k == 'File dir_v':
                    self.labelsInformationVideos[k].set(detailsVideos[0])
                    self.btnAdd['state'] = 'normal'
                if k == 'File name_v':
                    self.labelsInformationVideos[k].set(detailsVideos[1])
                if k == 'File extension_v':
                    self.labelsInformationVideos[k].set(detailsVideos[2])
                if k == 'Created on_v':
                    self.labelsInformationVideos[k].set(detailsVideos[3])
                if k == 'Encoder_v':
                    self.labelsInformationVideos[k].set(detailsVideos[4])
                if k == 'File size_v':
                    self.labelsInformationVideos[k].set(detailsVideos[5])
                if k == 'Codec_v':
                    self.labelsInformationVideos[k].set(detailsVideos[6])
                if k == 'Width_v':
                    self.labelsInformationVideos[k].set(detailsVideos[7])
                if k == 'Height_v':
                    self.labelsInformationVideos[k].set(detailsVideos[8])
                if k == 'Sample aspect ratio_v':
                    self.labelsInformationVideos[k].set(detailsVideos[9])
                if k == 'Display aspect ratio_v':
                    self.labelsInformationVideos[k].set(detailsVideos[10])
                if k == 'Pixel format_v':
                    self.labelsInformationVideos[k].set(detailsVideos[11])
                if k == 'Frame rate_v':
                    self.labelsInformationVideos[k].set(detailsVideos[12])
                if k == 'Duration ts_v':
                    self.labelsInformationVideos[k].set(detailsVideos[13])
                if k == 'Duration_v':
                    self.labelsInformationVideos[k].set(detailsVideos[14])
                if k == 'Bit rate_v':
                    self.labelsInformationVideos[k].set(detailsVideos[15])
                if k == 'Number of frames_v':
                    self.labelsInformationVideos[k].set(detailsVideos[16])
        if len(detailsAnnot) != 0:
            for k,v in self.labelsInformationAnnot.items():
                if k == 'File dir_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[0])
                if k == 'File name_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[1])
                if k == 'Annotated frames_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[2])
                if k == 'Number of classes_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[3])
                if k == 'Classes_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[4])
                if k == 'First annotation at_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[5])
                if k == 'Last annotation at_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[6])
                if k == 'Min object_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[7])
                if k == 'Max object_a':
                    self.labelsInformationAnnot[k].set(detailsAnnot[8])
            
    def _GetMaxWidthDescription(self):
        maximum = 0
        for desc in self.descriptionsVideo:
            if maximum < tkFont.Font().measure(desc):
                maximum = tkFont.Font().measure(desc)
        return maximum