import glob
import json
import os
import sys

from PIL import Image, ImageTk

import _init_paths
import utils
from Annotation import Annotation
from CheckBar import CheckBar
from ListBox import MyListBox
from PanelDetails import PanelDetails
from Player import Player
from Table import *
from VDAOVideo import VDAOVideo

if sys.version_info[0] < 2:  # add tkinker depending on the
    import Tkinter as tk
    import ttk
    import tkFont
else:
    import tkinter as tk
    import tkinter.ttk as ttk
    import tkinter.font as tkFont


class Main:
    def CommandSearchBtn(self):
        selectedTables = self.chbTables.GetOnlySelected()  #"table 1", "table 2" ...
        selectedObjects = self.chbObjects.GetOnlySelectedAndEnabled()  #"shoes", "dark-blue box" ...
        selectedVideosTypes = self.chbVideosTypes.GetOnlySelected(
        )  # "references" or "with objects"
        selectedIlluminations = self.chbIllumination.GetOnlySelected()  #"normal" or "extra"
        filteredVideos = self.filters.GetVideos(tables=selectedTables,
                                                objects=selectedObjects,
                                                videoTypes=selectedVideosTypes,
                                                illuminations=selectedIlluminations)
        items = []
        for vid in filteredVideos:
            video = "" if (vid.videoType == VideoType.Reference
                           or not hasattr(vid, 'video')) else '- Video %s' % vid.video
            position = "" if (vid.videoType == VideoType.Reference
                              or not hasattr(vid, 'position')) else vid.position
            version = "" if (vid.videoType == VideoType.Reference
                             or not hasattr(vid, 'version')) else vid.version
            position_version = "" if position == "" else "Pos: %s" % position
            position_version = position_version if version == "" else "%s Ver: %s" % (
                position_version, version)
            items.append([
                vid.sourceTable.name, vid.sourceTable.illumination, "with objects"
                if vid.videoType == VideoType.WithObjects else vid.videoType.name.lower(), vid.name,
                "reference" if vid.videoType == VideoType.Reference else
                ('%s %s' % (vid.object_class, video)),
                position_version.lstrip(), "" if
                (vid.videoType == VideoType.Reference or not hasattr(vid, 'part')) else 'Part: %s' %
                vid.part, "" if not hasattr(vid, 'url') else vid.url[vid.url.rfind('/') + 1:],
                "" if not hasattr(vid, 'url_annotation') else
                vid.url_annotation[vid.url_annotation.rfind('/') + 1:]
            ])
        self.mc_listbox.AddItems(items)
        self.lblCountItems['text'] = str(self.mc_listbox.GetTotalItems())

    def SelectItemFromListCallBack(self, itemDetails):
        # Get information from the selected video
        detailsSelectedVid = self.GetVideoDetails(itemDetails[7])  # 7th column is the video name
        if detailsSelectedVid != []:
            detailsSelectedAnnot = self.GetAnnotDetails(
                itemDetails[8], detailsSelectedVid[16]
            )  # 7th column is the annotation file and the 16th is the total nmbrframes
        else:
            detailsSelectedAnnot = self.GetAnnotDetails(
                itemDetails[8],
                None)  # 7th column is the annotation file and the 16th is the total nmbrframes
        self.pnlAllDetails.AddDetails(detailsSelectedVid, detailsSelectedAnnot)

    def WithObjectsChanged(self):
        if 'with objects' in self.chbVideosTypes.GetOnlySelected():  #if 'with objects' is checked
            self.chbObjects.EnableAllElements()  # Enable Elements from CheckBar of objects
        else:
            self.chbObjects.DisableAllElements()  # Disable Elements from CheckBar of objects

    def GetVideoDetails(self, videoUrl):
        details = []
        if videoUrl == '':
            return details
        # Get locally the physical file
        videoName = videoUrl[videoUrl.rfind('/') + 1:]
        for dirpath, _, filenames in os.walk(self.directory):
            if videoName in filenames:
                # Obtain the VDAOVideo object to retrieve video information
                myVideo = VDAOVideo(os.path.join(dirpath, videoUrl))
                fullFilePath = myVideo.videoInfo.getFilePath()
                details.append(fullFilePath[:fullFilePath.rfind('/') + 1])
                details.append(myVideo.videoInfo.getFileName())
                details.append(
                    str(myVideo.videoInfo.getFormat()) + ' (' +
                    str(myVideo.videoInfo.getFormatLong()) + ')')
                details.append(myVideo.videoInfo.getCreationDate())
                details.append(myVideo.videoInfo.getEnconderType())
                details.append('%s bytes' % myVideo.videoInfo.getSize())
                details.append(
                    str(myVideo.videoInfo.getCodecType()) + ' (' +
                    str(myVideo.videoInfo.getCodecLongType()) + ')')
                details.append(myVideo.videoInfo.getWidth())
                details.append(myVideo.videoInfo.getHeight())
                details.append(myVideo.videoInfo.getSampleAspectRatio())
                details.append(myVideo.videoInfo.getDisplayAspectRatio())
                details.append(myVideo.videoInfo.getPixelFormat())
                details.append(myVideo.videoInfo.getFrameRate())
                details.append(myVideo.videoInfo.getDurationTs())
                details.append(myVideo.videoInfo.getRealDuration())
                details.append(myVideo.videoInfo.getBitRate())
                details.append(myVideo.videoInfo.getNumberOfFrames())
        return details

    def GetAnnotDetails(self, annotUrl, totalFrames):
        details = []
        if annotUrl == '':
            return details
        # Get locally the physical file
        annotationName = annotUrl[annotUrl.rfind('/') + 1:]
        for dirpath, _, filenames in os.walk(self.directory):
            if annotationName in filenames:
                myAnnot = Annotation(os.path.join(dirpath, annotationName), totalFrames)
                classes = myAnnot.GetClassesObjects()
                nmbAnnotedFrames, minFrame, maxFrame, minObj, maxObj = myAnnot.GetNumberOfAnnotatedFrames(
                )
                details.append(dirpath)
                details.append(annotationName)
                details.append('%s (%.2f%%)' % (nmbAnnotedFrames, 100 * nmbAnnotedFrames /
                                                totalFrames))  #annotated frames and its percentage
                details.append(len(classes))  #number of classes
                strClasses = ''
                for a in classes:
                    strClasses = '%s, %s' % (strClasses, a)
                strClasses = strClasses[2:]
                details.append(strClasses)  #classes
                details.append('frame %s' % minFrame)  #first annotation
                details.append('frame %s' % maxFrame)  #last annotation
                # details.append('(x,y,r,b)=%s / Area: %s / Frame: %s'%(str(minObj[2]),minObj[0],minObj[1])) #min obj (area,frame,x,y,r,b)
                # details.append('(x,y,r,b)=%s / Area: %s / Frame: %s'%(str(maxObj[2]),maxObj[0],maxObj[1])) #max obj (area,frame,x,y,r,b)
                details.append('Area: %s / Frame: %s' %
                               (minObj[0], minObj[1]))  #min obj (area,frame,x,y,r,b)
                details.append('Area: %s / Frame: %s' %
                               (maxObj[0], maxObj[1]))  #max obj (area,frame,x,y,r,b)
        return details

    def on_closing(self):
        self.playerWindow.destroy()
        self.playerWindow = None

    def callback_AddVideoPlayer(self, videoPath, annotationPath):
        # If there the player wasnt created yet, let's create one
        if self.playerWindow == None:
            self.playerWindow = tk.Toplevel(self.root)
            # Event to close
            self.playerWindow.protocol("WM_DELETE_WINDOW", self.on_closing)
            # Create window with video 1
            self.player = Player(self.playerWindow)
        # Get video path and annotation path to create the video
        if annotationPath != []:
            annotationPath = os.path.join(annotationPath[0], annotationPath[1])
        else:
            annotationPath = ""
        if videoPath != []:
            videoPath = os.path.join(videoPath[0], videoPath[1])
        else:
            videoPath = ""
        # No video created or both videos were created
        if self.player.lastVideoCreated == 0 or self.player.lastVideoCreated == 2:
            # Let's create video 1
            self.player.AddVideo1(videoPath, annotationPath, currentFrameNbr=0)
            self.player.lastVideoCreated = 1
        # The created video was the video 1
        else:
            # Let's create video 2
            self.player.AddVideo2(videoPath, annotationPath, currentFrameNbr=0)
            self.player.lastVideoCreated = 2

    def callback_RemoveVideoPlayer(self, videoPath, annotationPath):
        raise IOError('Needs implementation')

    def __init__(self, jsonFilePath, directoryVideos):
        if not os.path.exists(jsonPath):
            raise IOError('JSON file %s does not exist' % jsonFilePath)
        self.directory = directoryVideos
        self.root = tk.Tk()
        self.root.title("VDAO")
        self.frame = tk.Frame()
        _, jsonFile = utils.splitPathFile(jsonFilePath)
        with open(jsonFilePath) as f:
            jsonData = json.load(f)
        currentPath = os.path.dirname(os.path.realpath(__file__))
        [sourcePackage, _tables, _videos] = SourcePackage.CreateSourcePackage(jsonData)
        self.filters = Filters(sourcePackage, _tables, _videos)
        # Get information for the filters
        tableNames = self.filters.GetAllTableNames()  # table names
        tableNames.sort()
        objectsClasses = self.filters.GetAllObjectsClasses()  # classes of objects
        objectsClasses.sort()
        if 'multiple objects' in objectsClasses:  # put 'multiple objects' in the end of the list
            objectsClasses.remove('multiple objects')
            objectsClasses = ['multiple objects'] + objectsClasses
        tableTypes = self.filters.GetAllTypes()  # tables with "single object" or "multiple objects"
        tableTypes.sort()
        illuminationTypes = self.filters.GetIlluminationTypes()
        illuminationTypes.sort()
        # Pegar iluminação
        # https://www.tutorialspoint.com/python/tk_panedwindow.htm
        # https://tkdocs.com/tutorial/complex.html#panedwindow
        # https://www.python-course.eu/tkinter_layout_management.php
        ###########################################################################
        # Adding UI
        ###########################################################################
        pnlMain = tk.PanedWindow(self.root, orient=tk.VERTICAL)  # Main panel
        pnlMain.pack(fill=tk.BOTH, expand=True)
        ### Filters
        pnlFilters = tk.PanedWindow(pnlMain, orient=tk.VERTICAL)
        pnlMain.add(pnlFilters)
        ## Table filters
        # Tables
        lfTables = tk.LabelFrame(pnlFilters, text='Tables', width=100, height=100)
        pnlFilters.add(lfTables)
        self.chbTables = CheckBar(lfTables, tableNames, maxCol=6)
        self.chbTables.AddElementAll('(All Tables)')
        # Illuminiation
        lfIllumination = tk.LabelFrame(pnlFilters, text='Illuminations', width=100, height=100)
        pnlFilters.add(lfIllumination)
        self.chbIllumination = CheckBar(lfIllumination, illuminationTypes, maxCol=6)
        self.chbIllumination.AddElementAll('(All Illuminations)')
        # References/Objects
        lfTypesVideos = tk.LabelFrame(pnlFilters, text='Videos', width=100, height=100)
        pnlFilters.add(lfTypesVideos)
        self.chbVideosTypes = CheckBar(lfTypesVideos, ["reference", "with objects"],
                                       commands=[None, self.WithObjectsChanged],
                                       maxCol=3)
        self.chbVideosTypes.AddElementAll("(All Videos)", command=self.WithObjectsChanged)
        # Objects
        lfObjects = tk.LabelFrame(pnlFilters, text='Objects', width=100, height=100)
        pnlFilters.add(lfObjects)
        self.chbObjects = CheckBar(lfObjects, objectsClasses, maxCol=5, initialState='disable')
        self.chbObjects.AddElementAll('(All Objects)')
        ## Button Search
        pnlButton = tk.PanedWindow(pnlMain, orient=tk.HORIZONTAL)
        pnlMain.add(pnlButton)
        btnSearch = tk.Button(pnlButton, text="Search", command=self.CommandSearchBtn)
        pnlButton.add(btnSearch)
        ### Listbox Panel
        pnlListBox = tk.PanedWindow(pnlMain, orient=tk.HORIZONTAL)
        pnlMain.add(pnlListBox)
        ## ListBox
        listBoxHeaders = [
            'Table',  #Table name
            # 'Description', #Description table + Part
            'Illum',  #Illumination
            'Type',  # Reference / with objects
            'Name',  #ex: "Object 03" or "Reference 01"
            'Class',  #Class of the object
            'Position/Version',  #Position and Version (Ex: "Black Backpack - Position 2 - V1" => Position: 2 Version: 1)
            'Part',
            'File',  #Ex: http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/obj-sing-amb-part01-video10.avi => "obj-sing-amb-part01-video10.avi"
            'Annotation'  #Annotation file
        ]
        self.mc_listbox = MyListBox(pnlListBox, listBoxHeaders, self.SelectItemFromListCallBack)
        ### Details of selected items of the List Box
        pnlDetailsLB = tk.PanedWindow(pnlMain, orient=tk.VERTICAL)
        pnlMain.add(pnlDetailsLB)
        # Count Items
        pnlCount = tk.PanedWindow(pnlDetailsLB, orient=tk.HORIZONTAL)
        pnlDetailsLB.add(pnlCount)
        lblTotalItems = tk.Label(pnlCount, text="Videos found:", justify=tk.LEFT)
        pnlCount.add(lblTotalItems, sticky=tk.W)
        self.lblCountItems = tk.Label(pnlCount,
                                      text=str(self.mc_listbox.GetTotalItems()),
                                      justify=tk.LEFT,
                                      font=("TkDefaultFont", 10, "bold"))
        pnlCount.add(self.lblCountItems, sticky=tk.W)
        # Label Frame details of the selected video
        lfSummary = tk.PanedWindow(pnlDetailsLB)
        pnlDetailsLB.add(lfSummary)
        detailsLabelsVideo = [
            'File dir', 'File name', 'File extension', 'Created on', 'Encoder', 'File size',
            'Codec', 'Width', 'Height', 'Sample aspect ratio', 'Display aspect ratio',
            'Pixel format', 'Frame rate', 'Duration ts', 'Duration', 'Bit rate', 'Number of frames'
        ]
        detailsLabelsAnnotation = [
            'File dir', 'File name', 'Annotated frames', 'Number of classes', 'Classes',
            'First annotation at', 'Last annotation at', 'Min object', 'Max object'
        ]
        self.pnlAllDetails = PanelDetails(lfSummary,
                                          detailsLabelsVideo, [],
                                          detailsLabelsAnnotation, [],
                                          eventAddVideoPlayer=self.callback_AddVideoPlayer,
                                          eventRemoveVideoPlayer=self.callback_RemoveVideoPlayer)
        # Footer
        logoPath = os.path.join(currentPath, '..', 'images', 'logo.png')
        logoImage = Image.open(logoPath)
        wpercent = .50  # reduce image in 50%
        basewidth = int(float(logoImage.size[0] * wpercent))
        hsize = int((float(logoImage.size[1]) * float(wpercent)))
        logoImage = logoImage.resize((basewidth, hsize), Image.ANTIALIAS)
        self.logoImage = ImageTk.PhotoImage(logoImage)  # reload image
        pnlFoot = tk.PanedWindow(pnlMain, orient=tk.HORIZONTAL)  # create horizontal panel
        pnlMain.add(pnlFoot)
        pnlLeft = tk.PanedWindow(pnlFoot)
        pnlFoot.add(pnlLeft)
        pnlRight = tk.PanedWindow(pnlFoot, orient=tk.VERTICAL)
        pnlFoot.add(pnlRight)
        lblLogo = tk.Label(self.root, image=self.logoImage)
        lblDescription = tk.Label(pnlRight,
                                  justify=tk.CENTER,
                                  padx=10,
                                  font=("Helvetica", 16, "bold"),
                                  text="Video Database of Abandoned Objects")
        lblReference = tk.Label(pnlRight,
                                justify=tk.CENTER,
                                padx=0,
                                font=("Helvetica", 10),
                                text="https://github.com/rafaelpadilla/DeepLearning-VDAO")
        pnlLeft.add(lblLogo)
        pnlRight.add(lblDescription)
        pnlRight.add(lblReference)
        # Create a player window
        self.playerWindow = None
        self.root.mainloop()


directoryVideos = '/media/storage/VDAO/vdao_object'
jsonPath = '/media/storage/repositorio/DeepLearning-VDAO/VDAO_Access/VDAO_files/vdao_videos.json'
main = Main(jsonPath, directoryVideos)
