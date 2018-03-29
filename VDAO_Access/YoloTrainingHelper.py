import numpy as np
import cv2
import utils
import os
import glob
from shutil import copyfile
import xml.etree.ElementTree as ET

class Detection:
    def __init__(self):
        self.classId = None
        self.confidence = None
        self.x_rel = None
        self.y_rel = None
        self.w_rel = None
        self.h_rel = None
        self.width_img = None
        self.height_img = None      

    def getRelativeBoundingBox(self):
        return (self.x_rel,self.y_rel,self.w_rel,self.h_rel)

    def getAbsoluteBoundingBox(self):
        return YOLOHelper.deconvert((self.width_img, self.height_img), (self.x_rel,self.y_rel,self.w_rel,self.h_rel))

    @staticmethod
    def compare(det1, det2):
        if det1.classId == det2.classId and \
           det1.confidence == det2.confidence and \
           det1.x_rel == det2.x_rel and \
           det1.y_rel == det2.y_rel and \
           det1.w_rel == det2.w_rel and \
           det1.h_rel == det2.h_rel and \
           det1.width_img == det2.width_img and \
           det1.height_img == det2.height_img:
           return True
        return False     
    
    @staticmethod
    def clone(detection):
        newDetection = Detection()
        newDetection.classId = detection.classId
        newDetection.confidence = detection.confidence
        newDetection.x_rel = detection.x_rel
        newDetection.y_rel = detection.y_rel
        newDetection.w_rel = detection.w_rel
        newDetection.h_rel = detection.h_rel
        newDetection.width_img = detection.width_img
        newDetection.height_img = detection.height_img
        return newDetection

class Detections:

    def __init__(self, txtFilePath=None, imgFilePath=None):
        self.detections = []
        if txtFilePath == None and imgFilePath == None:
            return

        # if no image path was given, get its width and height from the image with the same
        # name in the same path as the txtFilePath
        if imgFilePath == None:
            # given the .txt file, get its .jpg image corresponding file
            imgFilePath = txtFilePath.replace('_dets.txt','.jpg').replace('.txt', '.jpg')
            height = None
            width = None
        # given the imgFilePath, get images width and height
        if os.path.exists(imgFilePath):
            im = cv2.imread(imgFilePath)
            height, width, _ = im.shape
        else:
            raise ValueError('\nError! Not possible to get image width and height in order to read the %s file' % txtFilePath)

        f = open(txtFilePath)
        lines = f. readlines()
        f.close()
        lines = [n.replace('\n','') for n in lines]
        self.detections = []
        for l in lines:
            params = l.split(' ')
            det = Detection()
            # if detection has 6 parameters, we have: (id, confidence, x, y, w, h)
            if len(params) == 6:
                det.classId = int(params[0])
                det.confidence = float(params[1])
                det.x_rel = float(params[2])
                det.y_rel = float(params[3])
                det.w_rel = float(params[4])
                det.h_rel = float(params[5])
                det.width_img = int(width)
                det.height_img = int(height)
            # if detection has 5 parameters, we have: (id, x, y, w, h) => confidence is not informed
            elif len(params) == 5:
                det.classId = int(params[0])
                det.confidence = None
                det.x_rel = float(params[1])
                det.y_rel = float(params[2])
                det.w_rel = float(params[3])
                det.h_rel = float(params[4])
                det.width_img = int(width)
                det.height_img = int(height)
            else:
                raise ValueError('\nError! Line %s in the file %s does not have enough bounding box information' % (l, txtFilePath))
            self.detections.append(det)
    
    def getDetections(self):
        return self.detections

    def removeDetection(self, detection):
        for d in self.detections:
            if Detection.compare(d,detection):
                del self.detections[d]
                return
    
    def addDetection(self, detection):
        self.detections.append(detection)

    def count(self):
        return len(self.detections)

    def clone(self):
        newDetections = Detections()
        for d in self.detections:
            det = Detection.clone(d)
            newDetections.addDetection(det)
        return newDetections

    @staticmethod
    def evaluateDetections(gtDetection, evalDetection):
        #####################################################################
        # Evaluate Intersection over Union (IoU):
        #
        # - Only detected objects that are overlapped with the same class 
        #       groundtruth objects will be taken into account (e.g. a "cat"
        #       can only be compared to another "cat").
        # - Among multiple detections for a unique groundtruth bounding box,
        #       only the one with the highest IoU will be taken into consid-
        #       eration.
        # - As the highest IoU is taken into consideration, the bounding box
        #       pair that is considered a match (one from groundtruth and the 
        #       other is the detected one), is removed. This way these bound-
        #       ing boxes won't be considered in further checking for this 
        #       image. Then only maximum IoU will be added to IoU_sum for 
        #       the image.
        # - The IoU of an image pair is the average of its IoU.
        # - IoU of the image = IoU_sum / (True_positives + False_positives) which
        #       is the same as average(IoU_sum)
        #
        #####################################################################
        # Evaluate True Positive (TP) and False Positive (FP):
        #
        # - If detected "cat" isn't overlaped with "cat" (or overlaped with "dog"),
        #       then this is false_positive and its IoU = 0
        # - All non-maximum IoUs = 0 and they are false_positives.
        # - If a detected "cat" is somehow overlapped with a "cat", then it is 
        #       accounted as a True Positive and no other detection will be 
        #       considered for those bounding boxes.
        #####################################################################
        IoUs = []
        # Initiate True Positives and False Positive counts
        TP = 0
        FP = 0
        # for each evalDetection, find the best (lowest IOU) gtDetection
        # note: the eval detection must be the same class
        while evalDetection.count() > 0:
            for detEval in evalDetection.detections:
                bb = detEval.getAbsoluteBoundingBox()
                bestIoU = 0
                bestGT = None
                # find the bb with lowest IOU
                for detGT in gtDetection.detections:
                    
                    # detection must be the same class as the groundtruth
                    if detGT.classId == detEval.classId:
                        iou = YOLOHelper.iou(bb, detGT.getAbsoluteBoundingBox())

                        # # Show blank image with the bounding boxes
                        # img = np.zeros((detGT.height_img,detGT.width_img,3), np.uint8)
                        # aa = detEval.getAbsoluteBoundingBox()
                        # img = cv2.rectangle(img, (aa[0],aa[1]), (aa[2],aa[3]), (0,0,255), 4)
                        # for de in evalDetection.detections:
                        #     aaa = de.getAbsoluteBoundingBox()
                        #     img = cv2.rectangle(img, (aaa[0],aaa[1]), (aaa[2],aaa[3]), (0,0,255), 2)
                        # bbb = detGT.getAbsoluteBoundingBox()
                        # img = cv2.rectangle(img, (bbb[0],bbb[1]), (bbb[2],bbb[3]), (0,255,0), 4)
                        # for gt in gtDetection.detections:
                        #     bbb = gt.getAbsoluteBoundingBox()
                        #     img = cv2.rectangle(img, (bbb[0],bbb[1]), (bbb[2],bbb[3]), (0,255,0), 2)
                        # cv2.imshow("IoU",img)
                        # cv2.waitKey(0)

                        if iou > 1 or iou < 0:
                            raise ValueError('IOU value out of limits: %f' % iou)

                        if iou > bestIoU:
                            bestGT = detGT
                            bestIoU = iou
                    
                IoUs.append(bestIoU)
                # Increment False Positives or True Positives
                if bestIoU > 0: 
                    TP = TP + 1
                else:
                    FP = FP + 1

                # print("bestIoU: %f" % bestIoU)
                
                # it is also necessary to remove this best detection (match) from further
                # analysis, otherwise, it'd be "unfair"
                evalDetection.detections.remove(detEval)
                # also remove the groundtruth bounding box
                if bestGT != None:
                    gtDetection.detections.remove(bestGT)
                if evalDetection.count() > 0:
                    break #continue while loop
        # averageIoU = np.sum(IoUs) / (TP+FP) = averageIoU = np.average(IoUs)
        if IoUs != []:
            averageIoU = np.average(IoUs) 
        else:
            averageIoU = 0        
        # Return average IoU among all detected bounding boxes, True Positives and False Positives
        return averageIoU, TP, FP

    # @staticmethod
    # def evaluate_IoU(resultsPath, groundtruthPath):
    #     # resultsPath = '/home/rafael/thesis/temp/a' #pasta com os _dets.txt
    #     # groundtruthPath = '/home/rafael/thesis/simulations/data1/test_data' #pasta com os groundtruths .txt
    #     # Percorro os arquivos .txt em resultsPath para comparo com os mesmos em groundtruthsPath
    #     results = glob.glob(resultsPath+"/*.txt")
    #     IOUs = []
    #     for res in results:
    #         _, detTxt = utils.splitPathFile(res)
    #         # image is necessary to get original coordinates of the bounding boxes written on the file
    #         # for this example, the images are in the groundtruthPath
    #         imgPath = groundtruthPath+'/'+detTxt.replace('_dets.txt','.jpg').replace('.txt','.jpg')
    #         if os.path.isfile(imgPath) == False:
    #             imgPath = imgPath.replace('.jpg','.png')
    #         # given the file, get detections written on it
    #         evalDetections = Detections(res, imgPath)
    #         # get the path of the groundtruth file - Note: it doesnt have the '_dets' suffix
    #         gtFilePath = groundtruthPath+'/'+detTxt.replace('_dets.txt','.txt')
    #         groundTruthDetections = Detections(gtFilePath, imgPath)
            
    #         iou = YOLOHelper.evaluateDetections(groundTruthDetections, evalDetections)
    #         IoUs.append(iou)

class YOLOHelper:

    ###############################################################################
    # This method aims to generate the txt files needed to train YOLO
    ###############################################################################
    # dir_images     : directory containing the images
    # file_bb_info   : txt file containing bounding box, background, scale and other 
    #                  information related to the images merged with the background
    # out_dir_labels : directory where the generated txt files with the labels will be 
    #                  added
    @staticmethod
    def create_files_bb_yolo(dir_images, file_bb_info, out_dir_labels):
        if not out_dir_labels.endswith('/'):
            out_dir_labels = out_dir_labels+'/'
        if not dir_images.endswith('/'):
            dir_images = dir_images+'/'
        classId = 0
        # Read file with information
        fh1 = open(file_bb_info,"r")
        lastOpenedFile = ""
        for line in fh1:
            splitLine = line.split("\t")
            fileName = splitLine[0]
            xIni = splitLine[1]
            yIni = splitLine[2]
            xEnd = splitLine[3]
            yEnd = splitLine[4]
            image = splitLine[5]
            background = splitLine[6]
            scale = splitLine[7]
            angle = splitLine[8]
            flip = splitLine[9].replace("\n","")
            # if it is the first line of the txt, it does not contain information
            if fileName == "file_name":
                continue
            # make file name to write
            file2write = out_dir_labels+fileName.replace(".jpg","").replace(".png","")+".txt"
            # if this file had all bounding boxes written on, close it
            if lastOpenedFile != '' and os.path.isfile(file2write) == False:
                fh2.close()
                print("Finished writing on file: %s" % file2write)
            # get width and height of the image
            im = cv2.imread(dir_images+fileName)
            (x,y,w,h) = YOLOHelper.convert((int(im.shape[1]), int(im.shape[0])), (int(xIni),int(xEnd), int(yIni),int(yEnd)))
            (x2,y2,w2,h2) = YOLOHelper.deconvert((int(im.shape[1]), int(im.shape[0])), (x,y,w,h))
            if x2 != int(xIni) or y2 != int(yIni) or w2 != int(xEnd) or h2 != int(yEnd):
                print("Erro!")
            # open / create file needed by yolo to train
            fh2 = open(file2write,"a")
            lastOpenedFile = file2write
            fh2.write("%s %s %s %s %s\n" % (str(classId), x, y, w, h))




    # This method lists files in the folder directory. The files must match
    # the *.extension (*.jpg, *.txt, *.*).
    # The list of files will be in the outputTxtPath
    # If writeFullPath is True, it will be written the full path, otherwise only
    # the name of the file will be written.
    @staticmethod
    def writeListOfFiles(folderPath, outputTxtPath, extension='*', writeFullPath=True):
        if folderPath.endswith('/'):
            folderPath = folderPath+'*.'+extension
        else:
            folderPath = folderPath+'/*.'+extension
        # create output file
        fout = open(outputTxtPath, 'w')
        for f in  glob.glob(folderPath):
            # if needed to write only filename
            if writeFullPath == False:
                _,f = utils.splitPathFile(f)
            fout.write(f+'\n')
        fout.close()

    @staticmethod
    def parseDataFile(tags, filePath):
        f = open(filePath, 'r')
        text = f.read()
        text = text.replace(' ','')
        f.close()
        # TODO: Remove comments!
        content = []
        for t in tags:
            idx = text.find(t+'=')
            if idx == -1:
                content.append('')
                continue
            res = text[idx+len(t):]
            res = res[0:res.find("\n")]
            content.append(res.replace('=','').lstrip().rstrip())
            text = text.replace(t+res,'').replace('\n\n','\n')
        return content

    # VOC database has two folders that matters to the object detection: /Annotation and /JPG
    # The files in the /Annotation are xml files, that are not accepted by yolo.
    # Yolo needs the bounding boxes in a specific txt file.
    # This method converts the xml files in the 'in_xml_file_path' folder and convert them to
    # the .txt format needed by yolo, putting them in the 'out_txt_file_path' folder.
    # The classes is an array with the classes. 
    @staticmethod
    def convert_VOC_annotation(in_xml_file_path, out_txt_file_path, classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]):
        os.chdir(in_xml_file_path)
        xmlFiles = glob.glob("*.xml")
        for xmlFile in xmlFiles:
            out_file = out_txt_file_path+xmlFile.replace('.xml','.txt')
            out_file = open(out_file, 'w')
            in_file = open(xmlFile)
            tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                # difficult = obj.find('difficult').text
                cls = obj.find('name').text
                # if cls not in classes or int(difficult) == 1:
                #     continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = YOLOHelper.convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    ###############################################################################
    # This method shows images with bounding boxes given the .txt file in the yolo
    # format. Use this method to verify if the txt files containing bounding boxes
    # needed by Yolo during training are correct.
    ###############################################################################
    # dir_images        : directory containing the images
    # dir_boundingBox   : directory with the dir_boundingBox
    @staticmethod
    def show_image_with_bb(dir_images, dir_boundingBox):
        count = 0

        os.chdir(dir_images)
        for im in  glob.glob('*.jpg'):
            print("Reading file %s\n" % im)
            count = count+1
            # Check if respective txt file exist in the label folder
            txtFilePath = dir_boundingBox+'/'+im.replace(".jpg", ".txt")
            if os.path.isfile(txtFilePath):
                imagem = cv2.imread(dir_images+"/"+im)
                imWidth = imagem.shape[1]
                imHeight = imagem.shape[0]
                fh1 = open(txtFilePath, "r")
                for line in fh1:
                    line = line.replace("\n","")
                    splitLine = line.split(" ")
                    (xIn, yIn, xEnd, yEnd) = YOLOHelper.deconvert((imWidth, imHeight), (float(splitLine[1]), float(splitLine[2]),float(splitLine[3]),float(splitLine[4])))
                    cv2.rectangle(imagem,(xIn, yIn),(xEnd, yEnd),(0,255,0),1)
                # Show framedImage
                cv2.imshow('VDAO', imagem)
                cv2.waitKey(1000)
            else:
                print("Error! File %s not found.\n" % txtFilePath)
        print(count)

    ###############################################################################
    # This method returns an image with bounding boxes given by the .txt file in 
    # the yolo format. Use this method to generate images with bounding boxes with 
    # different colors in order to compare them.
    ###############################################################################
    # path_image       : path of the image
    # dir_boundingBox  : dir of the text file containing the bounding boxes info
    #                    if dir_boundingBox is a folder, it will look for the bb
    #                    (name+suffix.txt) inside the dir_boundingBox
    #                    if dir_boundingBox is a .txt file, it will be used to get
    #                    the boundingBox
    # labels           : array with labels to be added on top of the bounding box 
    #                    ex: ["person", "dog", "table"]
    # suffix           : suffix to be added in the txt file name. Needed  when imge
    #                    has name different than the txt file.
    #                    ex: '000001.jpg' ---bb---> 000001_dets.txt (suffix:'_dets')
    # colors           : array with colors of the bounding boxes
    #                    ex: [(255,0,0), (0,255,0), (0,0,255)]
    # thickness        : (int) thickness in pixels of the bounding box
    # hasConfidence    : (bool) if file with bounding box has confidence informed in 
    #                    the text. if not informed, method will find out by counting the
    #                    number of information in each line.
    #                    ex: without (False): index, x1_norm, y1_norm, cx_norm, cy_norm
    #                        with (True): index, confidence, x1_norm, y1_norm, cx_norm, cy_norm
    @staticmethod
    def get_image_with_bb(path_image, dir_boundingBox, 
        suffix = '',
        colors = [(229,0,31), (191,143,175),(121,137,242),(124,166,130),(242,133,61), (242,137,121), (226,0,242),(97,0,242), (191,225,255),(83,160,166), (0,102,0), (194,242,0), (191,153,0), (230,195,172), (140,49,35), (230,57,149), (195,108,217), (22,0,166), (0,230,214), (122,153,0),(115,77,0), (140,98,70),(102,77,77),(115,29,75),(88,57,115),(19,19,77), (48,143,191),(115,230,130), (113,115,86)], 
        thickness=2, hasConfidence=None, labels=None):
        print("Reading image %s\n" % path_image)
        if os.path.exists(path_image) == False:
            raise ValueError('Image file %s does not exist.' % path_image)
        path, img_name = utils.splitPathFile(path_image)
        if os.path.isdir(dir_boundingBox):
            path_bb = dir_boundingBox+'/'+img_name.replace('.jpg',suffix+'.txt').replace('.png','.txt')
        elif os.path.isfile(dir_boundingBox):
            path_bb = dir_boundingBox
        # Validate
        if os.path.isfile(path_bb) == False:
            raise ValueError('Cannot find txt file %s in the folder %s' %(img_name.replace('.jpg',suffix+'.txt'), path))
        imagem = cv2.imread(path_image)
        imWidth = imagem.shape[1]
        imHeight = imagem.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1
        safetyPixels = 0
        objectName = None
        fh1 = open(path_bb, "r")
        for line in fh1:
            line = line.replace("\n","")
            splitLine = line.split(" ")
            idx = int(splitLine[0]) #class
            r = colors[idx][0]
            g = colors[idx][1]
            b = colors[idx][2]
            # If coordinates have confidence
            if hasConfidence == None: # find out by counting number of information
                if len(splitLine) == 6:
                    hasConfidence = True
                else:
                    hasConfidence = False
            if hasConfidence == True:
                confidence = float(splitLine[1]) 
                x1 = float(splitLine[2])
                y1 = float(splitLine[3])
                x2 = float(splitLine[4])
                y2 = float(splitLine[5])
                if labels != None:
                    objectName = labels[idx] + str(" (%.2f)" % confidence)
            elif hasConfidence == False:
                x1 = float(splitLine[1])
                y1 = float(splitLine[2])
                x2 = float(splitLine[3])
                y2 = float(splitLine[4])
                if labels != None:
                    objectName = labels[idx]
            
            (xIn, yIn, xEnd, yEnd) = YOLOHelper.deconvert((imWidth, imHeight), (x1, y1, x2, y2))
            imagem = utils.add_bb_into_image(imagem, (xIn, yIn, xEnd, yEnd), (r,g,b), thickness, objectName)
            # Show image
            # cv2.imshow('objects', imagem)
            # # cv2.waitKey(5000)
        return imagem


    # When generating training files, it is necessary to create one file per image 
    # containing information about the bouding boxes of each image.
    # For each image, there must be one text (.txt) file with the same name as the image 
    # whose lines are information about the bouding boxes.
    # Each line must have:
    # id_image xCenter yCenter widthNorm heightNorm
    # The convert method converts the bouding boxes info in the format needed by yolo
    # size => (width, height) of the image
    # box => (X1, X2, Y1, Y2) of the bounding box
    @staticmethod
    def convert(size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        cx = (box[1] + box[0])/2.0 
        cy = (box[3] + box[2])/2.0 
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = cx*dw
        y = cy*dh
        w = w*dw
        h = h*dh
        # x,y => (bounding_box_center)/width_of_the_image
        # w => bounding_box_width / width_of_the_image
        # h => bounding_box_height / height_of_the_image
        return (x,y,w,h)
            
    # size => (width, height) of the image
    # box => (centerX, centerY, w, h) of the bounding box relative to the image
    @staticmethod
    def deconvert(size, box):
        w_box = round(size[0] * box[2])
        h_box = round(size[1] * box[3])
        
        xIn = round(((2*float(box[0]) - float(box[2]))*size[0]/2))
        yIn = round(((2*float(box[1]) - float(box[3]))*size[1]/2))

        xEnd = xIn + round(float(box[2])*size[0])
        yEnd = yIn + round(float(box[3])*size[1])
        return (xIn,yIn,xEnd,yEnd)

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if utils.boxesIntersect(boxA, boxB) == False:
            return 0
        interArea = YOLOHelper.getIntersectionArea(boxA,boxB)
        union = YOLOHelper.getUnionAreas(boxA,boxB,interArea=interArea)
        # if union == 0:
        #     return 0
        # intersection over union
        iou = interArea / union
        if iou < 0:
            return 0
        return iou
    
    @staticmethod
    def getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA) * (yB - yA)
    
    @staticmethod
    def getUnionAreas(boxA, boxB, interArea=None):
        area_A = YOLOHelper.getArea(boxA)
        area_B = YOLOHelper.getArea(boxB)
        if interArea == None:
            interArea = YOLOHelper.getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def getArea(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    # fileData : deve conter as tags:
    #           * names   : path com um arquivo .names contendo os nomes das classes
    #           * test    : path com um arquivo .txt com uma lista de imagens. Na pasta 
    #                       das imagens é obrigatório termos os respectivos .txt contendo
    #                       o id da classe e as coordenadas (x,y,w,h) normalizadas dos 
    #                       ounding boxes ground truth
    #           * results : pasta contendo arquivos .txt com os mesmos nomes dos arquivos
    #                       na pasta das imagens da tag anterior "test", porém com o sufixo
    #                       "_dets.txt" que contem id da classe, confidence e coordenadas 
    #                       (x,y,w,h) normalizadas.
    @staticmethod
    def validadeResults(fileDataPath):
        # array containing ground truth detections
        groundTruthDetections = []
        # array containing result detections
        resultDetections = []
        
        tags = ['names', 'test', 'results']
        values = YOLOHelper.parseDataFile(tags, fileDataPath)
        # read names
        if values[0] != '':
            f = open(values[0])
            names = f.readlines()
            f.close()
            names = [n.replace('\n','') for n in names]
        # read test files
        if values[1] != '':
            f = open(values[1])
            testFiles = f.readlines()
            f.close()
            tf = [n.replace('\n','') for n in testFiles]
            # Get the files and its paths
            testFiles = [utils.splitPathFile(n)[1] for n in tf]

            # AQUI: Tem que pegar os arquivos .txt ao inves dos .jpg e as informacoes de groundtruth desses arquivos
            # e coloca-las em um objeto detection e dar append no array groundTruthDetections[]


            # pathTestFiles = [utils.splitPathFile(n)[0] for n in tf]
            # print(pathTestFiles)

        # read result folder
        if values[2] != '' and os.path.isdir(values[2]):
            # the result folder must contain the 'xxx_dets.txt' files where prefix 
            # 'xxx' are the .txt test values in the array testFiles
            for groundTruthFile in testFiles:
                baseFile = groundTruthFile.replace('.jpg','')
                resultFile = values[2]+"/"+baseFile+"_dets.txt"
                # if result file exists, get its detection
                if os.path.exists(resultFile):
                    # given the file, get detections written on it
                    detections = Detections(resultFile).getDetections()
                    resultDetections.append(detections)

        # AQUI: Tem que comparar as bounding boxes dos resultDetections[] e groundTruthDetections[]

#################################################################################
# Example: Lists files in a folder writing them in a txt file
#################################################################################
#folderPath = '/home/rafael.padilla/thesis/simulations/data2/test_data/'
#outputTxtPath = '/home/rafael.padilla/thesis/simulations/data2/test.txt'
#YOLOHelper.writeListOfFiles(folderPath, outputTxtPath, extension='jpg', writeFullPath=True)

#################################################################################
# Example: It shows how to evaluate results with IoU (Intersection Over)
# Union).
# You need to inform the folder containing the _dets.txt file that contains
# the bounding boxes found by performing the test.
# Also need to inform the ground truth path with the .txt file that contains 
# the ground truth bounding boxes.
#############################################################################

#resultsPath = '/home/rafael.padilla/thesis/simulations/simulation6/results/res_test_final' #pasta com os _dets.txt
#groundtruthPath = '/home/rafael.padilla/thesis/simulations/data2/test_data' #pasta com os groundtruths .txt
##Percorro os arquivos .txt em resultsPath para comparo com os mesmos em groundtruthsPath
#results = glob.glob(resultsPath+"/*.txt")
#IoUs = []
#print("Starting calculating IoU between images:")
#count = 0
#for res in results:
#    _, detTxt = utils.splitPathFile(res)
#    if detTxt == 'log.txt' or detTxt == 'IOU_evaluation.txt':
#        continue
#    # image is necessary in order to get width and height to understand the bounding boxes coordinates.
#    # In this exampple, the images are in the groundtruthPath
#    imgPath = groundtruthPath+'/'+detTxt.replace('_dets.txt','.jpg').replace('.txt','.jpg')
#    if os.path.isfile(imgPath) == False:
#        imgPath = imgPath.replace('.jpg','.png')
#    # given the file, get detections written on it
#    evalDetections = Detections(res, imgPath)
#    # get the path of the groundtruth file - Note: it doesnt have the '_dets' suffix
#    gtFilePath = groundtruthPath+'/'+detTxt.replace('_dets.txt','.txt')
#    groundTruthDetections = Detections(gtFilePath, imgPath)
#    # print out info
#    _, f1 = nameGTimg = utils.splitPathFile(gtFilePath)
#    _, f2 = nameDetimg = utils.splitPathFile(res)    
#    print("IoU between %s and %s: " % (f1, f2), end='')
#    # calculate iou
#    iou = Detections.evaluateDetections(groundTruthDetections, evalDetections)
#    # print out result
#    print("%f" % iou)
#    count = count + 1
#    IoUs.append(iou)
#
#print("Pairs evaluated: %d" % count)
#print("Total iOU: %f" % np.sum(IoUs))
#print("Average iOU: %f" % np.average(IoUs))

#########################################################################
# Example: given a jpg and its corresponding txt containing the bb info 
# (in the yolo format) show image with labels and classes names
#########################################################################
# folder_image = '/home/rafael/thesis/simulations/data1/test_data'
# labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# os.chdir(folder_image)
# imgs = glob.glob("*.jpg")
# for i in imgs:
#     path_image = folder_image+'/'+i
#     imagem = YOLOHelper.get_image_with_bb(path_image, folder_image)
#     cv2.imshow('VDAO', imagem)
#     cv2.waitKey(5000)

