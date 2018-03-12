import os 

class Annotation:
    """
	This class represents a .txt file containing the labels and bounding
    boxes of objects in a given video.

        Developed by: Rafael Padilla
        SMT - Signal Multimedia and Telecommunications Lab
        COPPE - Universidade Federal do Rio de Janeiro
        Last modification: March 9th 2018
    """
    
    def __init__(self, annotationFilePath = None, totalFrames=None):
        self.totalFrames = totalFrames
        self.annotationFilePath = annotationFilePath
        self.listAnnotation = []
        self.parsed = self._parseFile()
        self.error = not self.parsed
    
    # Returns True if parse was successfully done, otherwise returns False
    def _parseFile(self):
        if os.path.exists(self.annotationFilePath) == False:
            return False

        # create array from 0 to totalFrames + 1
        items = self.totalFrames+1
        self.listAnnotation.clear()
        [self.listAnnotation.append((('NONE'),None)) for i in range(items)]

        f = open(self.annotationFilePath,"r")
        for line in f:
            params = line.split(' ')
            readFrame = int(params[1])
            self.listAnnotation[readFrame] = (params[0], (int(params[2]),int(params[3]),int(params[4]),int(params[5])))
        f.close()
        return True