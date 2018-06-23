import os
import requests

def convertIntToStr(number):
    numStr = str(number)
    if len(numStr) == 1:
        numStr = '0%s' % numStr
    return numStr

def downloadContent(url, saveDir):
    fileName = url.rsplit('/', 1)[1]
    if not os.path.isfile(os.path.join(saveDir,fileName)):
        os.chdir(saveDir)
        os.system('wget %s' % url) 
    
    # r = requests.get(url, allow_redirects=True)
    # if url.find('/'):
    #     fileName = url.rsplit('/', 1)[1]
    #     fileDir = os.path.join(saveDir,fileName)
    #     open(fileDir, 'wb').write(r.content)

# Create folder
# base_dir = '/local/home/common/datasets/VDAO/single_objs/'
base_dir = '/home/rafael/del/'
tables = [1,2,3,4,5,6,7,8,9,10]
numTables = len(tables)
objVideosPerTable = [10,5,17,9,5,14,1,2,1,2]
refVideosPerTable = [2,1,1,1,1,1,1,1,1,1]
objVideosIllumination = ['amb','amb','amb','ext','ext','ext','amb','amb','ext','ext']
objVideosSingleMulti = ['sing','sing','sing','sing','sing','sing','mult','mult','mult','mult']

_tableDirectory = 'table_#NUMTABLE'
_directory = 'Table_#NUMTABLE-Object_#NUMOBJECT'
_videoReferencePath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-#SINGLEMULTI-#ILLUMINATION-part#NUMTABLE-video#NUMVIDEO.avi'
_videoObjectPath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/obj-#SINGLEMULTI-#ILLUMINATION-part#NUMTABLE-video#NUMVIDEO.avi'
_annotationObjectPath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/ann/obj-#SINGLEMULTI-#ILLUMINATION-part#NUMTABLE-video#NUMVIDEO.txt'

for idTable in range(numTables):
    tableInt = tables[idTable]
    tableStr = convertIntToStr(tableInt)
    qteObjVideosPerTable = objVideosPerTable[idTable]
    illu = objVideosIllumination[idTable]
    singleMulti = objVideosSingleMulti[idTable]
    # Create vector with paths (['ref']: reference videos paths; ['obj']: object videos paths)
    qteRef = refVideosPerTable[idTable]
    # Create table directory if it does not exist
    directory = _tableDirectory.replace('#NUMTABLE', tableStr)
    directory = os.path.join(base_dir,directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)
    # Reference videos
    for refInt in range(qteRef):
        refStr = convertIntToStr(refInt+1)
        # reference video
        saveFileDir = os.path.join(directory,'Table_%s-Reference_%s' % (tableStr,refStr))
        if not os.path.exists(saveFileDir): #create file if it does not exist
            os.makedirs(saveFileDir)
        urlToDownload = _videoReferencePath.replace('#NUMTABLE',tableStr).replace('#NUMVIDEO',refStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
        downloadContent(urlToDownload,os.path.join(directory,saveFileDir))
        
    # Object videos and annotations
    for qteObj in range(qteObjVideosPerTable):
        qteObjStr = convertIntToStr(qteObj+1)
        # object video
        saveFileDir = os.path.join(directory,'Table_%s-Object_%s' % (tableStr,qteObjStr))
        if not os.path.exists(saveFileDir): #create file if it does not exist
            os.makedirs(saveFileDir)
        urlToDownload = _videoObjectPath.replace('#NUMTABLE',tableStr).replace('#NUMVIDEO',qteObjStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
        downloadContent(urlToDownload,os.path.join(directory,saveFileDir))
        # annotation file
        urlToDownload = _annotationObjectPath.replace('#NUMTABLE',tableStr).replace('#NUMVIDEO',qteObjStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
        downloadContent(urlToDownload,os.path.join(directory,saveFileDir))


