import os
import requests

def convertIntToStr(number):
    numStr = str(number)
    if len(numStr) == 1:
        numStr = '0%s' % numStr
    return numStr

def contentExists(url):
    r = requests.head(url)
    return r.status_code == requests.codes.ok

def downloadContent(url, saveDir):
    if contentExists(url):
        fileName = url.rsplit('/', 1)[1]
        if not os.path.isfile(os.path.join(saveDir,fileName)):
            os.chdir(saveDir)
            os.system('wget %s' % url) 
    else:
        print("Url does not exist: %s" % url)

def downloadVideosTreinamento():
    # Create folder
    # base_dir = '/local/home/common/datasets/VDAO/forTraining/'
    base_dir = '/home/rafael/del/'
    tables = [1,2,3,4,5,6,7,8,9,10]
    numTables = len(tables)
    objVideosPerTable = [10,5,17,9,5,14,1,2,1,2]
    refVideosPerTable = [2,1,1,1,1,1,1,1,1,1]
    # Ex: Urls
    # Table 1:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-amb-part01-video01.avi
    # Table 2:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-amb-part02-video01.avi
    # Table 3:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-amb-part03-video01.avi
    # Table 4:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-ext-part01-video01.avi
    # Table 5:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-ext-part02-video01.avi
    # Table 6:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-sing-ext-part03-video01.avi
    # Table 7:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-mult-amb-part01-video01.avi
    # Table 8:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-mult-amb-part02-video01.avi
    # Table 9:  http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-mult-ext-part01-video01.avi
    # Table 10: http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-mult-ext-part02-video01.avi
    objVideosIllumination = ['amb','amb','amb','ext','ext','ext','amb','amb','ext','ext']
    objVideosSingleMulti = ['sing','sing','sing','sing','sing','sing','mult','mult','mult','mult']
    videoParts = ['part01','part02','part03','part01','part02','part03','part01','part02','part01','part02']
    _tableDirectory = 'table_#NUMTABLE'
    _videoReferencePath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/ref-#SINGLEMULTI-#ILLUMINATION-#VIDEOPARTS-video#NUMVIDEO.avi'
    _videoObjectPath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/avi/obj-#SINGLEMULTI-#ILLUMINATION-#VIDEOPARTS-video#NUMVIDEO.avi'
    _annotationObjectPath = 'http://www02.smt.ufrj.br/~tvdigital/database/objects/data/ann/obj-#SINGLEMULTI-#ILLUMINATION-#VIDEOPARTS-video#NUMVIDEO.txt'
    for idTable in range(numTables):
        tableInt = tables[idTable]
        tableStr = convertIntToStr(tableInt)
        qteObjVideosPerTable = objVideosPerTable[idTable]
        illu = objVideosIllumination[idTable]
        singleMulti = objVideosSingleMulti[idTable]
        videoPart = videoParts[idTable]
        print('#####################################################################################')
        print('%s' % tableStr)
        print('#####################################################################################')
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
            if not os.path.exists(saveFileDir): #create folder if it does not exist
                os.makedirs(saveFileDir)
            urlToDownload = _videoReferencePath.replace('#VIDEOPARTS',videoPart).replace('#NUMVIDEO',refStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
            downloadContent(urlToDownload,os.path.join(directory,saveFileDir))
        # Object videos and annotations
        for qteObj in range(qteObjVideosPerTable):
            qteObjStr = convertIntToStr(qteObj+1)
            # object video
            saveFileDir = os.path.join(directory,'Table_%s-Object_%s' % (tableStr,qteObjStr))
            if not os.path.exists(saveFileDir): #create folder if it does not exist
                os.makedirs(saveFileDir)
            urlToDownload = _videoObjectPath.replace('#VIDEOPARTS',videoPart).replace('#NUMVIDEO',qteObjStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
            downloadContent(urlToDownload,os.path.join(directory,saveFileDir))
            # annotation file
            urlToDownload = _annotationObjectPath.replace('#VIDEOPARTS',videoPart).replace('#NUMVIDEO',qteObjStr).replace('#ILLUMINATION',illu).replace('#SINGLEMULTI', singleMulti)
            downloadContent(urlToDownload,os.path.join(directory,saveFileDir))

def downloadVideosTeste():
      # Create folder
    # base_dir = '/local/home/common/datasets/VDAO/forTesting/'
    base_dir = '/home/rafael/del/'
    tables = [i for i in range(1,60)] #1,2,3,...,59
    # Ex: Urls
    # Table 1:
    # http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/ref-video01.avi
    # http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/obj-video01.avi
    # Table 2
    # http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/ref-video02.avi
    # http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/obj-video02.avi
    _tableDirectory = 'table_#NUMTABLE'
    _videoReferencePath = 'http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/ref-video#NUMTABLE.avi'
    _videoObjectPath = 'http://www02.smt.ufrj.br/~tvdigital/database/research/data/avi/obj-video#NUMTABLE.avi'
    for idTable in tables:
        tableStr = convertIntToStr(idTable)
        print('#####################################################################################')
        print('%s' % tableStr)
        print('#####################################################################################')
        # Reference videos
        # Create table directory if it does not exist
        directory = _tableDirectory.replace('#NUMTABLE', tableStr)
        directory = os.path.join(base_dir,directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)
        # Reference video -> Theres always one reference video (Table_X-Reference_01/)
        saveFileDir = os.path.join(directory,'Table_%s-Reference_01' % (tableStr))
        if not os.path.exists(saveFileDir): #create folder if it does not exist
            os.makedirs(saveFileDir)
        urlToDownload = _videoReferencePath.replace('#NUMTABLE',tableStr)
        downloadContent(urlToDownload,os.path.join(directory,saveFileDir))
        # Object video -> Theres always one reference video (Table_X-Reference_01/)
        saveFileDir = os.path.join(directory,'Table_%s-Object_01' % (tableStr))
        if not os.path.exists(saveFileDir): #create folder if it does not exist
            os.makedirs(saveFileDir)
        urlToDownload = _videoObjectPath.replace('#NUMTABLE',tableStr)
        downloadContent(urlToDownload,os.path.join(directory,saveFileDir))

#####################################################################################
# Vídeos para treinamento
# http://www02.smt.ufrj.br/~tvdigital/database/objects/page_01.html
#####################################################################################
# downloadVideosTreinamento()


#####################################################################################
# Vídeos para teste
# http://www02.smt.ufrj.br/~tvdigital/database/research/page_01.html
#####################################################################################
downloadVideosTeste()