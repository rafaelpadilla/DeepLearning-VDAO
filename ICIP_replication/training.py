import os 
from enum import Enum
from collections import Counter
import json
import glob
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir,'../VDAO_Access/VDAO_files' ))
from Table import SourcePackage
sys.path.insert(0, os.path.join(current_dir,'../VDAO_Access/' ))
from VDAOVideo import VDAOVideo

DIR = '/local/home/common/datasets/VDAO/'
FOLDER = os.path.join(DIR,'vdao_research')
TOTAL_TABLES = 59

tables_dir = [os.path.join(FOLDER,str(i)) for i in range(1,59+1)]

objects = Enum('Object', 'STRING_ROLL BAG WHITE_BOX LAMP_BULB_BOX SPOTLIGHT_BOX MUG BLUE_COAT WRENCH BOTTLE BLUE_BOX BACKPACK PINK_BACKPACK BOTTLE_CAP UMBRELLA GREEN_BOX SHOE DARK_BLUE_BOX CAMERA_BOX PINK_BOTTLE BLACK_BACKPACK WHITE_JAR BROWN_BOX TOWEL BLACK_COAT')



# Based on the json file, get all research videos
jsonFilePath = os.path.join(current_dir,'vdao_training.json')
f = open(jsonFilePath)
with open(jsonFilePath) as f:
    jsonData = json.load(f)
[sourcePackage, tables, videos] = SourcePackage.CreateSourcePackage(jsonData)

# Separate videos into objects and references
objects = [v for v in videos if 'object' in v.name.lower()]
references = [v for v in videos if 'reference' in v.name.lower()]

# Due to the size constraint of the VDAO, the 59-video set was split onto nine disjoint test subsets, 
# each one containing all pairs of target/reference videos of a given object. 
# We trained nine networks using one video group for test, and the eight groups for training, 
# while separating 10% of the frames for parameter validation.

# Get amount of target videos for each class
amount_videos = Counter(o.object_class for o in objects)

# Get amount of frames available for each class
for objclass in amount_videos:
    # Get all tables containing the object
    tables = [o.sourceTable.name for o in objects if o.object_class == objclass]
    numberOfFrames = 0
    for t in tables:
        paths = glob.glob(os.path.join(FOLDER,t)+'/ref-*.avi')
        assert len(paths) == 1 # make sure it has only one video containing object
        myVideo = VDAOVideo(paths[0])
        # print(paths[0])
        numberOfFrames += myVideo.videoInfo.getNumberOfFrames()
    print('%s (%d): %d'%(objclass,len(tables),numberOfFrames))


aaa = 123
