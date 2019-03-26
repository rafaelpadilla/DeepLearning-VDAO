import json
import time
import numpy as np
import os
import socket
import glob
import shutil
import torch
import torchvision.transforms as transforms
import _init_paths_
from VDAOVideo import VDAOVideo
import cv2
import My_Resnet
from PIL import Image
from torch.autograd import Variable
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")

#######################################################################
# Defining input and output folders
#######################################################################
def define_folders():
    hostname = socket.gethostname()
    dirRead, outputDir = '' , ''
    if hostname == 'rafael-Lenovo-Z40-70': # notebook pessoal
        pass
    elif hostname == 'notesmt': # notebook SMT
        # Frames are read from the same folder as the features are saved
        dirRead = '/media/storage/VDAO/' 
        outputDir = '/media/storage/VDAO/' 
    elif hostname == 'teresopolis.smt.ufrj.br': # teresopolis
        pass
    elif hostname.startswith("node") or hostname.startswith("head"): #nodes do cluster smt
        dirRead = "/home/rafael.padilla/workspace/rafael.padilla/" 
        outputDir = "/nfs/proc/rafael.padilla/"
    elif hostname.startswith('taiwan') or hostname.startswith('zermatt'): # maquina com GPU taiwan
         dirRead = "/home/rafael.padilla/workspace/rafael.padilla/" 
         outputDir = "/nfs/proc/rafael.padilla/"
    else:  # Path not defined
        raise Exception('Error: Folder with videos is not defined!')
    return dirRead, outputDir

def get_feature_vector(image, layer_name, feature_size, transformations, pooling_size):
    # 1. If not loaded yet, load the image with Pillow library
    if isinstance(image, str):
        image = Image.open(image)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = torch.tensor(transformations(image).unsqueeze(0),dtype=torch.float, device=device)
    # 3. Create a vector of zeros that will hold our feature vector
    layer = resnet_layers_and_extractors[layer_name]
    # Define pooling
    if pooling_size != None:
        pooling_layer = torch.nn.AvgPool2d(pooling_size)
    feature_map = torch.zeros(feature_size)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        if pooling_size != None:
            feature_map.copy_(pooling_layer(o.data.squeeze()))
        else:
            feature_map.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    resnet50(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return feature_map

# Lambda to insert "0" in 1-digit numbers (eg: 4->"04")
l_double_digit = lambda x : '0'+str(x) if len(str(x)) == 1 else str(x)

# Load the pretrained model
resnet50 = My_Resnet.resnet50(pretrained=True)

JSON_FILE = 'vdao_object.json'
target_objects = ['shoe', 'towel', 'brown box', 'black coat', 'black backpack', 'dark-blue box', 'camera box', 'white jar', 'pink bottle']
folds_number = {'fold_1': target_objects[0],
                'fold_2': target_objects[1],
                'fold_3': target_objects[2],
                'fold_4': target_objects[3],
                'fold_5': target_objects[4],
                'fold_6': target_objects[5],
                'fold_7': target_objects[6],
                'fold_8': target_objects[7],
                'fold_9': target_objects[8]
}

# Output size of each layer when input image is is 3,720.1280
layers_and_sizes_no_resize = {
    'conv1'     : (64,360,640),
    'residual1' : (256,180,320), # layer 1
    'residual2' : (256,180,320), # layer 1
    'residual3' : (256,180,320), # layer 1
    'residual4' : (512,90,160),  # layer 2 
    'residual5' : (512,90,160),  # layer 2 
    'residual6' : (512,90,160),  # layer 2 
    'residual7' : (512,90,160),  # layer 2 
    'residual8' : (1024,45,80),  # layer 3
    'residual9' : (1024,45,80),  # layer 3
    'residual10': (1024,45,80),  # layer 3
    'residual11': (1024,45,80),  # layer 3
    'residual12': (1024,45,80),  # layer 3
    'residual13': (1024,45,80),  # layer 3
    'residual14': (2048,23,40),  # layer 4
    'residual15': (2048,23,40),  # layer 4
    'residual16': (2048,23,40),  # layer 4
}


resnet_layers_and_extractors = {
    'conv1'     : resnet50._modules.get('conv1'),
    'residual1' : resnet50._modules.get('layer1')._modules.get('0'),
    'residual2' : resnet50._modules.get('layer1')._modules.get('1'),
    'residual3' : resnet50._modules.get('layer1')._modules.get('2'),
    'residual4' : resnet50._modules.get('layer2')._modules.get('0'),
    'residual5' : resnet50._modules.get('layer2')._modules.get('1'),
    'residual6' : resnet50._modules.get('layer2')._modules.get('2'),
    'residual7' : resnet50._modules.get('layer2')._modules.get('3'),
    'residual8' : resnet50._modules.get('layer3')._modules.get('0'),
    'residual9' : resnet50._modules.get('layer3')._modules.get('1'),
    'residual10': resnet50._modules.get('layer3')._modules.get('2'),
    'residual11': resnet50._modules.get('layer3')._modules.get('3'),
    'residual12': resnet50._modules.get('layer3')._modules.get('4'),
    'residual13': resnet50._modules.get('layer3')._modules.get('5'),
    'residual14': resnet50._modules.get('layer4')._modules.get('0'),
    'residual15': resnet50._modules.get('layer4')._modules.get('1'),
    'residual16': resnet50._modules.get('layer4')._modules.get('2'),
}

avg_pooling_sizes = {
    'conv1'     : 20,
    'residual1' : 28,
    'residual2' : 28,
    'residual3' : 28,
    'residual4' : 21,
    'residual5' : 21,
    'residual6' : 21,
    'residual7' : 21,
    'residual8' : 14,
    'residual9' : 14,
    'residual10': 14,
    'residual11': 14,
    'residual12': 14,
    'residual13': 14,
    'residual14': 7,
    'residual15': 7,
    'residual16': 7,
}

# Print network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# summary(resnet50.to(device), input_size)
# Set model to evaluation mode
resnet50.to(device)
resnet50.eval()
# Get directories to read frames from and write feature maps
dir_read, dir_save = define_folders()

def get_size_feature_vector(input_size, layers_to_extract, post_pooling=False):
    '''
    Given the input size and, names of the layers to extract and the need of a post pooling,
    obtain the size of the feature vector.
    * input_size: Size of the input image [height, width, channels] (Ex: [360,640,3])
    * layers_to_extract: List containing the names of layers to extract (Ex: ['conv1', 'residual1', 'residual2'])
    * post_pooling: Boolean informing if a post pooling is needed after extracting feature. If True, the size of the pooling is obtained from the avg_pooling_sizes
    return: torch.Size object containing the dimensions of the output tensor.
    '''
    # Create empty input
    transformations = transforms.Compose([transforms.ToTensor()])
    img = np.zeros(input_size)
    t_img = torch.tensor(transformations(img).unsqueeze(0), dtype=torch.float, device=device)
    # t_img = transformations(img).unsqueeze(0).clone().detach()
    # Create feature_map dictionary to store the sizes
    feature_maps_sizes = {}
    # Define a function that will copy the output of a layer
    def get_activation(name):
        def hook(m, i, o):
            # Define pooling
            if post_pooling:
                pooling_size = avg_pooling_sizes[name]
                # Calculate output size with padding = 0 and stride = pooling_size: As in the ICIP implementation
                # [channels,(height-kernel_size+2*padding)/stride)+1,(width-kernel_size+2*padding)/stride)+1]
                pooling_layer = torch.nn.AvgPool2d(pooling_size)
                feature_maps_sizes[name] = pooling_layer(o.data.squeeze()).shape
            else:
                feature_maps_sizes[name] = o.data.squeeze().shape
        return hook
    for layer_name in layers_to_extract:
        # Get layer to add a hook
        layer = resnet_layers_and_extractors[layer_name]
        # Attach that function to our selected layer
        h = layer.register_forward_hook(get_activation(layer_name))
        # pass the image empty by the model
        resnet50(t_img)
        # Detach hook from thelayer
        h.remove()
    # Return the feature vector
    return feature_maps_sizes

def generate_and_save_features(dir_videos, table_name, dir_to_save_features, layer_name, resize_factor, post_pooling):
    input_size=[720,1280,3]
    # Resize according to the factor
    input_size = [int(input_size[0]/resize_factor),int(input_size[1]/resize_factor),input_size[2]]
    # Get the size of the feature given the layer_name, input_size and the post_poolint (bool)
    size_feature = get_size_feature_vector(input_size=input_size, layers_to_extract=[layer_name], post_pooling=post_pooling)
    size_feature = size_feature[layer_name]
    # Define transformations to be used to extract features
    resize_transform = transforms.Resize(input_size[:2])
    normalize_transform = transforms.Normalize(mean=My_Resnet.mean_imagenet,
                                 std=My_Resnet.std_imagenet)
    to_tensor_transform = transforms.ToTensor()
    transformations = transforms.Compose([
            resize_transform,
            to_tensor_transform,
            normalize_transform])
    if post_pooling:
        pooling_size = avg_pooling_sizes[layer_name]
    else:
        pooling_size = None
    # Get all videos within the dir_videos
    print(os.path.join(dir_videos,table_name.replace(' ','_'),'*','*.avi'))
    videos_paths = glob.glob(os.path.join(dir_videos,table_name.replace(' ','_'),'*','*.avi'))
    # Loop through each video
    for video_path in videos_paths:
        print('Reading frames from video %s' % video_path)
        vid = VDAOVideo(video_path)
        # Depending on the video, it is a target or a reference
        if '-Object_' in video_path:
            tag = '-Object_'
            type_feat = 'tar'
        elif '-Reference_' in video_path:
            tag = '-Reference_'
            type_feat = 'ref'
        # Get table number
        table_number = table_name.replace('table ','')
        # Get Object/Reference video number
        idx = video_path.index(tag)
        object_number = video_path[idx+len(tag):]
        idx = object_number.index('/')
        object_number = object_number[:idx]
        # Create dir chain to store the features
        dirs = os.path.dirname(video_path).split('/')
        idx = dirs.index(table_name.replace(' ','_'))
        dir_features_to_save_table = dir_to_save_features
        for i in dirs[idx:]:
            dir_features_to_save_table = os.path.join(dir_features_to_save_table,i)
        dir_features_to_save_table = os.path.join(dir_features_to_save_table,layer_name)
        print('Saving frames in %s' % dir_features_to_save_table)
        if not os.path.isdir(dir_features_to_save_table):
            os.makedirs(dir_features_to_save_table)
        total_frames = vid.videoInfo.getNumberOfFrames()
        for frame_num in range(1,total_frames+1):
            frame = vid.GetFrame(frame_num)
            if not frame[0]:
                raise Exception('Error: Invalid frame!')
            # As frames are retrived as BGR, convert it to RGB
            frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            feature_vector = get_feature_vector(Image.fromarray(np.uint8(frame)), layer_name, size_feature, transformations, pooling_size)
            # Define name to save the feature
            feat_name = f'feat_[%s]_{layer_name}_t{table_number}_{type_feat}_vid{object_number}_frame_%d.npy' % (os.path.basename(video_path).replace('.avi',''), frame_num-1)
            feat_path = os.path.join(dir_features_to_save_table,feat_name)
            # Garante que a feature ainda não foi gerada
            assert os.path.isfile(feat_path) == False
            np.save(feat_path,feature_vector)

####################################################################################
################################### Definitions ####################################
####################################################################################
# Change here to 'research' or 'object'
database_type = 'research'
# Change here 'shortest_distance' or 'dtw'
alignment_mode = 'shortest_distance'
layers_to_extract = ['conv1', 'residual1', 'residual2', 'residual3', 
                     'residual4', 'residual5', 'residual6', 'residual7', 
                     'residual8', 'residual9', 'residual10', 'residual11', 
                     'residual12', 'residual13', 'residual14', 'residual15', 
                     'residual16']
tables_to_process = ['table %s' % l_double_digit(i) for i in range(1,60)]
tables_to_process = ['table 01']
# Parameters
resize_factor = 2
apply_pooling = True
start = time.time()
print('Starting process at: %s'%start)
print('Main parameters:')
print('* resize_factor = %f'%resize_factor)
print('* apply_pooling = %s'%apply_pooling)

dir_videos = os.path.join(dir_read, 'vdao_%s/'%database_type)
dir_save = os.path.join(dir_save,'vdao_alignment_%s'%database_type ,alignment_mode,'features')
print('Folder to read videos: %s' % dir_videos)
print('Folder to save features: %s' % dir_save)

dir_save_features = os.path.join(dir_videos,)
for table_name in tables_to_process:
    print('\nGenerating features from table %s' % table_name)
    for layer_name in layers_to_extract:
        print('* Layer: %s' % layer_name)
        generate_and_save_features(dir_videos, table_name, dir_save, layer_name, resize_factor=resize_factor,post_pooling=apply_pooling)

end = time.time()
print('\nFinished process with %s seconds'%(end-start))


