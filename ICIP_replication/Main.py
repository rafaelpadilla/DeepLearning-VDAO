import json
import numpy as np
import os
import socket
#######################################################################
# Defining input and output folders
#######################################################################
def define_folders():
    hostname = socket.gethostname()
    dirVideos, outputDir = '' , ''
    if hostname == 'rafael-Lenovo-Z40-70': # notebook pessoal
        pass
    elif hostname == 'notesmt': # notebook SMT
        dirVideos = '/media/storage/VDAO/' 
        outputDir = '/media/storage/VDAO/' 
    elif hostname == 'teresopolis.smt.ufrj.br': # teresopolis
        pass
    elif hostname.startswith("node") or hostname.startswith("head"): #nodes do cluster smt
        dirVideos = "/nfs/proc/rafael.padilla/" 
        outputDir = "/nfs/proc/rafael.padilla/"
    else:  # Path not defined
        raise Exception('Error: Folder with videos is not defined!')
    return dirVideos, outputDir

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

def get_objects_info(classes):
    ret = {}
    with open(JSON_FILE, "r") as read_file:
        data = json.load(read_file)
        for t in data['tables']:
            for o in data['tables'][t]['objects']:
                if o['object_class']  in classes:
                    if t not in ret:
                        ret[t] = []
                    ret[t].append(o)
    return ret

#type = 'tar' e 'ref'
def get_file_filters(fold, types=['obj'], include_table_folder=False):
    target_class = folds_number[fold]
    search_terms = []
    items = target_objects.copy()
    items.remove(target_class)
    for item in items:
        obj_info = get_objects_info(item)
        for table in obj_info:
            folder_prefix = folder_prefix = '%s/'%table if include_table_folder else ''
            t = table.replace('table_','')
            if 'tar' in types:
                st = ['t%s_obj%s*' % (t, o['name'].replace('object ','')) for o in obj_info[table]]
                [search_terms.append(folder_prefix+s) for s in st]
            if 'ref' in types:
                st = ['t%s_*ref*' % t]
                [search_terms.append(folder_prefix+s) for s in st]

    return search_terms


# search_terms = get_file_filters('fold_1',['tar'])


import torch
import torchvision.transforms as transforms
import My_Resnet
from PIL import Image
from torch.autograd import Variable
from torchsummary import summary

# Load the pretrained model
resnet50 = My_Resnet.resnet50(pretrained=True)
# Print network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = (3,720,1280)
# summary(resnet50.to(device), input_size)
# Set model to evaluation mode
resnet50.eval()
# Set normalization to be used with pretrained weights
normalize = transforms.Normalize(mean=My_Resnet.mean_imagenet,
                                 std=My_Resnet.std_imagenet)
to_tensor = transforms.ToTensor()

# Not rescaling image
# scaler = transforms.Scale(input_size[1:3])

def get_feature_vector(image_path, feature_size):
    # 1. Load the image with Pillow library
    img = Image.open(image_path)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    feature_map = torch.zeros(feature_size)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        feature_map.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    resnet50(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return feature_map

layers_and_sizes = {
    'conv1':      (64,360,640),
    'residual1':  (256,180,320), # layer 1
    'residual2':  (256,180,320), # layer 1
    'residual3':  (256,180,320), # layer 1
    'residual4':  (512,90,160),  # layer 2 
    'residual5':  (512,90,160),  # layer 2 
    'residual6':  (512,90,160),  # layer 2 
    'residual7':  (512,90,160),  # layer 2 
    'residual8':  (1024,45,80),  # layer 3
    'residual9':  (1024,45,80),  # layer 3
    'residual10': (1024,45,80),  # layer 3
    'residual11': (1024,45,80),  # layer 3
    'residual12': (1024,45,80),  # layer 3
    'residual13': (1024,45,80),  # layer 3
    'residual14': (2048,23,40),  # layer 4
    'residual15': (2048,23,40),  # layer 4
    'residual16': (2048,23,40),  # layer 4
}

layers_and_extractors = {
    'conv1':      resnet50._modules.get('conv1'),
    'residual1':  resnet50._modules.get('layer1')._modules.get('0'),
    'residual2':  resnet50._modules.get('layer1')._modules.get('1'),
    'residual3':  resnet50._modules.get('layer1')._modules.get('2'),
    'residual4':  resnet50._modules.get('layer2')._modules.get('0'),
    'residual5':  resnet50._modules.get('layer2')._modules.get('1'),
    'residual6':  resnet50._modules.get('layer2')._modules.get('2'),
    'residual7':  resnet50._modules.get('layer2')._modules.get('3'),
    'residual8':  resnet50._modules.get('layer3')._modules.get('0'),
    'residual9':  resnet50._modules.get('layer3')._modules.get('1'),
    'residual10': resnet50._modules.get('layer3')._modules.get('2'),
    'residual11': resnet50._modules.get('layer3')._modules.get('3'),
    'residual12': resnet50._modules.get('layer3')._modules.get('4'),
    'residual13': resnet50._modules.get('layer3')._modules.get('5'),
    'residual14': resnet50._modules.get('layer4')._modules.get('0'),
    'residual15': resnet50._modules.get('layer4')._modules.get('1'),
    'residual16': resnet50._modules.get('layer4')._modules.get('2'),
}

layers_to_extract = ['conv1', 'residual1', 'residual2', 'residual3', 
                     'residual4', 'residual5', 'residual6', 'residual7', 
                     'residual8', 'residual9', 'residual10', 'residual11', 
                     'residual12', 'residual13', 'residual14', 'residual15', 
                     'residual16']
# Get directories to read frames from and write feature maps
dir_read, dir_save = define_folders()
dir_read = os.path.join(dir_read,'shortest_distance_results','frames')
dir_save = os.path.join(dir_save,'shortest_distance_results','features')


############### TRANSFORMAR TODAS IMAGENS EM FEATURES ##################### 
# #  Loop through layers to extract features
# for layer_name in layers_to_extract:
#     print('#'*80)
#     print('Extracting features from layer: %s' % layer_name)
#     # Use the model object to select the desired layer
#     layer = layers_and_extractors[layer_name]
#     size_output_layer = layers_and_sizes[layer_name]

#     # Define directory to save features
#     dir_to_save_features = os.path.join(dir_save, layer_name)
#     if not os.path.isdir(dir_to_save_features):
#         os.makedirs(dir_to_save_features)
    
#     # Loop through each image
#     for subdir, dirs, files in os.walk(dir_read):
#         for file_name in files:
#             image_file_path = os.path.join(subdir, file_name)
#             if not image_file_path.endswith('.png'):
#                 print('-> Error: not a recognized image file: %s' % file_name)
#                 continue
#             # Get feature map
#             feature_map = get_feature_vector(image_file_path, size_output_layer)
#             # Save it
#             path_to_save = 'f_%s_%s' % (layer_name,file_name.replace('.png','.npy'))
#             np.save(os.path.join(dir_to_save_features,path_to_save),feature_map)
#             print('Feature %s sucessfully saved.' % path_to_save)
# print('Done!')




layers_to_extract = ['residual16']
search_terms = 't01_obj1'


#  Loop through layers to extract features
for layer_name in layers_to_extract:
    print('#'*80)
    print('Extracting features from layer: %s' % layer_name)
    # Use the model object to select the desired layer
    layer = layers_and_extractors[layer_name]
    size_output_layer = layers_and_sizes[layer_name]

    # Define directory to save features
    dir_to_save_features = os.path.join(dir_save, layer_name)
    if not os.path.isdir(dir_to_save_features):
        os.makedirs(dir_to_save_features)
    
    # Loop through search term
    for st in search_terms:
        st = os.path.join(dir_read, st)
        files = glob.glob(st)
        for image_file_path in files:
            if not image_file_path.endswith('.png'):
                print('-> Error: not a recognized image file: %s' % file_name)
                continue
        # # Get feature map
        # feature_map = get_feature_vector(image_file_path, size_output_layer)
        # # Save it
        # path_to_save = 'f_%s_%s' % (layer_name,file_name.replace('.png','.npy'))
        # np.save(os.path.join(dir_to_save_features,path_to_save),feature_map)
        # print('Feature %s sucessfully saved.' % path_to_save)
print('Done!')