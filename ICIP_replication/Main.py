import json
import numpy as np
import os
import socket
import glob
import torch
import torchvision.transforms as transforms
import My_Resnet
from PIL import Image
from torch.autograd import Variable
from torchsummary import summary

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
    elif hostname.startswith('taiwan'): # maquina com GPU taiwan
         dirVideos = "/nfs/proc/rafael.padilla/" 
         outputDir = "/nfs/proc/rafael.padilla/"
    else:  # Path not defined
        raise Exception('Error: Folder with videos is not defined!')
    return dirVideos, outputDir

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
def get_file_filters(fold, types=['tar'], include_table_folder=False):
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

def get_feature_vector(image_path, layer_name, layers_and_sizes):
    # 1. Load the image with Pillow library
    img = Image.open(image_path)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(transformations(img).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    layer = layers_and_extractors[layer_name]
    feature_size = layers_and_sizes[layer_name]
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

# Lambda to insert "0" in 1-digit numbers (eg: 4->"04")
l_double_digit = lambda x : '0'+str(x) if len(str(x)) == 1 else str(x)

def get_corresponding_reference(target_path):
    ' Given a target frame file, obtain its corresponding reference.'
    folder, file = os.path.split(target_path)
    classe = 'neg'
    if '_ann' in file:
        classe = 'pos'
    file = file.replace('_ann','')
    # Get table number given the file name
    table_number = get_table_number(file)
    table_number_str = l_double_digit(table_number)
    # Get object number given the file name
    object_number = get_object_number(file)
    object_number_str = l_double_digit(object_number)
    # Get path (0 or 1)
    path_number = get_path_number(file)
    # Get frame number given the file name
    num_frame = get_frame_number(file)
    file_ref = 't%s_path%d_ref*_%d.png' % (table_number_str,path_number,num_frame)
    file_ref = os.path.join(folder,file_ref)
    file_ref = glob.glob(file_ref)
    assert len(file_ref) == 1
    file_ref = file_ref[0]
    ret = { 'reference_found_file' : file_ref,
            'class'          : classe,
            'table_number'   : table_number_str,
            'object_number'  : object_number_str,
            'frame_number'   : num_frame,
            'path_number'    : path_number }
    return ret

def get_table_number(frame_path):
    frame_path = frame_path[:frame_path.find('_')]
    return int(frame_path.replace('t',''))

def get_object_number(frame_path):
    frame_path = frame_path[frame_path.find('obj'):]
    frame_path = frame_path[:frame_path.find('_')]
    return int(frame_path.replace('obj',''))

def get_frame_number(frame_path):
    frame_path = frame_path.replace('_ann','')
    return int(frame_path[frame_path.rfind('_')+1:].replace('.png',''))

def get_path_number(frame_path):
    frame_path = frame_path[frame_path.find('path'):]
    frame_path = frame_path[:frame_path.find('_')]
    return int(frame_path.replace('path',''))
    
def is_frame_multiple_of(frame_path, value):
    'Given a frame file path, check if it is multiple of a certain value'
    frame_number = get_frame_number(frame_path)
    return True if frame_number%value == 0 else False

def generate_features(dir_read, dir_to_save_features, layer_name, frame_search_term_ref):
    if not os.path.isdir(dir_to_save_features):
        os.makedirs(dir_to_save_features)
    # Loop through search term
    for st in frame_search_term_ref:
        st = os.path.join(dir_read, st)
        # Obtain files matching the search term (st)
        files = glob.glob(st)
        for img_tar_file_path in files:
            if not img_tar_file_path.endswith('.png'):
                print('-> Error: not a recognized image file: %s' % file_name)
                continue
            # Only obtain feature map if it is multiple of 17 (skip every 14 frames)
            if not is_frame_multiple_of(img_tar_file_path, 17):
                continue
            # Find number of target frame and check if it is multiple of 17, in order to skip 17 frames
            print('file target: %s' % os.path.split(img_tar_file_path)[1])
            # Get feature map for target
            #feature_map_target = get_feature_vector(img_tar_file_path, layer_name, layers_and_sizes_224_398)
            # Get associated reference
            dict_ret = get_corresponding_reference(img_tar_file_path)
            file_ref = dict_ret['reference_found_file']
            classe = dict_ret['class']
            num_frame = dict_ret['frame_number']
            object_number = dict_ret['object_number']
            path_number = dict_ret['path_number']
            print('file reference: %s' % os.path.split(file_ref)[1])
            print('classe: %s | num_frame: %d | path: %d | object_number: %s' % (classe, num_frame, path_number, object_number))
            # Obtain and save the differences of features
            #feature_map_ref = get_feature_vector(file_ref, layer_name, layers_and_sizes_224_398)
            # Make sure both feature maps have the same shape
            #assert feature_map_ref.shape == feature_map_target.shape
            #diff = (feature_map_ref - feature_map_target).numpy()


            #path_to_save = 'feat_%s_diff_%s_frame_%s.npy' % (classe,layer_name,num_frame)
            path_to_save = 'feat_%s_diff_%s_obj%s_path%s_frame%s.npy' % (classe,layer_name, object_number, path_number,num_frame)
            
            #np.save(os.path.join(dir_to_save_features,path_to_save),diff)
            print('Feature %s sucessfully saved.' % path_to_save)
            print('-')
            return path_to_save


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

# Output size of each layer when input image is 3,224,398
layers_and_sizes_224_398 = {
    'conv1'     : (64,112,199),
    'residual1' : (256,56,100), # layer 1
    'residual2' : (256,56,100), # layer 1
    'residual3' : (256,56,100), # layer 1
    'residual4' : (512,28,50),  # layer 2 
    'residual5' : (512,28,50),  # layer 2 
    'residual6' : (512,28,50),  # layer 2 
    'residual7' : (512,28,50),  # layer 2 
    'residual8' : (1024,14,25),  # layer 3
    'residual9' : (1024,14,25),  # layer 3
    'residual10': (1024,14,25),  # layer 3
    'residual11': (1024,14,25),  # layer 3
    'residual12': (1024,14,25),  # layer 3
    'residual13': (1024,14,25),  # layer 3
    'residual14': (2048,7,13),  # layer 4
    'residual15': (2048,7,13),  # layer 4
    'residual16': (2048,7,13),  # layer 4
}

layers_and_extractors = {
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

layers_to_extract = ['conv1', 'residual1', 'residual2', 'residual3', 
                     'residual4', 'residual5', 'residual6', 'residual7', 
                     'residual8', 'residual9', 'residual10', 'residual11', 
                     'residual12', 'residual13', 'residual14', 'residual15', 
                     'residual16']

# Print network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# input_size = (3,720,1280)
# input_size = (3,224,398)
# summary(resnet50.to(device), input_size)
# Set model to evaluation mode
resnet50.eval()
# Set normalization to be used with pretrained weights
normalize = transforms.Normalize(mean=My_Resnet.mean_imagenet,
                                 std=My_Resnet.std_imagenet)
to_tensor = transforms.ToTensor()
# Smaller direction (height will be 224) (it will transform the image into width=398, height=224)
resize = transforms.Resize(224)
# Transformations
transformations = transforms.Compose([
        resize,
        transforms.ToTensor(),
        normalize])
# Get directories to read frames from and write feature maps
dir_read, dir_save = define_folders()
dir_read = os.path.join(dir_read,'shortest_distance_results','frames','*')
dir_save = os.path.join(dir_save,'shortest_distance_results','features')

folds_to_generate = ['fold_2']
_list = []
for fold_name in folds_to_generate:
    print('#'*80)
    print('Fold: %s (%s)' % (fold_name, folds_number[fold_name]))
    # Get reference 
    search_terms = get_file_filters(fold_name,['tar'])
    print('-'*80)
    [print('Search term: %s'%st) for st in search_terms]
    print('-'*80)
    for layer_name in layers_to_extract:
        # Define directory to save features
        dir_to_save_features = os.path.join(dir_save, fold_name, layer_name) 
        # Loop through layers to extract1
        print('Extracting features from layer: %s' % layer_name)
        print('-'*80)
        path_saved = generate_features(dir_read,dir_to_save_features,layer_name,search_terms)
        _list.append(path_saved)
        a = 123

