import os
import random

import numpy as np
import pandas as pd

import _init_paths
from blending import blend_iterative_blur
from generic_utils import get_files_paths, get_target_reference_frames
from my_enums import MethodToBlend
from ObjectHelper import ObjectDatabase
from VDAOVideo import VDAOVideo

random_seed = 123

####################################################################################################
# Step 1: Get only frames of reference videos that are not present in the testing (research) videos
####################################################################################################
vdao_dir = '/media/storage/VDAO/vdao_object'
# Get all all videos files from VDAO
all_videos_vdao = get_files_paths(vdao_dir, 'avi')
# Separate the reference reference videos
all_reference_videos = [v for v in all_videos_vdao if 'ref' in v]
# There are sections of the reference videos that are seen in test (research)
# We must disconsider those sections
target_reference_frames = get_target_reference_frames()
patches_to_disconsider = {}
for i in target_reference_frames:
    if i['reference file'] not in patches_to_disconsider:
        patches_to_disconsider[i['reference file']] = []
    # Adding 1 changes the reference from the annotation to the API's
    patches_to_disconsider[i['reference file']].append(
        (i['reference start frame'] + 1, i['reference final frame'] + 1))
# For each reference video, get the frames that are not used in the test (research)
frames_to_consider = {}
total_reference_frames = 0
for ref_vid in all_reference_videos:
    name_file = os.path.split(ref_vid)[-1]
    candidate_frames = list(range(1, VDAOVideo(ref_vid).videoInfo.getNumberOfFrames() + 1))
    total_reference_frames += len(candidate_frames)
    # Get the videos patches to disconsider and remove it from candidate_frames
    for patch in patches_to_disconsider.get(name_file, []):
        [
            candidate_frames.remove(i) for i in list(range(patch[0], patch[1] + 1))
            if i in candidate_frames
        ]
    frames_to_consider[ref_vid] = candidate_frames

print('There is a total of %d reference frames in the VDAO database.' % total_reference_frames)
total_frames_to_be_used = sum([len(k) for v, k in frames_to_consider.items()])
print('There is a total of %d reference frames that can be used in data augmentation.' %
      total_frames_to_be_used)
print('In other words, only %.2f%% of the reference frames are not presented in the test set.' %
      (100 * total_frames_to_be_used / total_reference_frames))

####################################################################################################
# Step 2: Generating data augmentation
####################################################################################################
# Load probability distributions disconsidering areas=0

current_path = os.path.dirname(os.path.realpath(__file__))
prob_distr_file = 'bounding_boxes_areas_distribution.csv'
csv = pd.read_csv(os.path.join(current_path, prob_distr_file))
areas = list(csv['área bounding box'])
occurencies = list(csv['ocorrências'])
total_occurencies = sum(occurencies)
distributions = [i / total_occurencies for i in occurencies]
group_samples_to_obtain = np.arange(1000, 101_000, 1000)
# Create dictionary with VDAOVideo objects
dict_ref_videos = {}
[dict_ref_videos.update({i: VDAOVideo(i)}) for i in all_reference_videos]
# Define aloi database path
root_samples_dir = '/media/storage/datasets/aloi'
for samples in group_samples_to_obtain:
    # choose a random reference video
    ref_video_path = random.choice(all_reference_videos)
    # choose a random frame
    ref_frame_number = random.choice(frames_to_consider[ref_video_path])

    # TODO:
    # Fora do loop definir tamanhos (dentro da distribuicao) pra pegar
    # Pegar objeto de referencia da ALOI
    # Pegar rotacao e posicao aleatoria
    # Chamar os métodos
    # Escrever em um arquivo txt as configuracoes do método

for video_path, list_frames in frames_to_consider.items():
    video = VDAOVideo(video_path)
    for frame_nbr in list_frames:
        retreived, background_image, shape = video.GetFrame(frame_nbr)
        assert retreived, f'Frame {frame_nbr} not found in {video}.'
        # Generate data samples (augmentation) for each approach
        image_path = os.path.join(root_samples_dir, 'png', '151_i110.png')
        mask_path = os.path.join(root_samples_dir, 'mask', '151_c1.png')
        a = blend_iterative_blur(image_path, mask_path, background_image)
        a = 123
