import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import _init_paths
from blending import blend_iterative_blur
from definitions import (aloi_paths, csv_file_distribution_areas, random_seed, vdao_videos_dir)
from generic_utils import get_files_paths, get_target_reference_frames
from my_enums import MethodToBlend
from ObjectHelper import ObjectDatabase
from VDAOVideo import VDAOVideo

####################################################################################################
# Step 1: Get only frames of reference videos that are not present in the testing (research) videos
####################################################################################################
vdao_dir = vdao_videos_dir['train']
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
# Step 2: Getting distribution of areas in the VDAO videos
####################################################################################################
# Load probability distributions disconsidering areas=0
csv = pd.read_csv(csv_file_distribution_areas)
areas = list(csv['área bounding box'])
occurencies = list(csv['ocorrências'])
total_occurencies = sum(occurencies)
distributions = [i / total_occurencies for i in occurencies]

####################################################################################################
# Step 3: Generating data augmentation
####################################################################################################
# Create dict with 100 groups of samples. Each group with 1000 data augmented images
# Areas of the augmented images have to follow the distribution of the original dataset
dict_sizes_per_group = {}
[
    dict_sizes_per_group.update({'%s' % i: np.random.choice(areas, 1000, p=distributions)})
    for i in range(1, 101)
]
# Create dictionary with VDAOVideo objects
dict_ref_videos = {}
[dict_ref_videos.update({i: VDAOVideo(i)}) for i in all_reference_videos]
width_background, height_background = (1280, 720)
# Get all possible ALOI images
aloi_images = get_files_paths(aloi_paths['images'], 'png')
width_aloi, height_aloi = (768, 576)

# For each one of the 100 groups create 1000 background with 1 aloi image inserted on each
for group, areas in dict_sizes_per_group.items():
    # For each area, get a random aloi object and a random background
    for area in areas:
        # choose a random ALOI image
        rand_aloi_img = random.choice(aloi_images)
        # choose a random reference video
        rand_ref_video_path = random.choice(all_reference_videos)
        # choose a random frame
        rand_ref_frame_number = random.choice(frames_to_consider[rand_ref_video_path])
        # get a random rotation angle
        rand_angle = random.randrange(0, 361, 1)
        # get random flip
        rand_flip = random.choice([True, False])
        # get random proportional factor
        rand_proportional_factor = np.random.random_sample()

# xIni=0,
# yIni=0,
# scale_factor=1,
# rotation_angle=0

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
