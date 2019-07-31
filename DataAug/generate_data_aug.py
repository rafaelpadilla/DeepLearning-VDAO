import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import _init_paths
from blending import apply_transformations, blend_iterative_blur, rotate_image
from definitions import (aloi_paths, csv_file_distribution_areas, dir_save_images, random_seed,
                         vdao_videos_dir)
from generic_utils import get_files_paths, get_target_reference_frames
from my_enums import MethodToBlend
from VDAO_Access.ObjectHelper import ObjectDatabase
from VDAO_Access.VDAOVideo import VDAOVideo

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
areas_bb = list(csv['área bounding box'])
occurencies_bb = list(csv['ocorrências'])
# Disconsidering areas=0
areas_bb = areas_bb[1:]
occurencies_bb = occurencies_bb[1:]
total_occurencies_bb = sum(occurencies_bb)
distributions = [i / total_occurencies_bb for i in occurencies_bb]

####################################################################################################
# Step 3: Generating data augmentation
####################################################################################################
# Create dict with 100 groups of samples. Each group with 1000 data augmented images
# Areas of the augmented images have to follow the distribution of the original dataset
dict_sizes_per_group = {}
[
    dict_sizes_per_group.update({'%s' % i: np.random.choice(areas_bb, 1000, p=distributions)})
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
for group, rand_areas in dict_sizes_per_group.items():
    count_image = 0
    data_images_group = []
    # Create folder to save images
    folder_to_save_samples = os.path.join(dir_save_images, 'teste_%s' % group)
    if not os.path.isdir(folder_to_save_samples):
        os.makedirs(folder_to_save_samples)
    # For each area, get a random aloi object and a random background
    for area in rand_areas:
        # choose a random ALOI image
        rand_aloi_img = random.choice(aloi_images)
        # get its associated mask
        rand_aloi_mask = os.path.split(rand_aloi_img)[1].split('_')[0]
        rand_aloi_mask = os.path.join(
            os.path.split(rand_aloi_img)[0].replace(aloi_paths['images'], aloi_paths['masks']),
            rand_aloi_mask + '_c1.png')
        # choose a random frame from a random reference video
        rand_ref_video_path = random.choice(all_reference_videos)
        # rand_ref_video_path = '/media/storage/VDAO/vdao_object/table_04/Table_04-Reference_01/ref-sing-ext-part01-video01.avi'
        rand_ref_frame_number = random.choice(frames_to_consider[rand_ref_video_path])
        frame_background = dict_ref_videos[rand_ref_video_path].GetFrame(rand_ref_frame_number)[1]
        # get a random rotation angle
        rand_angle = random.randrange(0, 361, 1)
        # get random flip
        rand_flip = random.choice([True, False])
        # Como uma área pode aparecer com várias proporções (ex. área 133200 aparece em 24 diferentes
        # proporções), escolho uma proporção aleatória entre a mín e a máx para poder calcular a
        # altura (new_height)  e largura (new_width) que a imagem da ALOI será redimensionada. Desta
        # forma, tento manter a proporção do objeto verdadeiro da VDAO que contém aquela área escolhida.
        # get random proportional factor between min and max among all occurrences of this area
        proportions = csv.loc[csv['área bounding box'] == area]
        prop_min = float(proportions['prop min'].iloc[0].replace(',', '.'))
        prop_max = float(proportions['prop max'].iloc[0].replace(',', '.'))
        count_loop = 0
        while True:
            rand_proportion = np.random.uniform(low=prop_min, high=prop_max)
            # Calculate new height and width
            new_height = int(round(np.sqrt(area / rand_proportion)))
            new_width = int(round(area / new_height))
            if count_loop > 100:
                area = area - 1
            # get random position disconsidering proportions after rotating
            _, shape_new_size_rotation = apply_transformations(rand_aloi_img, new_height, new_width,
                                                               rand_angle, rand_flip)
            # Escolhe uma posição randomica dentro da imagem
            rand_pos_x, rand_pos_y = -1, -1
            while rand_pos_x < 0:
                rand_pos_x = int(np.random.uniform(0,
                                                   width_background - shape_new_size_rotation[1]))
            while rand_pos_y < 0:
                rand_pos_y = int(
                    np.random.uniform(0, height_background - shape_new_size_rotation[0]))
            if (rand_pos_x + new_width <= width_background) and (rand_pos_y + new_height <=
                                                                 height_background):
                break
            count_loop += 1

        # Gather all random data used for this image
        data_images_group.append({
            'count': count_image,
            'aloi_image': os.sep.join(rand_aloi_img.split(os.sep)[-3:]),
            'aloi_mask': os.sep.join(rand_aloi_mask.split(os.sep)[-3:]),
            'vdao_video': os.sep.join(rand_ref_video_path.split(os.sep)[-4:]),
            'frame_number': rand_ref_frame_number,
            'rot_angle': rand_angle,
            'initial_position': (rand_pos_x, rand_pos_y),
            'object_size': (new_height, new_width),
            'area': (new_height * new_width),
            'flip': rand_flip
        })

        novo_background, [min_x, min_y, max_x,
                          max_y], pos_mask = blend_iterative_blur(rand_aloi_img,
                                                                  rand_aloi_mask,
                                                                  frame_background,
                                                                  xIni=rand_pos_x,
                                                                  yIni=rand_pos_y,
                                                                  new_height=new_height,
                                                                  new_width=new_width,
                                                                  rotation_angle=rand_angle,
                                                                  flip_horizontally=rand_flip)

        count_image += 1
        frame_background[min_y:max_y, min_x:max_x, :] = novo_background
        frame_background = cv2.rectangle(frame_background, pos_mask[0], pos_mask[1], (0, 0, 255), 2)
        cv2.imwrite(os.path.join(folder_to_save_samples, 'sample_%s.png' % count_image),
                    frame_background)
        print('[Group %s]: sample_%s.png saved' % (group, count_image))
    pd.DataFrame(data_images_group).to_csv(os.path.join(folder_to_save_samples,
                                                        'group_%s.csv' % group),
                                           sep='\t',
                                           index=False,
                                           columns=[
                                               "count", "aloi_image", "aloi_mask", 'vdao_video',
                                               'frame_number', 'rot_angle', 'initial_position',
                                               'object_size', 'area', 'flip'
                                           ])
