import os
import sys

import numpy as np

import _init_paths
from Align_Annotations_Research import define_tables


# Get files paths
def get_files_paths(folder, extension):
    ret = []
    for root, dirs, files in os.walk(folder):
        ret += [os.path.join(root, file) for file in files if file.endswith('.%s' % extension)]
    return ret


def get_target_reference_frames():
    ret = []
    tables_in_target_videos = define_tables()
    # Nao considerar table de 01 a 07, pois estes são os videos do Mateus que eu não tenho as referências
    tables_to_disconsider = ['table 0%d' % i for i in range(1, 8)]
    for table, info in tables_in_target_videos.items():
        if table in tables_to_disconsider:
            continue

        ret.append({
            'reference file': info['reference file'],
            'reference start frame': info['reference start frame'],
            'reference final frame': info['reference final frame']
        })
    return ret


def euclidean_distance(list1, list2):
    return np.linalg.norm(np.asarray(list1).astype(float) - np.asarray(list2).astype(float))
