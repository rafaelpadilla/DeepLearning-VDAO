import os
import socket

# Get local machine
hostname = socket.gethostname()
current_path = os.path.dirname(os.path.abspath(__file__))

random_seed = 123

if hostname == 'notesmt':
    BASE_DIR = '/media/storage/datasets/aloi'
    DATABASE_DIR = '/media/storage/VDAO'
elif 'smt.ufrj.br' in hostname:
    BASE_DIR = '/nfs/proc/rafael.padilla'
    DATABASE_DIR = '/home/rafael.padilla/workspace/rafael.padilla/'

vdao_videos_dir = {
    'train': os.path.join(DATABASE_DIR, 'vdao_object'),
    'test': os.path.join(DATABASE_DIR, 'vdao_research')
}

csv_file_distribution_areas = os.path.join(current_path, '..',
                                           'bounding_boxes_areas_distribution.csv')

aloi_root_path = os.path.join(BASE_DIR, 'aloi')
aloi_paths = {
    'images': os.path.join(aloi_root_path, 'png'),
    'masks': os.path.join(aloi_root_path, 'mask')
}
