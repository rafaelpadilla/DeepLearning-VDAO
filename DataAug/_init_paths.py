###########################################################################################
#                                                                                         #
# Set up paths for Video Alignment                                                        #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: July 18tj, 2019                                               #
###########################################################################################

import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, '..', '..', 'Video-Alignment')
add_path(libPath)
libPath = os.path.join(currentPath, 'utils')
add_path(libPath)
libPath = os.path.join(currentPath, '..', 'VDAO_Access')
libPath = os.path.join(currentPath, '..', 'VDAO_Access')
add_path(libPath)
