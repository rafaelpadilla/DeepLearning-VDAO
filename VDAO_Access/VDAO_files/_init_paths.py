###########################################################################################
#                                                                                         #
# Set up paths for VDAO Files                                                             #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: June 24th, 2018                                               #
###########################################################################################

import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, '..')
add_path(libPath)
libPath = os.path.join(currentPath, '..', '..')
add_path(libPath)
