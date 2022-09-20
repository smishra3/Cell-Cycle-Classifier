import sys
import os
import numpy as np
import math
import shutil

dir_path_temp = '/allen/aics/assay-dev/users/Suraj/PCNA_Classification/data_2d_im2lb0_bb/'
tar_path = '/allen/aics/assay-dev/users/Suraj/PCNA_Classification/data_2d_im2lb0_bb_pro/'

#dir_path = '/allen/aics/assay-dev/users/Suraj/PCNA_Classification/data_3d_im2lb0/lSG2/'
#tar_path = '/allen/aics/assay-dev/users/Suraj/PCNA_Classification/data_3d_im2lb0_pro/'

#cv0 = train < 0.5; 0.5 < test < 0.8; 0.8 < val < 1.0 
#cv1 = 0.25 < train < 0.75; 0.75 < test < 1.0 & test < 0.05; 0.05 < val < 0.25
#cv2 = 0.5 < train < 1.0; test < 0.3; 0.3 < val < 0.5
#cv3 = 0.75 < train < 1.0 & train < 0.25; 0.25 < test < 0.55; 0.55 < val < 0.75

##############################################################################
count = 0
class_idx = ['G1', 'G2', 'M', 'eS', 'eSmS', 'mS', 'mSlS', 'lS', 'lSG2']

for idx in class_idx:
    print(idx)
    count = 0
    dir_path = dir_path_temp + idx + '/'
    for subdir in sorted(os.listdir(dir_path)):
        subdir_path = os.path.join(dir_path, subdir)
        count = count+1
        if (count <= math.ceil(len(os.listdir(dir_path))*0.5)):
            dest = os.path.join(tar_path, 'train', idx, subdir)
            shutil.copy(subdir_path,dest)
            print("train %d" % (count))
        elif (count > math.ceil(len(os.listdir(dir_path))*0.5)) and (count <= math.ceil(len(os.listdir(dir_path))*0.8)):
            dest = os.path.join(tar_path, 'test', idx, subdir)
            shutil.copy(subdir_path,dest)
            print("test %d" % (count))
        elif (count > math.ceil(len(os.listdir(dir_path))*0.8)) and (count <= math.ceil(len(os.listdir(dir_path))*1)):
            dest = os.path.join(tar_path, 'val', idx, subdir)
            shutil.copy(subdir_path,dest)
            print("val %d" % (count))
        else:
            print("error detected #######  %d" % (count))

