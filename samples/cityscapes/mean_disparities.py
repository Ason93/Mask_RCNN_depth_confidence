#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import glob
from skimage import io
import numpy as np

cityscapesPath_fine = '/home/ason/datasets/cityscapes_extra/disparity'
searchDisparities_fine = os.path.join(cityscapesPath_fine, "train", "*", "*_disparity.png")
# search files
filesDisparities_fine = glob.glob(searchDisparities_fine)
num_fine = len(filesDisparities_fine)
print("num_fine:", num_fine)

cityscapesPath_coarse = '/media/ason/文档1/Dataset/cityscapes/disparity_trainextra/disparity'
searchDisparities_coarse = os.path.join(cityscapesPath_coarse, "train_extra", "*", "*_disparity.png")
# search files
filesDisparities_coarse = glob.glob(searchDisparities_coarse)
num_coarse = len(filesDisparities_coarse)
print("num_coarse:", num_coarse)

filesDisparities_extra = filesDisparities_fine + filesDisparities_coarse
num_extra = len(filesDisparities_extra)
print("num_extra:", num_extra)

mean_disparities = []
max_disparities = []
min_disparities = []
for files in filesDisparities_extra:
    disparity = io.imread(files)

    max_depth = 255 #140
    depths = (disparity.astype(np.float32) - 1) / 256
    valid_pixels = (depths > 0.1)
    depths[valid_pixels] = max_depth - depths[valid_pixels]

    mean_disparity = np.mean(depths)
    mean_disparities.append(mean_disparity)

    max_disparity = np.max(depths)
    max_disparities.append(max_disparity)

    min_disparity = np.min(depths)
    min_disparities.append(min_disparity)

print("mean:", sum(mean_disparities)/num_extra)
print("max:", max(max_disparities))
print("min:", min(min_disparities))
