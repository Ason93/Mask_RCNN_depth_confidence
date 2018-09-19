#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:47:32 2018

@author: ason
"""

import cv2
import numpy as np

unchanage_img = cv2.imread('/media/ason/文档1/Dataset/cityscapes/disparity_trainvaltest/disparity/train/aachen/aachen_000000_000019_disparity.png',cv2.IMREAD_UNCHANGED)
#unchanage_img = cv2.imread('/home/ason/datasets/labelme_test/berlin_000000_000019_leftImg8bit_json/label.png',cv2.IMREAD_UNCHANGED)

num=np.unique(unchanage_img)
print('num:', num)
print('size:', len(num))