#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

import zipfile
import urllib.request
import shutil
import PIL
import yaml

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

class CityscapesDataset(utils.Dataset):
    """The dataset consists of classes, masks.
    """

    # How many instances (objects) in the image
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # Analyze the yaml file obtained from labelme to
    # obtain the corresponding instance tag for each layer of the mask.
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # Rewrite draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_cityscapes(self, height, width, filesImages):
        pass

    def load_mask(self, image_id):

        info = self.image_info[image_id]
        count = 1  # number of object
        img = PIL.Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        ids_form = []
        for i in range(len(labels)):
            if labels[i].find("person") != -1:
                #                print ("person")
                labels_form.append("person")
                ids_form.append(1)
            elif labels[i].find("rider") != -1:
                #                print ("rider")
                labels_form.append("rider")
                ids_form.append(2)
            elif labels[i].find("car") != -1:
                #                print ("car")
                labels_form.append("car")
                ids_form.append(3)
            elif labels[i].find("truck") != -1:
                #                print ("truck")
                labels_form.append("truck")
                ids_form.append(4)
            elif labels[i].find("bus") != -1:
                #                print ("bus")
                labels_form.append("bus")
                ids_form.append(5)
            elif labels[i].find("motorcycle") != -1:
                #                print ("motorcycle")
                labels_form.append("motorcycle")
                ids_form.append(6)
            elif labels[i].find("bicycle") != -1:
                #                print ("bicycle")
                labels_form.append("bicycle")
                ids_form.append(7)

        class_ids = np.array([self.class_names.index(s) for s in labels_form])

        return mask, class_ids.astype(np.int32)