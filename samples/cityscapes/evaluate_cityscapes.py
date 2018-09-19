#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Mask R-CNN cityscapes inference

Using the trained model to detect and segment objects.'''
import os
import sys
import random
import colorsys
import skimage.io

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config

# get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import yaml
import PIL


class InferenceConfig(Config):
    """Configuration for training on the cityscapes dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # Backbone network architecture
    # Supported values are: resnet50, resnet101, mobilenet_v1
    BACKBONE = "resnet101"

    # A GPU with 12GB memory can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + cityscapes

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7

    # set dataset config
    WIDTH = 2048
    HEIGHT = 1024

    # save or not
    SAVE = True

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

        # Add classes
        self.add_class("cityscapes", 1, "person")
        self.add_class("cityscapes", 2, "rider")
        self.add_class("cityscapes", 3, "car")
        self.add_class("cityscapes", 4, "truck")
        self.add_class("cityscapes", 5, "bus")
        self.add_class("cityscapes", 6, "motorcycle")
        self.add_class("cityscapes", 7, "bicycle")

        id = 0  # the ids of images

        for image_path in filesImages:
            image_path_list = image_path.split('/')
            # print(image_path_list)
            image_path_list[4] = 'gtFine_test'
            label_path = '/'.join(image_path_list)
            mask_path = label_path.replace("_leftImg8bit.png",
                                           "_gtFine_instanceTrainIds_labelme.png")
            yaml_path = label_path.replace("_leftImg8bit.png",
                                           "_gtFine_instanceLabelmeNames.yaml")

            # print('mask_path:', mask_path)
            # print('yaml_path:', yaml_path)

            with open(yaml_path) as f:
                temp = yaml.load(f.read())
                labels = temp['label_names']
                del labels[0]

            if labels:
                self.add_image("cityscapes", image_id=id, path=image_path, width=width,
                               height=height, mask_path=mask_path, yaml_path=yaml_path)

                id = id + 1

            # print("image_path:",image_path)
            # print("mask_path:",mask_path)
            # print("yaml_path:",yaml_path)

        print("number of images:", id)

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

# Compute VOC-style Average Precision
def compute_batch_ap(dataset, config, image_ids):
    APs = []
    # Local path to trained weights file
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    CITYSCAPES_MODEL_PATH = os.path.join(MODEL_DIR, 'cityscapes20180626T1016',
                                         'mask_rcnn_cityscapes_0018.h5')

    '''Create Model and Load Trained Weights'''

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on cityscapes
    model.load_weights(CITYSCAPES_MODEL_PATH, by_name=True)
    print ("image_ids", image_ids)
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        print ("gt_class_id", gt_class_id)
        print ("gt_bbox", gt_bbox)
        print ("gt_mask", gt_mask)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    return APs

if __name__ == '__main__':

    # set the config and show
    config = InferenceConfig()
    config.display()

    # Directory of images to run detection on
    IMAGE_DIR = '/home/ason/datasets/leftImg8bit_test'
    # search files
    IMAGES = os.path.join(IMAGE_DIR, "val", "*", "*_leftImg8bit.png")
    images_evaluate = glob.glob(IMAGES)

    # Train dataset
    dataset_evaluate = CityscapesDataset()
    dataset_evaluate.load_cityscapes(config.HEIGHT, config.WIDTH, images_evaluate)

    # Must call before using the dataset
    dataset_evaluate.prepare()

    # Compute mAP @ IoU=50 on Batch of Images
    APs = compute_batch_ap(dataset_evaluate, config, dataset_evaluate.image_ids)
    print("mAP @ IoU=50: ", np.mean(APs))

