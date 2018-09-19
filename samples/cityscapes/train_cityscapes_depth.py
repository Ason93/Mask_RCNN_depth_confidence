#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Mask R-CNN - Train on Cityscapes Dataset,
              backbone is mobilenet_v1, resenet101 or resnet50'''

import os
import sys
import numpy as np
import PIL
import yaml
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import cityscapes
from cityscapes import CityscapesDataset

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

'''[1]configuration'''


class CityscapesConfig(Config):
    """Configuration for training on the cityscapes dataset.
    Derives from the base Config class and overrides values specific
    to the cityscapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"

    # Backbone network architecture
    # Supported values are: resnet50, resnet101, mobilenet_v1
    BACKBONE = "resnet101"
    # set mobilenet alpha, rows and depth_multiplyer
    if BACKBONE == "mobilenet_v1":
        ALPHA = 1.0
        ROWS = 224
        DEP_MUL = 1

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Train or freeze batch normalization layers
    TRAIN_BN = True 

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 7 classes

    # Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use bigger anchors because our image and objects are big
    # RPN_ANCHOR_SCALES = (8 * 8, 16 * 8, 32 * 8, 64 * 8, 128 * 8)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (8 * 4, 16 * 4, 32 * 4, 64 * 4, 128 * 4)  # anchor side in pixels

    #    # Reduce training ROIs per image because the images are small and have
    #    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #    TRAIN_ROIS_PER_IMAGE = 32
    #
    #    # Use a small epoch since the data is simple
    #    STEPS_PER_EPOCH = 100
    #
    #    # use small validation steps since the epoch is small
    #    VALIDATION_STEPS = 5305

    # set dataset config
    WIDTH = 2048
    HEIGHT = 1024

    # split fine dataset into train and val
    VAL_SPLIT = 0.2
    # Whether use coarse dataset
    COARSE = False
    if COARSE:
        # split coarse dataset into train1, val1, train2, val2
        PART_SPLIT1 = 0.4
        PART_SPLIT2 = 0.5
        PART_SPLIT3 = 0.9

    # set show config
    SHOW_IMAGES = False
    if SHOW_IMAGES:
        SHOW_IMAGES_NUMBER = 4
        SHOW_MASKS_PER_IMAGE = 2

    # set which model init with
    # imagenet, coco, mobilenet_v1 or last
    INIT_WITH = "last"

    # Image mean of cityscapes(RGB)
    MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

    # Whether add depth features
    ADD_DEPTH = True
    if ADD_DEPTH:
        MEAN_DISPARITY = np.array([175.243])
        # disparity_fine 7530.4114081432435
        # disparity_coarse 7534.217726138363
        # disparity_extra 7533.724787080577
        # max disparity: 32257
        # min disparity: 0
        # mean inverse depth: 82.94117132687057   140
        # max inverse depth: 139.0                140
        # min inverse depth:: -0.00390625         140
        # mean: 175.2428347132419                 255
        # max: 254.0                              255
        # min: -0.00390625                        255
    SELECTED = False

'''[2]Dataset

Extend the Dataset class and add a method to load the cityscapes dataset,
 `load_cityscapes()`, and override the following methods:'''

class TrainDataset(CityscapesDataset):
    """The dataset consists of classes, masks.
    """

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

            if image_path_list[5] == 'leftImg8bit_extra':
                image_path_list[5] = 'gtCoarse'
                label_path = '/'.join(image_path_list)
                # print('label_path:', label_path)
                mask_path = label_path.replace("_leftImg8bit.png",
                                               "_gtCoarse_instanceTrainIds_labelme.png")
                yaml_path = label_path.replace("_leftImg8bit.png",
                                               "_gtCoarse_instanceLabelmeNames.yaml")

                if config.ADD_DEPTH:
                    image_path_list[5] = 'disparity_extra'
                    diaparity_path_dir = '/'.join(image_path_list)
                    disparity_path = diaparity_path_dir.replace("_leftImg8bit.png",
                                                   "_disparity.png")

            elif image_path_list[5] == 'leftImg8bit':
                image_path_list[5] = 'gtFine'
                label_path = '/'.join(image_path_list)
                mask_path = label_path.replace("_leftImg8bit.png",
                                               "_gtFine_instanceTrainIds_labelme.png")
                yaml_path = label_path.replace("_leftImg8bit.png",
                                               "_gtFine_instanceLabelmeNames.yaml")

#                print("image_path:",image_path)
#                print("mask_path:",mask_path)
#                print("yaml_path:",yaml_path)

                if config.ADD_DEPTH:
                    image_path_list[5] = 'disparity'
                    diaparity_path_dir = '/'.join(image_path_list)
                    disparity_path = diaparity_path_dir.replace("_leftImg8bit.png",
                                                   "_disparity.png")

            with open(yaml_path) as f:
                temp = yaml.load(f.read())
                labels = temp['label_names']
                del labels[0]

            if labels:
                if config.ADD_DEPTH:
                    self.add_image_disparity("cityscapes", image_id=id, path=image_path,
                                             disparity_path=disparity_path,
                                             width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)
                else:
                    self.add_image("cityscapes", image_id=id, path=image_path, width=width,
                                   height=height, mask_path=mask_path, yaml_path=yaml_path)

                id = id + 1

        print("number of images:", id)


if __name__ == '__main__':

    config = CityscapesConfig()
    config.display()

    cityscapesPath = '/home/ason/datasets/cityscapes_extra'

    '''--------------------------------coarse dataset-------------------------------'''
    if config.COARSE:
        searchImages_coarse = os.path.join(cityscapesPath, "leftImg8bit_extra", "train_extra", "*", "*_leftImg8bit.png")
        # search files
        filesImages_coarse = glob.glob(searchImages_coarse)

        # Split the dataset into four parts
        num_all = len(filesImages_fine)
        num_part1 = int(num_all * config.PART_SPLIT1)
        num_part2 = int(num_all * config.PART_SPLIT2)
        num_part3 = int(num_all * config.PART_SPLIT3)

        # Train dataset part1: for depth head
        dataset_train_coarse1 = TrainDataset()
        dataset_train_coarse1.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_coarse[:num_part1])
        dataset_train_coarse1.prepare()

        # Validation dataset part1: for depth head
        dataset_val_coarse1 = TrainDataset()
        dataset_val_coarse1.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_coarse[num_part1:num_part2])
        dataset_val_coarse1.prepare()

        # Train dataset part2: for all
        dataset_train_coarse2 = TrainDataset()
        dataset_train_coarse2.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_coarse[num_part2:num_part3])
        dataset_train_coarse2.prepare()

        # Validation dataset: for all
        dataset_val_coarse2 = TrainDataset()
        dataset_val_coarse2.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_coarse[num_part3:])
        dataset_val_coarse2.prepare()

    '''--------------------------------fine dataset-------------------------------'''

    searchImages_fine = os.path.join(cityscapesPath, "leftImg8bit", "train", "*", "*_leftImg8bit.png")
    # search files
    filesImages_fine = glob.glob(searchImages_fine)

    # split the dataset into train and val
    num_val = int(len(filesImages_fine) * config.VAL_SPLIT)
    num_train = len(filesImages_fine) - num_val

    # Train dataset
    dataset_train_fine = TrainDataset()
    dataset_train_fine.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_fine[:num_train])
    dataset_train_fine.prepare()

    # Validation dataset
    dataset_val_fine = TrainDataset()
    dataset_val_fine.load_cityscapes(config.HEIGHT, config.WIDTH, filesImages_fine[num_train:])
    dataset_val_fine.prepare()

    '''-------------------------Load and display random samples from fine dataset-------------------'''
    if config.SHOW_IMAGES:
        image_ids = np.random.choice(dataset_train_fine.image_ids, config.SHOW_IMAGES_NUMBER)
        for image_id in image_ids:
            image = dataset_train_fine.load_image(image_id)
            mask, class_ids = dataset_train_fine.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, dataset_train_fine.class_names,
                                        limit=config.SHOW_MASKS_PER_IMAGE)

    '''--------------------------Create model in training mode-----------------'''
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    '''--------------------------Which weights to start with-------------------'''
    # imagenet, coco or last

    if config.INIT_WITH == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)

    elif config.INIT_WITH == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

    elif config.INIT_WITH == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    '''Training

    # Train in two stages:
    # 1. Only the heads.
    # Here we're freezing all the backbone layers 
    # and training only the randomly initialized layers 
    # (i.e. the ones that we didn't use pre-trained weights from MS COCO).
    #  To train only the head layers, pass `layers='heads'` to the `train()` function.

    # 2. Fine-tune all layers.
    # For this simple example it's not necessary, 
    # but we're including it to show the process. 
    # Simply pass `layers="all` to train all layers.'''

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    if config.COARSE:
        model.train(dataset_train_coarse1, dataset_val_coarse1,
                    learning_rate=config.LEARNING_RATE,
                    epochs=5,
                    layers='depth')

        model.train(dataset_train_coarse2, dataset_val_coarse2,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=20,
                    layers='all')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train_fine, dataset_val_fine,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=30,
                layers="all")

    '''------------------Save weights-----------------------'''
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    model_path = os.path.join(MODEL_DIR, "mask_rcnn_cityscapes_depth.h5")
    model.keras_model.save_weights(model_path)

