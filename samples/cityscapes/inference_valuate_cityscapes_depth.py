#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Mask R-CNN cityscapes inference

Using the trained model to detect and segment objects.'''
import os
import sys
import colorsys
import numpy as np
import time
import glob
import yaml

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize

# Import cityscapes
from cityscapes import CityscapesDataset

# get_ipython().run_line_magic('matplotlib', 'inline')

class InferenceConfig(Config):
    """Configuration for inferencing on the cityscapes dataset.
    Derives from the base Config class and overrides some values.
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

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 7 classes

    # Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use bigger anchors because our image and objects are big
    # RPN_ANCHOR_SCALES = (8 * 8, 16 * 8, 32 * 8, 64 * 8, 128 * 8)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (8 * 4, 16 * 4, 32 * 4, 64 * 4, 128 * 4)  # anchor side in pixels

    # Image mean of cityscapes(RGB)
    MEAN_PIXEL = np.array([103.939, 116.779, 123.68])

    # set dataset config
    WIDTH = 2048
    HEIGHT = 1024

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

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
    SELECTED = True
    # Class Names
    CLASS_NAMES = ['BG', 'person', 'rider', 'car',
                   'truck', 'bus', 'motorcycle', 'bicycle']

    # save or not
    SAVE = True


class InferenceDataset(CityscapesDataset):

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
            image_path_list[5] = 'gtFine_test'
            label_path = '/'.join(image_path_list)
            mask_path = label_path.replace("_leftImg8bit.png",
                                           "_gtFine_instanceTrainIds_labelme.png")
            yaml_path = label_path.replace("_leftImg8bit.png",
                                           "_gtFine_instanceLabelmeNames.yaml")
            # print('id:', id)
            if config.ADD_DEPTH:
                image_path_list[5] = 'disparity_test'
                diaparity_path_dir = '/'.join(image_path_list)
                disparity_path = diaparity_path_dir.replace("_leftImg8bit.png",
                                                            "_disparity.png")
            # Make sure more than one instance in the image
            with open(yaml_path) as f:
                temp = yaml.load(f.read())
                labels = temp['label_names']
                del labels[0]

            if labels:
                if config.ADD_DEPTH:
                    self.add_image("cityscapes", image_id=id, path=image_path,
                                   width=width, height=height, mask_path=mask_path,
                                   yaml_path=yaml_path, disparity_path = disparity_path)
                else:
                    self.add_image("cityscapes", image_id=id, path=image_path, width=width,
                                   height=height, mask_path=mask_path, yaml_path=yaml_path)

                id = id + 1

            # print("image_path:",image_path)
            # print("mask_path:",mask_path)
            # print("yaml_path:",yaml_path)

        print("number of images:", id)

# Inferance and compute VOC-style Average Precision
def inference_evaluate(dataset, config, image_ids):

    # Local path to trained weights file
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    CITYSCAPES_MODEL_PATH = os.path.join(MODEL_DIR, 'cityscapes20180626T1016',
                                         'mask_rcnn_cityscapes_0018.h5')

    '''Create Model and Load Trained Weights'''

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on cityscapes
    model.load_weights(CITYSCAPES_MODEL_PATH, by_name=True)

    APs = []
    runtime_list = []
    for image_id in image_ids:

        if config.ADD_DEPTH:
            # Load GT, original image and disparity
            image, image_meta, gt_class_id, gt_bbox, gt_mask, disparity =\
                modellib.load_image_gt_depth(dataset, config,
                                       image_id, use_mini_mask=False)
        else:
            # Load GT and original image
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset, config,
                                       image_id, use_mini_mask=False)

        # Run detection and calculate runtime of each image
        if config.ADD_DEPTH:
            results, runtime = model.detect_disparity([image], [disparity], verbose=1)
        else:
            results, runtime = model.detect([image], verbose=1)
        r = results[0]
        runtime_list.append(runtime)

        class_names = InferenceConfig.CLASS_NAMES
        # Save results
        if config.SAVE == True:
            # output file path
            input_path = dataset.image_info[image_id]['path']
            output_path = input_path.replace("leftImg8bit_test",
                                             "leftImg8bit_test_result_disparity")
            print("output_path:", output_path)
            print('\n')

            visualize.save_instances(output_path=output_path, image=image,
                           boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'],
                           class_names=class_names, scores=r['scores'], figsize=(16, 16))
        # Show results
        else:
            visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'],
                                        class_ids=r['class_ids'], class_names=class_names,
                                        scores=r['scores'], figsize=(16, 16))

        # Compute mAP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                              r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)

    # Calculate and show the avarage runtime
    print("inference runtime per image : {:.3f} s".format(np.mean(runtime_list)))
    # Compute and show the average APs
    print("mAP @ IoU=50: ", np.mean(APs))


if __name__ == '__main__':

    # Set the config and show
    config = InferenceConfig()
    config.display()

    # Directory of images to run detection on
    IMAGE_DIR = '/home/ason/datasets/cityscapes_disparity_test/leftImg8bit_test'
    # Search files
    IMAGES = os.path.join(IMAGE_DIR, "val", "test", "*_leftImg8bit.png")
    images_evaluate = glob.glob(IMAGES)

    # Train dataset
    dataset_evaluate = InferenceDataset()
    dataset_evaluate.load_cityscapes(config.HEIGHT, config.WIDTH, images_evaluate)

    # Must call before using the dataset
    dataset_evaluate.prepare()

    # Compute and show mAP @ IoU=50 on Batch of Images
    inference_evaluate(dataset_evaluate, config, dataset_evaluate.image_ids)


