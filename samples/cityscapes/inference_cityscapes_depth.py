#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Mask R-CNN cityscapes inference

Using the trained model to detect and segment objects.'''
import os
import sys
import numpy as np
import time
import skimage.io
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize


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
    ADD_DEPTH = False
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
    # Class Names
    CLASS_NAMES = ['BG', 'person', 'rider', 'car',
                   'truck', 'bus', 'motorcycle', 'bicycle']

    # save or not
    SAVE = True

if __name__ == '__main__':

    # set the config and show
    config = InferenceConfig()
    config.display()

    # Local path to trained weights file
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    CITYSCAPES_MODEL_PATH = os.path.join(MODEL_DIR, 'cityscapes20180626T1016',
                                         'mask_rcnn_cityscapes_0018.h5')

    # Directory of images to run detection on
    IMAGE_DIR = '/home/ason/datasets/cityscapes_disparity_test/leftImg8bit_test'
    # Search files
    IMAGES = os.path.join(IMAGE_DIR, "val", "test", "*_leftImg8bit.png")
    images_evaluate = glob.glob(IMAGES)
    '''Create Model and Load Trained Weights'''

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on cityscapes
    model.load_weights(CITYSCAPES_MODEL_PATH, by_name=True)

    # Class Names
    class_names = ['BG', 'person', 'rider', 'car',
                   'truck', 'bus', 'motorcycle', 'bicycle']

    '''Run Object Detection'''

    runtime_list = []
    for image_path in images_evaluate:

        # Load images from the images folder
        image = skimage.io.imread(image_path)
        print("image_path:", image_path)

        # Run detection and calculate runtime of each image
        if config.ADD_DEPTH:
            disparity_path = image_path.replace("leftImg8bit_test", "disparity_test")
            disparity_path = disparity_path.replace("_leftImg8bit.png", "_disparity.png")
            disparity = skimage.io.imread(disparity_path)
            disparity = np.expand_dims(disparity, axis=-1)
            print("disparity_path:", disparity_path)
            results, runtime = model.detect_disparity([image], [disparity], verbose=1)
        else:
            results, runtime = model.detect([image], verbose=1)

        r = results[0]
        runtime_list.append(runtime)

        # Save and show results
        if config.SAVE == True:
            # output file path
            output_path = image_path.replace("leftImg8bit_test", "leftImg8bit_test_result_disparity")
            print("output_path:", output_path)
            print('\n')
            visualize.save_instances(output_path = output_path, image=image,
                           boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'],
                           class_names=class_names, scores=r['scores'], figsize=(32, 16))
        # Show results only
        else:
            visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'],
                                        class_ids=r['class_ids'], class_names=class_names,
                                        scores=r['scores'], figsize=(32, 16))

    # Calculate and show the avarage runtime
    print("inference runtime per image : {:.3f} s".format(np.mean(runtime_list)))
