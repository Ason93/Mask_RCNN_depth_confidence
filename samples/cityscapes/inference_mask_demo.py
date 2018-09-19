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

    # set dataset config
    WIDTH = 1280
    HEIGHT = 720

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.85

    # Whether add depth features
    ADD_DEPTH = False
    if ADD_DEPTH:
        MEAN_DISPARITY = np.array([7533.725])
        # disparity_fine 7530.4114081432435
        # disparity_coarse 7534.217726138363
        # disparity_extra 7533.724787080577

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
    IMAGE_DIR = '/home/dllinux/datasets/mask_demo'
    # Search files
    IMAGES = os.path.join(IMAGE_DIR, "underground", "*.png")
    files_images = glob.glob(IMAGES)

    '''Create Model and Load Trained Weights'''

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=IMAGE_DIR, config=config)

    # Load weights trained on cityscapes
    model.load_weights(CITYSCAPES_MODEL_PATH, by_name=True)

    # Class Names
    class_names = ['BG', 'person', 'rider', 'car',
                   'truck', 'bus', 'motorcycle', 'bicycle']

    '''Run Object Detection'''

    # get the path of images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]
    id = 0 # image id

    all_start = time.time()

    for f in files_images:

        # Load images from the images folder
        # input_path = os.path.abspath(os.path.join(IMAGE_DIR, f))
        image = skimage.io.imread(f)
        print("input_path:", f)

        # Run detection and calculate runtime of each image
        results, runtime = model.detect([image], verbose=1)

        r = results[0]

        # Save and show results
        if config.SAVE == True:
            # output file path
            output_path = f.replace("mask_demo", "mask_demo_result")
            print("output_path:", output_path)
            print('\n')
            visualize.save_instances(output_path = output_path, image=image,
                           boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'],
                           class_names=class_names, scores=r['scores'], figsize=(28, 16))
        # Show results only
        else:
            visualize.display_instances(image=image, boxes=r['rois'], masks=r['masks'],
                                        class_ids=r['class_ids'], class_names=class_names,
                                        scores=r['scores'], figsize=(28, 16))

        id += 1

    all_end = time.time()
    average_time = (all_end - all_start) / id
    print("inference runtime per image : {:.3f} s".format(average_time))
