#!usr/bin/env python
# -*- coding:utf-8 -*-

'''Mask R-CNN coco inference

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
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # save or not
    SAVE = True


if __name__ == '__main__':

    # set the config and show
    config = InferenceConfig()
    config.display()

    # Local path to trained weights file
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')

    # Directory of images to run detection on
    IMAGE_DIR = '/home/ason/datasets/images'

    '''Create Model and Load Trained Weights'''

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on cityscapes
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    '''Run Object Detection'''

    # get the path of images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    id = 0 # image id

    # # Run detection
    all_start = time.time()

    for f in file_names:

        # Load images from the images folder
        input_path = os.path.abspath(os.path.join(IMAGE_DIR, f))
        image = skimage.io.imread(input_path)
        print("input_path:", input_path)

        detect = model.detect
        results = detect([image], verbose=1)

        r = results[0]

        # Save and show results
        if config.SAVE == True:
            # output file path
            output_path = input_path.replace("images", "images_results")
            print("output_path:", output_path)
            print('\n')
            visualize.save_instances(output_path = output_path, image=image,
                           boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'],
                           class_names=class_names, scores=r['scores'], title = "")
        # Show results only
        else:
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                       class_names, r['scores'])

        id += 1

    all_end = time.time()
    average_time = (all_end - all_start) / id
    print("inference runtime per image : {:.3f} s".format(average_time))

