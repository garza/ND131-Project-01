#!/usr/bin/env python3
"""
 Example of inference with TensorFlow, not OpenVino
"""

import tarfile
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from pathlib import Path
##from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

import os
import sys
import logging as log
import numpy as np

from object_detection.utils import ops as utils_ops
##from object_detection.utils import label_map_util
##from object_detection.utils import visualization_utils as vis_util

class NetworkTF:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.model = None
        self.input_blob = None
        self.output_blob = None
        self.inference_result = None

    def load_model(self, model_name):
        log.info("TensorFlow Version: %s", tf.__version__)
        model_dir = model_name + "/saved_model"
        log.info("model_dir: %s", model_dir)
        self.model = tf.saved_model.load(str(model_dir))
        self.model = self.model.signatures['serving_default']
        return self.model

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        log.info("get_input_shape:")
        log.info(self.model.inputs);
        return self.model.inputs

    def get_output_name(self):
        return self.model.output_dtypes

    def get_output_info(self):
        return self.model.output_shapes

    def infer(self, image):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]
        output_dict = self.model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections
        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        self.inference_result = output_dict
        return

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.inference_result
