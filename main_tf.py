"""People Counter."""
"""
    TensorFlow version for comparison
"""

import os
import sys
import time
import socket
import json
import cv2

import logging as log
from time import time
import paho.mqtt.client as mqtt
import numpy as np

from argparse import ArgumentParser
from inference_tf import NetworkTF

from collections import defaultdict
from io import StringIO
##from matplotlib import pyplot as plt
from PIL import Image

from buffer import Buffer

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
PERSON_CLASS = 1
BUFFER_AVERAGE_CUTOFF = 0.25
IOU_THRESHOLD = 0.2

log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
##log.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    log.info('MQTT Connected')
    return client


def infer_on_stream(args, client, stats):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    category_index = []
    ##label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Initialise the class
    infer_network = NetworkTF()
    buffer = Buffer()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model)
    net_input_shape = infer_network.get_input_shape()
    net_output_name = infer_network.get_output_name()
    net_output_info = infer_network.get_output_info()
    log.info("network input")
    log.info(net_input_shape)
    log.info("network output info")
    log.info(net_output_info)

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    #steam input shape:
    width = int(cap.get(3))
    height = int(cap.get(4))
    input_width = 0
    input_height = 0
    total_person_count = 0
    duration = 0

    cur_request_id = 0
    next_request_id = 1
    render_time = 0
    parsing_time = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        start_time = time()
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        input_width = 300
        input_height = 300
        p_frame = cv2.resize(frame, (input_width, input_height))
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.infer(p_frame)
        inf_time = time() - start_time
        start_time = time()
        results = infer_network.get_output()
        image_np = np.array(frame)
        draw_boxes(
            inf_time,
            image_np,
            results['detection_boxes'],
            results['detection_classes'],
            results['detection_scores'],
            category_index,
            results.get('detection_masks_reframed', None)
            )
        render_time = time() - start_time
        stats.append(dict(it=inf_time, rt=render_time))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if key_pressed == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    ##client.disconnect()

def draw_boxes(inference_time, frame, boxes, classes, scores, index, masks):
    normalized = True
    thickness = 8
    det_label = "Person"
    ##log.info("draw-boxes: %s", boxes)
    for idx, b in enumerate(boxes):
        ##log.info("draw box!")
        ##log.info("draw-boxes-for: %s", b)
        ##box_info = box[0]

        ##color = (int(min((idx + 80) * 12.5, 255)),
        ##         min(b[5] * 7, 255), min(b[5] * 5, 255))
        ##cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
        ##cv2.putText(frame,
        ##            "#" + det_label + ' ' + str(round(b[4] * 100, 1)) + ' %',
        ##            (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        inf_time_message = "Inference time: {:.3f} ms".format(inference_time * 1e3)
        ##cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)


def report_stats(stats):
    stat_count = len(stats)
    inference_total = 0
    render_total = 0
    for i in stats:
        inference_total = inference_total + ( i['it'])
        ##log.info(i['it'] * 1e3)
        render_total = render_total + i['rt']
    render = render_total/stat_count
    inf = inference_total/stat_count
    log.info("stats count report %s", len(stats))
    log.info("Inference Average: {:.3f} ms".format(inf * 1e3))
    log.info("Render Average: {:.3f} ms".format(render * 1e3))
    return dict(inf=inf, render=render)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    ##client = connect_mqtt()
    # Perform inference on the input stream
    inference_stats = []
    infer_on_stream(args, None, inference_stats)
    report_stats(inference_stats)

if __name__ == '__main__':
    main()
