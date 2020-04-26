"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
from inference import Network
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
    # Initialise the class
    infer_network = Network()
    buffer = Buffer()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    ##net_input_shape = [1, 3, 600, 600]
    net_output_name = infer_network.get_output_name()
    net_input_name = infer_network.get_input_blob_name()
    net_input_shape = infer_network.get_input_shape()
    net_output_info = infer_network.get_output_info()
    log.info("network output name")
    log.info(net_output_name)
    log.info("network output info")
    log.info(net_output_info.shape)
    log.info("network input shape")
    log.info(net_input_name)
    log.info(net_input_shape)
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
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        input_width = net_input_shape[2]
        input_height = net_input_shape[3]
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        start_time = time()
        infer_network.exec_net(p_frame)
        render_time = 0
        inf_time = 0

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            inf_time = time() - start_time
            ###restart clock to capture evaluate/draw time
            start_time = time()
            boxes = post_process(result, width, height, PERSON_CLASS)
            ##if len(boxes) > 1:
                ##log.info("initial boxes: %s", boxes)
            boxes = list(boxes.values())
            boxes = nms(boxes)
            buffer_avg = 0

            if len(boxes) > 0:
                ##we have a person in frame (maybe)
                first_prop = boxes[0]
                confidence = first_prop[4]
                buffer.add(confidence)
                buffer_avg = buffer.average()
                if confidence > args.prob_threshold:
                    if duration > 0:
                        ##this is not the first time they have been in the frame
                        ##increase duration and move along
                        duration = duration + 1
                    else:
                        ##very first time this person has entered the frame
                        ##pulse out new count
                        total_person_count = total_person_count + 1
                        duration = duration + 1
                    client.publish("person", json.dumps({"count": 1, "total": total_person_count}))
                    draw_box(frame, boxes, inf_time)
                else:
                    ##we have a person in frame, but they don't meet confidence threshold
                    if duration > 0:
                        ##we know we were tracking someone last frame
                        ##so check our rolling buffer average
                        if buffer_avg > BUFFER_AVERAGE_CUTOFF:
                            ##same person, keep counting, move along
                            duration = duration + 1
                            client.publish("person", json.dumps({"count": 1, "total": total_person_count}))
                            draw_box(frame, boxes, inf_time)
                        else:
                            ##log.info("NO-DRAW: c:%s, b:%s, d:%s : else:if:else", confidence, buffer_avg, duration)
                            ##no longer meet confidence or buffer avg
                            client.publish("person", json.dumps({"count": 0, "total": total_person_count}))
                            client.publish("person/duration", json.dumps({"duration": duration}))
                            duration = 0
                            buffer.flush()
                    else:
                        ##log.info("NO-DRAW: c:%s, b:%s, d:%s : else:else", confidence, buffer_avg, duration)
                        ##also nobody in the last frame (duration == 0)
                        client.publish("person", json.dumps({"count": 0, "total": total_person_count}))
            else:
                ##no boxes with our target class was found, make sure we didn't see one in the last frame (or so)
                buffer.add(0)
                buffer_avg = buffer.average()
                if buffer_avg > BUFFER_AVERAGE_CUTOFF:
                    ##we has someone previously, keep counting, move along
                    duration = duration + 1
                else:
                    ##nobody previously, nobody now, make sure we say so
                    client.publish("person", json.dumps({"count": 0, "total": total_person_count}))
                    if duration > 0:
                        ##we were previously tracking someone, pulse out duration before zeroing out
                        client.publish("person/duration", json.dumps({"duration": duration}))
                        duration = 0

            render_time = time() - start_time
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1e3)
            cv2.putText(frame, render_time_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            stats.append(dict(it=inf_time, rt=render_time))
            ### TODO: Extract any desired stats from the results ###
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if key_pressed == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def iou(box_1, box_2):
    width_overlap = min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])
    height_overlap = min(box_1[3], box_2[3]) - max(box_1[1], box_2[1])
    if width_overlap < 0 or height_overlap < 0:
        overlap_area = 0
    else:
        overlap_area = width_overlap * height_overlap
    area1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    union_area = area1 + area2 - overlap_area
    if union_area == 0:
        return 0
    return overlap_area / union_area

## first try at implementing nonmax supression
def nms(boxes):
    reduced = []
    if len(boxes) > 0:
        ##log.info("nms 4 starting with boxes: %s", boxes)
        boxes = sorted(boxes, key=lambda box : box[4], reverse=True)
        reduced.append(boxes[0])
        for i in range(len(boxes)):
            if boxes[i][4] == 0:
                continue
            for j in range(i + 1, len(boxes)):
                thisIOU = iou(boxes[i], boxes[j])
                if thisIOU < IOU_THRESHOLD:
                    log.info("throwing out index: %s, %s", i, j)
                    reduced.append(boxes[j])
    ##log.info("nms - returning boxes: %s, %s", boxes, reduced)
    return reduced

def post_process(result, width, height, class_filter):
    boxes = {}
    iw = width
    ih = height
    data = result[0][0]
    for number, proposal in enumerate(data):
        ##log.info("proposal imid: %s", number)
        if proposal[2] > 0:
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            if proposal[1] == class_filter:
                if not number in boxes.keys():
                    boxes[number] = []
                    boxes[number] = [xmin, ymin, xmax, ymax, confidence, label]
    return boxes

def draw_box(frame, boxes, inference_time):
    det_label = "Person"
    ##log.info("draw-boxes: %s", boxes)
    for idx, b in enumerate(boxes):
        ##log.info("draw-boxes-for: %s", b)
        ##box_info = box[0]
        color = (int(min((idx + 80) * 12.5, 255)),
                 min(b[5] * 7, 255), min(b[5] * 5, 255))
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
        cv2.putText(frame,
                    "#" + det_label + ' ' + str(round(b[4] * 100, 1)) + ' %',
                    (b[0], b[1] + 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)
        inf_time_message = "Inference time: {:.3f} ms".format(inference_time * 1e3)
        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

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
    client = connect_mqtt()
    # Perform inference on the input stream
    inference_stats = []
    infer_on_stream(args, client, inference_stats)
    report_stats(inference_stats)

if __name__ == '__main__':
    main()
