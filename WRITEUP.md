# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

One I had the project working with OpenVINO and a model that I had converted from TensorFlow's Model Garden, I created a new version (main_tf.py and inference_tf.py) that instead used TensorFlow to evaluate each frame in order to compare inference speed and and load on resources.  I found the documentation and resources at the TensorFlow Model Garden to be far superior to those found at the Caffee Model Zoo and ONNX model repositories.

Three models used for comparison and benchmarking:

- [ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)
- [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
- [ssd_mobilenet_v2_coco_2018_03_29](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)

| Model Name | Performance Before IR Conversion | Performance Time After IR Conversion  | Size |
| - |:-:| -:| -:|
| ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03 | 39ms | 14ms | 13MB
| faster_rcnn_inception_v2_coco_2018_01_28 | 350ms      |   200ms | 51MB
| ssd_mobilenet_v2_coco_2018_03_29 | 60ms | 20ms | 64MB

## SSD MobileNet v1 PPN Shared Box ##
This model, found in TensorFlow Model Garden turn out to be the best performaing.  While the Faster RCNN Inception model was very accurate, often returning confidences in the higher 90%s when a person was in the frame, it performed slower by over a factor of 10.

In comparison, the SSD MobileNet v2 model was just as fast, but came with an impact in accuracy.  Even with a lower threshold of 0.3 it often failed to find people in the frame when one existed leading to an overcount of total people in the test clip (on average 14-15 people detected).

## Assess Model Use Cases

Being able to detect the presence of a person in a video feed or stream can have a multitude of use cases.  For example, it could be used in security to count the ingress and egress into a facility or secure room where the application of authenticated security might not be viable (keycards or key locks).  This is useful because it could augment existing security practices (having a security feed monitored by a professional) and help lessen the burden of monitoring if a facility is using multiple camera feeds.

In retail business, there is often a need to see which products can capture the attention of potential customers, so this type of model can be used to compare the effectiveness (by using average duration) of a physical marketing display or even to evaluate the optimal layout of a grocery shelf ("placing product A on the top shelf resulted in customers doubling their duration in that aisle").

For customer satisfaction, the people counter + duration could also be a way of measuring the effectiveness of a in-person help desk or customer service desk at a store front.  For example, monitoring the average duration of customers at a service desk for a department store would give the company an idea of how many people were having issues that required service desk expertise, and how long or quick employees were able to service each customer.

Although our test clip only had at most one person in the frame, with a wider viewing angle, another application of this model could be to measure distances between people or even a room's occupancy level.  With the arrival of the COVID-19 pandemic, having a way to monitor a business's ability to ensure customers or employees follow social distancing rules could be another application if we can reliably calculate distance between two distinct people identified by this model.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Some of the models tested had a particularly hard time recognizing people in full long sleve and pants of the same color (usually black clothing).  Another point is the test video was well lit and the camera focal length was locked.  We were unable to test to see if this model was as accurate in poorer lighting or at different viewing angles of the same room.  Also, the diversity of the pool of model people used in this clip was very homogenous.  All seemed to be caucasian middle aged people in casual business attire.  Further testing would want to ensure a more diverse model pool that more closely matched the type of people that would normall be in frame after deployment.

## Model Research and Conversion

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
  - [ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --input_model model-02/frozen_inference_graph.pb \
    --tensorflow_object_detection_api_pipeline_config model-02/pipeline.config \
    --reverse_input_channels \
    --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

    ```
  - The model ended up being the best model evaluated with a good middle ground and the least amount of frame dropping (where the model did not find any people in frame when a person was still there)
  - I tried to improve the model for the app by...
    - implementing a rolling buffer (buffer.py) of confidence to make sure any dropouts did not result in extra counting or duration bouncing in main.py

- Model 2: [faster_rcnn_inception_v2_coco_2018_01_28]
  - [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments...

  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
  --input_model model-03/frozen_inference_graph.pb \
  --tensorflow_object_detection_api_pipeline_config model-03/pipeline.config \
   --output=detection_boxes,detection_scores,num_detections \
   --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
   ```
  - The model was insufficient for the app because while the confidence numbers were quite accurate, the time for performing an inference was around 200ms making the resulting media stream to the UI too slow
  - I attempted to improve this model by tweaking the input shape of the image required by the model using the Model Optimizer's --input_shape parameter.  However, this flag only allows you to make the shape larger, not smaller and the smallest size defined by the model was 600x600 (twice as big as Model 1)

- Model 3: [ssd_mobilenet_v2_coco_2018_03_29]
  - [ssd_mobilenet_v2_coco_2018_03_29](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments...

  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
  --input_model model-04/frozen_inference_graph.pb \
  --tensorflow_object_detection_api_pipeline_config model-04/pipeline.config \
   --output=detection_boxes,detection_scores,num_detections \
   --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
   ```
  - The model was insufficient for the app because accuracy for when people were in the frame were too low, even when using a probabilty threshold of 0.25 and a 10 frame rolling buffer, I still got frequent dropouts resulting in double counting several of the same people and inaccurate duration calculations.


## Sources Used For Code ##

I relied heavily on the projects from the initial foundation course for alot of the code in this project, as well as the example python demos distributed by OpenVINO:
  - ND132 Lesson 2 - Exercise 1 - Convert A Tensorflow Model
  - ND132 Lesson 3 - Exercise 4 - Custom Layers
  - ND132 Lesson 5 - Exercise 1 - Handling Input Streams
  - ND132 Lesson 5 - Exercise 3 - Server Communications

Python Demo Code:
 - [object_detection_demo_ssd_async](https://docs.openvinotoolkit.org/2020.1/_inference_engine_ie_bridges_python_sample_object_detection_sample_ssd_README.html)
 - [object_detection_demo_yolo3_async](https://docs.openvinotoolkit.org/2020.1/_demos_python_demos_object_detection_demo_yolov3_async_README.html)

When creating the TensorFlow test harness for analyzing inference speed before using an IR model, the example code at the TensorFlow Model Garden helped immensely in getting up to speed and inferring result locally:

When implementing Nonmax Supression and Intersection Over Union (IOU) on my resulting bounding boxes, I looked for alot of similar examples of this function, some that came in extremely handy was:

- [Object Detection With YOLO](http://machinethink.net/blog/object-detection-with-yolo/)
- [object_detection_demo_yolo3_async](https://docs.openvinotoolkit.org/2020.1/_demos_python_demos_object_detection_demo_yolov3_async_README.html)
