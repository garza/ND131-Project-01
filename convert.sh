#!/bin/sh


## python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
## --input_model=/home/workspace/model/frozen_inference_graph.pb \
## --tensorflow_object_detection_api_pipeline_config /home/workspace/model/pipeline.config \
## --reverse_input_channels \
## --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json


python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
--input_model model-02/frozen_inference_graph.pb \
--tensorflow_object_detection_api_pipeline_config model-02/pipeline.config \
--reverse_input_channels \
--tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
