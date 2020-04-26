#!/bin/sh

#CPU_EXT="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
#libinference_engine.dylib
#CPU_EXT="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
#CPU_EXT="/opt/intel/openvino_2020.2.117/deployment_tools/inference_engine/lib/intel64/libinference_engine.dylib"
#echo $CPU_EXT

#python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

/opt/local/bin/python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libinference_engine.dylib -d CPU -pt 0.6

#| /Users/garza/Development/ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
