#!/bin/sh

#for macos, no CPU extension is given, or at least I couldn't find the right one to reference, when none is given
#OpenVINO by default will assume CPU

#also note, on MacOS, invocations from a sh script does not also create a context where the DYLIB Path is also set, so you must use the command line

python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -d CPU -pt 0.6 | /Users/garza/Development/ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

python3 main.py -i resources/test.jpg -m frozen_inference_graph.xml -d CPU -pt 0.6 | /Users/garza/Development/ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


#/opt/local/bin/python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libinference_engine.dylib -d CPU -pt 0.6

#| /Users/garza/Development/ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
