#! /bin/bash
model=./person-vehicle-bike-detection-crossroad-yolov3-1020.xml
./vino_img_2021r4 -i ./out -m $model -labels classes.txt -at yolo -r
