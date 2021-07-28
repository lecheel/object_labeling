# object_labeling
[Intel's OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).

## Install OpenVINO Toolkit

Go to [OpenVINO HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Download 2021.R4 version and install.

Follow the official Get Started Guides: https://docs.openvinotoolkit.org/latest/get_started_guides.html

## Set the Environment Variables

### Ubuntu:

```
source /opt/intel/openvino_2021/bin/setupvars.sh
```

## build
```
mkdir build
cd build
cmake ..
make -j8
cd ..
./ops.sh
```

## download model
```
cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
python3 downloader.py --name person-vehicle-bike-detection-crossroad-yolov3-1020
python3 downloader.py --name yolo-v4-tf
python3 converter.py --name yolo-v4-tf
cp public/yolo-v4-tf/FP32/yolo-v4-tf.* $projdir
```

automation labeling via openvino yolo

![od_app](https://github.com/lecheel/object_labeling/blob/main/res/label.gif)
