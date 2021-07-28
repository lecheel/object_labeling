# object_labeling
[Intel's OpenVINO Toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).

## Install OpenVINO Toolkit

Go to [OpenVINO HomePage](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

Download 2021.R4 version and install.

Follow the official Get Started Guides: https://docs.openvinotoolkit.org/latest/get_started_guides.html

## Set the Environment Variables

### Ubuntu:

```
/opt/intel/openvino_2021/bin/setupvars.sh
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

automation labeling via openvino yolo

![od_app](https://github.com/lecheel/object_labeling/blob/main/res/od_app1.jpg)
![od_lab](https://github.com/lecheel/object_labeling/blob/main/res/od_app2.jpg)
![labeling](https://github.com/lecheel/object_labeling/blob/main/res/labeling_shotcut.jpg)
