# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)

add_demo(NAME object_detection_2021r4
    SOURCES main.cpp
    HEADERS ${H_FILES}
    DEPENDENCIES monitors models pipelines)

add_demo(NAME vino_img_2021r4
    SOURCES ./vino_img.cpp
    HEADERS ${H_FILES}
    DEPENDENCIES monitors models pipelines)

