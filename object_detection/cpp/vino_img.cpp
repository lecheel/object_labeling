/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/args_helper.hpp>
#include <utils/slog.hpp>
#include <utils/images_capture.h>
#include <utils/default_flags.hpp>
#include <utils/performance_metrics.hpp>
#include <unordered_map>
#include <gflags/gflags.h>

#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_faceboxes.h>
#include <models/detection_model_retinaface.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <glob.h>
#include <zmqpp/zmqpp.hpp>
#include <chrono>
#include <pthread.h>

using namespace std;

#define ZMQ 1

std::vector<std::string> img_files; 
std::vector<std::string> lab_imgs; 
std::vector<std::string> pub_info; 


DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] = "Required. Architecture type: centernet, faceboxes, retinaface, ssd or yolo";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] = "Optional. Specify the target device to infer on (the list of available devices is shown below). "
"Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
"The demo will look for a suitable plugin for a specified device.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char performance_counter_message[] = "Optional. Enables per-layer performance report.";
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "
"Absolute path to the .xml file with the kernel descriptions.";
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
"Absolute path to a shared library with the kernel implementations.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char input_resizable_message[] = "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
"throughput mode (for HETERO and MULTI device cases use format "
"<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char iou_thresh_output_message[] = "Optional. Filtering intersection over union threshold for overlapping boxes.";
static const char yolo_af_message[] = "Optional. Use advanced postprocessing/filtering algorithm for YOLO.";
static const char output_resolution_message[] = "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(labels, "", labels_message);
DEFINE_bool(pc, false, performance_counter_message);
DEFINE_string(c, "", custom_cldnn_message);
DEFINE_string(l, "", custom_cpu_library_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_double(iou_t, 0.5, iou_thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(yolo_af, true, yolo_af_message);
DEFINE_string(output_resolution, "", output_resolution_message);

/**
* \brief This function shows a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i                        " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -pc                       " << performance_counter_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -iou_t                    " << iou_thresh_output_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -yolo_af                  " << yolo_af_message << std::endl;
}

class ColorPalette {
private:
    std::vector<cv::Scalar> palette;

    static double getRandom(double a = 0.0, double b = 1.0) {
        static std::default_random_engine e;
        std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
        return dis(e);
    }

    static double distance(const cv::Scalar& c1, const cv::Scalar& c2) {
        auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
        auto ds = std::fabs(c1[1] - c2[1]);
        auto dv = std::fabs(c1[2] - c2[2]);

        return dh * dh + ds * ds + dv * dv;
    }

    static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet, const std::vector<cv::Scalar>& colorCandidates) {
        std::vector<double> distances;
        distances.reserve(colorCandidates.size());
        for (auto& c1 : colorCandidates) {
            auto min = *std::min_element(colorSet.begin(), colorSet.end(),
                [&c1](const cv::Scalar& a, const cv::Scalar& b) { return distance(c1, a) < distance(c1, b); });
            distances.push_back(distance(c1, min));
        }
        auto max = std::max_element(distances.begin(), distances.end());
        return colorCandidates[std::distance(distances.begin(), max)];
    }

    static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor) {
        cv::Mat rgb;
        cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
    }

public:
    explicit ColorPalette(size_t n) {
        palette.reserve(n);
        std::vector<cv::Scalar> hsvColors(1, { 1., 1., 1. });
        std::vector<cv::Scalar> colorCandidates;
        size_t numCandidates = 100;

        hsvColors.reserve(n);
        colorCandidates.resize(numCandidates);
        for (size_t i = 1; i < n; ++i) {
            std::generate(colorCandidates.begin(), colorCandidates.end(),
                []() { return cv::Scalar{ getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0) }; });
            hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
        }

        for (auto& hsv : hsvColors) {
            // Convert to OpenCV HSV format
            hsv[0] *= 179;
            hsv[1] *= 255;
            hsv[2] *= 255;

            palette.push_back(hsv2rgb(hsv));
        }
    }

    const cv::Scalar& operator[] (size_t index) const {
        return palette[index % palette.size()];
    }
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result, const ColorPalette& palette, OutputTransform& outputTransform, int idx) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image

    std::ostringstream yolo_txt;
    std::string txt="";
    for (auto& obj : result.objects) {
        if (FLAGS_r) {
            slog::info << " "
                << std::left << std::setw(9) << obj.label << " | "
                << std::setw(10) << obj.confidence << " | "
                << std::setw(4) << std::max(int(obj.x), 0) << " | "
                << std::setw(4) << std::max(int(obj.y), 0) << " | "
                << std::setw(4) << std::min(int(obj.x + obj.width), outputImg.cols) << " | "
                << std::setw(4) << std::min(int(obj.y + obj.height), outputImg.rows)
                << slog::endl;
        }

        int my=std::max(int(obj.y), 0);
        int mx=std::max(int(obj.x), 0);
        int mw=int(obj.width);
        int mh=int(obj.height);

        float cx = float((mx + mx+mw)) / 2 / outputImg.cols;
        float cy = float((my + my+mh)) / 2 / outputImg.rows;
        float yw = float(mw) / outputImg.cols;
        float yh = float(mh) / outputImg.rows;

        outputTransform.scaleRect(obj);
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        auto color = palette[obj.labelID];
        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, { 230, 230, 230 }, 3);
        cv::putText(outputImg, obj.label + conf.str(),
            cv::Point2f(obj.x, obj.y - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
        cv::rectangle(outputImg, obj, color, 2);
        yolo_txt << std::fixed << std::setprecision(5) << obj.labelID << " " << cx << " " << cy << " " << yw << " " << yh << endl;
    }
    txt.append(yolo_txt.str());
    if (yolo_txt.str().size()>0)
        pub_info.push_back(yolo_txt.str());

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            outputTransform.scaleCoord(lmark);
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    }
    catch (const std::bad_cast&) {}

    return outputImg;
}

std::vector<std::string> glob(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}

string getFileName(const string& s) {
  char sep = '/';
#ifdef _WIN32
  sep = '\\';
#endif
  size_t i = s.rfind(sep, s.length());
  if (i != string::npos) 
  {
    string filename = s.substr(i+1, s.length() - i);
    size_t lastindex = filename.find_last_of("."); 
    string rawname = filename.substr(0, lastindex); 
    return(rawname);
  }

  return("");
}


void *zmq_labinfo(void *argument) {

    zmqpp::context context;
    zmqpp::socket_type type = zmqpp::socket_type::subscribe;
    zmqpp::socket socket_sub(context, type);

    socket_sub.subscribe("");
    socket_sub.connect("tcp://localhost:5208");
    cout << "Wait for zmq tcp://localhost:5208" << endl;
    string text;

    while(true) {
        // Receive (blocking call)
        zmqpp::message message;
        socket_sub.receive(message);

        message >> text;

        unsigned long ms = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
        cout << "[RECV] at " << ms << ": \"" << text << "\"" << endl;
        lab_imgs.push_back(text);
    }
}

void *zmq_pubinfo(void *argument) {
    string xxx;
    //const string endpoint = "tcp://localhost:5209";
    const string endpoint = "tcp://*:5209";

    // Create a publisher socket
    zmqpp::context context;
    zmqpp::socket_type type = zmqpp::socket_type::publish;
    zmqpp::socket socket_pub (context, type);

    // Open the connection
    cout << "\033[92mBinding to tcp://*:5209\033[0m" << endl;
    socket_pub.bind(endpoint);
    //socket_pub.connect(endpoint);

    while (true) {
        if (pub_info.empty())
        {
            this_thread::sleep_for(chrono::milliseconds(1) );
        } else {

            xxx = pub_info.back();
            pub_info.pop_back();
            zmqpp::message message;
            message << xxx;
            socket_pub.send(message);
        }
    }
    cout << "\033[91mshoud not coming here pubinfo" << endl;

}


int main(int argc, char *argv[]) {

    pthread_t thread1,thread2;
    pthread_create( &thread1, NULL, zmq_labinfo, (void*) "thread 1");
    pthread_create( &thread2, NULL, zmq_pubinfo, (void*) "thread 2");

    try {
        PerformanceMetrics metrics;

        slog::info << "InferenceEngine: " << printable(*InferenceEngine::GetInferenceEngineVersion()) << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        slog::info << "Reading input" << slog::endl;


        //img_files = glob(FLAGS_i+"/*.jpg");
        cv::Mat curr_frame;

        //------------------------------ Running Detection routines ----------------------------------------------
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty())
            labels = DetectionModel::loadLabels(FLAGS_labels);
        ColorPalette palette(labels.size() > 0 ? labels.size() : 100);

        std::unique_ptr<ModelBase> model;
        if (FLAGS_at == "centernet") {
            model.reset(new ModelCenterNet(FLAGS_m, (float)FLAGS_t, labels));
        }
        else if (FLAGS_at == "faceboxes") {
            model.reset(new ModelFaceBoxes(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, (float)FLAGS_iou_t));
        }
        else if (FLAGS_at == "retinaface") {
            model.reset(new ModelRetinaFace(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, (float)FLAGS_iou_t));
        }
        else if (FLAGS_at == "ssd") {
            model.reset(new ModelSSD(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, labels));
        }
        else if (FLAGS_at == "yolo") {
            model.reset(new ModelYolo3(FLAGS_m, (float)FLAGS_t, FLAGS_auto_resize, FLAGS_yolo_af, (float)FLAGS_iou_t, labels));
        }
        else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        InferenceEngine::Core core;

        AsyncPipeline pipeline(std::move(model),
            ConfigFactory::getUserConfig(FLAGS_d, FLAGS_l, FLAGS_c, FLAGS_pc, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
            core);
        Presenter presenter(FLAGS_u);

        bool keepRunning = true;
        int64_t frameNum = -1;
        std::unique_ptr<ResultBase> result;
        uint32_t framesProcessed = 0;

        PerformanceMetrics renderMetrics;

        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();
        size_t found = FLAGS_output_resolution.find("x");

        //int maxNum = img_files.size();
        int idx=0;

        while (keepRunning) {
            string fName;

            while (true) {
                if (lab_imgs.empty())
                {
                    this_thread::sleep_for(chrono::milliseconds(1) );
                } else {
                    fName = lab_imgs.back();
                    lab_imgs.pop_back();
                    break;
                }
            }

            //if (idx==maxNum)
            //    break;

            if (pipeline.isReadyToProcess()) {
                auto startTime = std::chrono::steady_clock::now();

                //--- Capturing frame
                //fName=getFileName(text.c_str());
                curr_frame = cv::imread(fName, cv::IMREAD_COLOR);

                if (curr_frame.empty()) {
                    if (frameNum == -1) {
                        throw std::logic_error("Can't read an image from the input");
                    }
                    else {
                        // Input stream is over
                        break;
                    }
                }

                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                    std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            frameNum=1;   // disable async 0 <-> 1 

            if (frameNum == 0) {   // first frameNum 0
                if (found == std::string::npos) {
                    outputResolution = curr_frame.size();
                }
                else {
                    outputResolution = cv::Size{
                        std::stoi(FLAGS_output_resolution.substr(0, found)),
                        std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))
                    };
                    outputTransform = OutputTransform(curr_frame.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();
                }
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them are available.
            //pipeline.waitForData();
            pipeline.waitForTotalCompletion();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            while (keepRunning && (result = pipeline.getResult())) {
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), palette, outputTransform, framesProcessed);
                presenter.drawGraphs(outFrame);
                framesProcessed++;

            }
            idx++;
            //this_thread::sleep_for(chrono::milliseconds(50) );
        }

#if 0
        //// ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();
        for (; framesProcessed <= frameNum; framesProcessed++) {
            while (!(result = pipeline.getResult())) {}
            cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), palette, outputTransform, framesProcessed);
        }
#endif

    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    pthread_join(thread1,NULL);
    pthread_join(thread2,NULL);
    slog::info << slog::endl << "The execution has completed successfully" << slog::endl;
    return 0;
}
