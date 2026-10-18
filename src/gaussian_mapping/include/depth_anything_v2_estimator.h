#ifndef DEPTH_ANYTHING_V2_ESTIMATOR_H
#define DEPTH_ANYTHING_V2_ESTIMATOR_H

#include <opencv2/core.hpp>
#ifdef VIWGS_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#include <string>
#include <vector>
#include <memory>

namespace gaussian_mapping {

class DepthAnythingV2Estimator {
public:
    DepthAnythingV2Estimator() = default;

    bool loadModel(const std::string& model_path, int input_size, const std::string& device);
    bool isReady() const { return ready_; }

    // Input image must be RGB CV_32FC3 [0,1] or CV_8UC3.
    bool inferDepth(const cv::Mat& rgb_image, cv::Mat& depth_out);

private:
    bool ready_ = false;
    int input_size_ = 518;

#ifdef VIWGS_WITH_ONNX
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
#endif
};

}  // namespace gaussian_mapping

#endif
