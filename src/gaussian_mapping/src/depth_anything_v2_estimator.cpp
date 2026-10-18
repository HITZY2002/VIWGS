#include "depth_anything_v2_estimator.h"

#include <ros/ros.h>
#include <opencv2/imgproc.hpp>
#ifdef VIWGS_WITH_ONNX
#include <onnxruntime_c_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <array>
#include <cstring>

namespace gaussian_mapping {

#ifdef VIWGS_WITH_ONNX
namespace {
Ort::Env& getOrtEnv()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "depth_anything_v2");
    return env;
}
}  // namespace

bool DepthAnythingV2Estimator::loadModel(const std::string& model_path, int input_size, const std::string& device)
{
    if (model_path.empty()) {
        ROS_WARN("[DepthAnythingV2] Model path is empty, depth fusion disabled.");
        ready_ = false;
        return false;
    }

    std::string dev = device;
    std::transform(dev.begin(), dev.end(), dev.begin(), ::tolower);
    session_options_ = Ort::SessionOptions();
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        if (dev == "cuda" || dev == "cuda_fp16") {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = SIZE_MAX;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        }

        session_ = std::make_unique<Ort::Session>(getOrtEnv(), model_path.c_str(), session_options_);
    } catch (const std::exception& e) {
        ROS_ERROR("[DepthAnythingV2] Failed to load model %s: %s", model_path.c_str(), e.what());
        ready_ = false;
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t num_inputs = session_->GetInputCount();
    const size_t num_outputs = session_->GetOutputCount();
    if (num_inputs == 0 || num_outputs == 0) {
        ROS_ERROR("[DepthAnythingV2] Invalid ONNX model IO count: in=%zu out=%zu", num_inputs, num_outputs);
        ready_ = false;
        return false;
    }

    input_names_storage_.clear();
    output_names_storage_.clear();
    input_names_.clear();
    output_names_.clear();

    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_storage_.emplace_back(name.get());
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_storage_.emplace_back(name.get());
    }
    for (auto& s : input_names_storage_) {
        input_names_.push_back(s.c_str());
    }
    for (auto& s : output_names_storage_) {
        output_names_.push_back(s.c_str());
    }

    input_size_ = std::max(128, input_size);
    ready_ = true;
    ROS_INFO("[DepthAnythingV2] Loaded ONNXRuntime model: %s (input=%d, device=%s)",
             model_path.c_str(), input_size_, dev.c_str());
    return true;
}

bool DepthAnythingV2Estimator::inferDepth(const cv::Mat& rgb_image, cv::Mat& depth_out)
{
    if (!ready_ || rgb_image.empty()) {
        return false;
    }

    cv::Mat rgb_u8;
    if (rgb_image.type() == CV_8UC3) {
        rgb_u8 = rgb_image;
    } else if (rgb_image.type() == CV_32FC3) {
        cv::Mat tmp;
        rgb_image.convertTo(tmp, CV_8UC3, 255.0);
        rgb_u8 = tmp;
    } else {
        return false;
    }

    cv::Mat input;
    cv::resize(rgb_u8, input, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);

    cv::Mat input_f32;
    input.convertTo(input_f32, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> chw;
    cv::split(input_f32, chw);
    const std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    const std::array<float, 3> stdv = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        chw[c] = (chw[c] - mean[c]) / stdv[c];
    }

    std::vector<float> input_tensor_values(static_cast<size_t>(3 * input_size_ * input_size_));
    const size_t channel_stride = static_cast<size_t>(input_size_ * input_size_);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            input_tensor_values.data() + c * channel_stride,
            chw[c].ptr<float>(),
            channel_stride * sizeof(float));
    }

    const std::array<int64_t, 4> input_shape = {1, 3, input_size_, input_size_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size());

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size());

    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        return false;
    }

    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    float* out_ptr = output_tensors[0].GetTensorMutableData<float>();

    cv::Mat depth;
    if (shape.size() == 4 && shape[0] == 1) {
        const int h = static_cast<int>(shape[2]);
        const int w = static_cast<int>(shape[3]);
        depth = cv::Mat(h, w, CV_32F, out_ptr).clone();
    } else if (shape.size() == 3 && shape[0] == 1) {
        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);
        depth = cv::Mat(h, w, CV_32F, out_ptr).clone();
    } else if (shape.size() == 2) {
        const int h = static_cast<int>(shape[0]);
        const int w = static_cast<int>(shape[1]);
        depth = cv::Mat(h, w, CV_32F, out_ptr).clone();
    } else {
        ROS_ERROR("[DepthAnythingV2] Unexpected output shape rank: %zu", shape.size());
        return false;
    }

    if (depth.empty()) {
        return false;
    }

    cv::Mat depth_positive;
    cv::max(depth, 1e-6f, depth_positive);

    cv::resize(depth_positive, depth_out, rgb_u8.size(), 0, 0, cv::INTER_CUBIC);
    return true;
}

#else
bool DepthAnythingV2Estimator::loadModel(const std::string&, int, const std::string&)
{
    ROS_ERROR("Depth fusion requires a build with -DVIWGS_WITH_ONNX=ON.");
    return false;
}

bool DepthAnythingV2Estimator::inferDepth(const cv::Mat&, cv::Mat&)
{
    return false;
}
#endif

}  // namespace gaussian_mapping
