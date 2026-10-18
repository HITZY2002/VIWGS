// SPDX-License-Identifier: GPL-3.0-or-later
// VIWGS ROS entrypoint. Core mapping is derived from Photo-SLAM; see LICENSE and THIRD_PARTY.md.
#include <ros/ros.h>
#include <torch/cuda.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>
#include "gaussian_mapper.h"
#include "ros_data_provider.h"
#ifdef VIWGS_BUILD_VIEWER
#include "viewer/imgui_viewer.h"
#endif

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gaussian_mapping");
    ros::NodeHandle nh;
    ros::NodeHandle params("~");
    std::shared_ptr<GaussianMapper> mapper;
    std::thread mapping_thread;
    std::atomic<bool> mapping_failed{false};
    int result = 0;
    try {
        std::string config, output_dir;
        bool enable_viewer = false;
        double fx, fy, cx, cy;
        int width, height;
        params.param("enable_viewer", enable_viewer, false);
#ifndef VIWGS_BUILD_VIEWER
        if (enable_viewer)
            throw std::runtime_error("Viewer requires -DVIWGS_BUILD_VIEWER=ON at build time.");
#endif
        if (!params.getParam("gaussian_config_file", config) || !std::filesystem::is_regular_file(config))
            throw std::runtime_error("Set gaussian_config_file to an existing algorithm YAML file.");
        if (!params.getParam("output_dir", output_dir) || output_dir.empty())
            throw std::runtime_error("Set output_dir to a writable result directory.");
        if (!params.getParam("fx", fx) || !params.getParam("fy", fy) ||
            !params.getParam("cx", cx) || !params.getParam("cy", cy) ||
            !params.getParam("image_width", width) || !params.getParam("image_height", height) ||
            fx <= 0 || fy <= 0 || width <= 0 || height <= 0)
            throw std::runtime_error("Load camera intrinsics and image size through mapping_config.");
        if (!torch::cuda::is_available())
            throw std::runtime_error("An NVIDIA GPU and a working CUDA LibTorch installation are required.");

        std::filesystem::create_directories(output_dir);
        auto provider = std::make_shared<gaussian_mapping::ROSDataProvider>(nh, params);
        provider->setCameraIntrinsics(fx, fy, cx, cy, width, height);
        mapper = std::make_shared<GaussianMapper>(provider, config, output_dir, 0, torch::kCUDA);
        ROS_INFO("VIWGS initialized; waiting for keyframes.");
        mapping_thread = std::thread([&]() {
            try {
                mapper->run();
            } catch (const std::exception& e) {
                ROS_ERROR("Mapping failed: %s", e.what());
                mapping_failed = true;
            }
            ros::shutdown();
        });
        ros::AsyncSpinner spinner(2);
        spinner.start();
#ifdef VIWGS_BUILD_VIEWER
        if (enable_viewer) {
            ImGuiViewer viewer(mapper, true);
            viewer.run();
            ros::shutdown();
        } else
#endif
        {
            ros::waitForShutdown();
        }
    } catch (const std::exception& e) {
        ROS_ERROR("VIWGS: %s", e.what());
        result = 1;
    }
    ros::shutdown();
    if (mapper) mapper->signalStop();
    if (mapping_thread.joinable()) mapping_thread.join();
    return mapping_failed ? 1 : result;
}
