/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "imgui_viewer.h"
#include <ros/ros.h>

namespace {
glm::mat4 eigenMat4ToGlm(const Eigen::Matrix4f& mat)
{
    glm::mat4 out(1.0f);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[c][r] = mat(r, c);
        }
    }
    return out;
}

bool projectPointToScreen(
    const glm::mat4& cam_trans,
    const Eigen::Vector3f& p_w,
    int screen_w,
    int screen_h,
    ImVec2& out)
{
    glm::vec4 clip = cam_trans * glm::vec4(p_w.x(), p_w.y(), p_w.z(), 1.0f);
    if (clip.w <= 1e-6f) {
        return false;
    }
    const float inv_w = 1.0f / clip.w;
    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;

    // clip-space to screen-space (origin: top-left)
    out.x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(screen_w);
    out.y = (1.0f - (ndc_y * 0.5f + 0.5f)) * static_cast<float>(screen_h);
    return true;
}

void drawSegmentProjected(
    ImDrawList* drawlist,
    const glm::mat4& cam_trans,
    const Eigen::Vector3f& p0,
    const Eigen::Vector3f& p1,
    int screen_w,
    int screen_h,
    ImU32 color,
    float thickness)
{
    ImVec2 s0, s1;
    if (!projectPointToScreen(cam_trans, p0, screen_w, screen_h, s0)) return;
    if (!projectPointToScreen(cam_trans, p1, screen_w, screen_h, s1)) return;
    drawlist->AddLine(s0, s1, color, thickness);
}

void drawFrustumProjected(
    ImDrawList* drawlist,
    const glm::mat4& cam_trans,
    const Sophus::SE3f& Twc,
    float w,
    float h,
    float z,
    int screen_w,
    int screen_h,
    ImU32 color,
    float thickness)
{
    const Eigen::Matrix3f R = Twc.rotationMatrix();
    const Eigen::Vector3f t = Twc.translation();

    auto tf = [&](const Eigen::Vector3f& pc) {
        return R * pc + t;
    };

    const Eigen::Vector3f p0 = tf(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    const Eigen::Vector3f p1 = tf(Eigen::Vector3f( w,  h, z));
    const Eigen::Vector3f p2 = tf(Eigen::Vector3f( w, -h, z));
    const Eigen::Vector3f p3 = tf(Eigen::Vector3f(-w, -h, z));
    const Eigen::Vector3f p4 = tf(Eigen::Vector3f(-w,  h, z));

    drawSegmentProjected(drawlist, cam_trans, p0, p1, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p0, p2, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p0, p3, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p0, p4, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p1, p2, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p2, p3, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p3, p4, screen_w, screen_h, color, thickness);
    drawSegmentProjected(drawlist, cam_trans, p4, p1, screen_w, screen_h, color, thickness);
}

void drawTrajectoryProjected(
    ImDrawList* drawlist,
    const glm::mat4& cam_trans,
    const std::vector<Eigen::Vector3f>& points,
    int screen_w,
    int screen_h,
    ImU32 color,
    float thickness)
{
    if (points.size() < 2) {
        return;
    }

    for (size_t i = 1; i < points.size(); ++i) {
        drawSegmentProjected(drawlist, cam_trans, points[i - 1], points[i], screen_w, screen_h, color, thickness);
    }
}

bool projectWorldToImage(
    const Sophus::SE3f& Tcw,
    const Eigen::Vector3f& Pw,
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height,
    cv::Point& uv)
{
    const Eigen::Vector3f Pc = Tcw.rotationMatrix() * Pw + Tcw.translation();
    if (Pc.z() <= 1e-4f) {
        return false;
    }

    const float u = fx * (Pc.x() / Pc.z()) + cx;
    const float v = fy * (Pc.y() / Pc.z()) + cy;
    if (!std::isfinite(u) || !std::isfinite(v)) {
        return false;
    }
    if (u < 0.0f || u >= static_cast<float>(width) || v < 0.0f || v >= static_cast<float>(height)) {
        return false;
    }

    uv.x = static_cast<int>(std::round(u));
    uv.y = static_cast<int>(std::round(v));
    return true;
}

void drawCameraRectMarkerOnImage(
    cv::Mat& img,
    const Sophus::SE3f& Twc_cam,
    const Sophus::SE3f& Tcw_view,
    float fx,
    float fy,
    float cx,
    float cy,
    int width,
    int height,
    float marker_w,
    float marker_h,
    float marker_z,
    const cv::Scalar& color,
    int thickness)
{
    const Eigen::Matrix3f R = Twc_cam.rotationMatrix();
    const Eigen::Vector3f t = Twc_cam.translation();
    auto tf = [&](const Eigen::Vector3f& pc) {
        return R * pc + t;
    };

    // Rectangle on camera-local plane z=marker_z
    const Eigen::Vector3f p1w = tf(Eigen::Vector3f( marker_w,  marker_h, marker_z));
    const Eigen::Vector3f p2w = tf(Eigen::Vector3f( marker_w, -marker_h, marker_z));
    const Eigen::Vector3f p3w = tf(Eigen::Vector3f(-marker_w, -marker_h, marker_z));
    const Eigen::Vector3f p4w = tf(Eigen::Vector3f(-marker_w,  marker_h, marker_z));

    cv::Point p1, p2, p3, p4;
    if (!projectWorldToImage(Tcw_view, p1w, fx, fy, cx, cy, width, height, p1)) return;
    if (!projectWorldToImage(Tcw_view, p2w, fx, fy, cx, cy, width, height, p2)) return;
    if (!projectWorldToImage(Tcw_view, p3w, fx, fy, cx, cy, width, height, p3)) return;
    if (!projectWorldToImage(Tcw_view, p4w, fx, fy, cx, cy, width, height, p4)) return;

    cv::line(img, p1, p2, color, thickness, cv::LINE_AA);
    cv::line(img, p2, p3, color, thickness, cv::LINE_AA);
    cv::line(img, p3, p4, color, thickness, cv::LINE_AA);
    cv::line(img, p4, p1, color, thickness, cv::LINE_AA);
    // Diagonals
    cv::line(img, p1, p3, color, std::max(1, thickness - 1), cv::LINE_AA);
    cv::line(img, p2, p4, color, std::max(1, thickness - 1), cv::LINE_AA);
}

const char* poseOptStatusToText(int status_code)
{
    switch (status_code) {
    case 0: return "Ran";
    case 1: return "Disabled";
    case 2: return "SkippedByLoad";
    case 3: return "RenderLockBusy";
    case 4: return "NoTask";
    case 7: return "HitMsBudget";
    case 8: return "HitChunkBudget";
    case 9: return "HitStepBudget";
    default: return "Unknown";
    }
}

cv::Mat makeDepthPreviewImage(const cv::Mat& depth_like)
{
    if (depth_like.empty()) {
        return cv::Mat();
    }

    cv::Mat depth32;
    if (depth_like.type() == CV_32FC1) {
        depth32 = depth_like;
    } else {
        depth_like.convertTo(depth32, CV_32FC1);
    }

    cv::Mat valid_mask = (depth32 > 1e-6f) & (depth32 == depth32);
    if (cv::countNonZero(valid_mask) <= 0) {
        return cv::Mat();
    }

    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(depth32, &min_v, &max_v, nullptr, nullptr, valid_mask);
    if (!std::isfinite(min_v) || !std::isfinite(max_v) || max_v <= min_v + 1e-8) {
        return cv::Mat();
    }

    cv::Mat norm_u8;
    cv::Mat depth_clamped = depth32.clone();
    depth_clamped.setTo(min_v, depth_clamped < min_v);
    depth_clamped.setTo(max_v, depth_clamped > max_v);
    depth_clamped = (depth_clamped - static_cast<float>(min_v)) / static_cast<float>(max_v - min_v);
    depth_clamped.convertTo(norm_u8, CV_8UC1, 255.0);

    cv::Mat colored_bgr;
    cv::applyColorMap(norm_u8, colored_bgr, cv::COLORMAP_TURBO);
    cv::Mat colored_rgb;
    cv::cvtColor(colored_bgr, colored_rgb, cv::COLOR_BGR2RGB);

    cv::Mat colored_f32;
    colored_rgb.convertTo(colored_f32, CV_32FC3, 1.0 / 255.0);
    return colored_f32;
}
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "[ImGuiViewer]GLFW Error %d: %s\n", error, description);
}

ImGuiViewer::ImGuiViewer(
    std::shared_ptr<GaussianMapper> pGausMapper,
    bool training)
    : glfw_window_width_(1600),
      glfw_window_height_(900),
      panel_width_(372),
            display_panel_height_(360),
      training_panel_height_(440),
      camera_panel_height_(144),
      SLAM_image_viewer_scale_(1.0f),
      training_(training)
{
    this->pGausMapper_ = pGausMapper;

    cv::Size im_size;
    image_height_ = pGausMapper->scene_->cameras_.begin()->second.height_;
    image_width_ = pGausMapper->scene_->cameras_.begin()->second.width_;
    viewpointF_ = pGausMapper->scene_->cameras_.begin()->second.params_[1];
    im_size = cv::Size(image_width_, image_height_);

    main_fx_ = pGausMapper->scene_->cameras_.begin()->second.params_[0];
    main_fy_ = pGausMapper->scene_->cameras_.begin()->second.params_[1];

    // Gaussian Mapper settings
    std::filesystem::path cfg_file_path = pGausMapper->config_file_path_;
    readConfigFromFile(cfg_file_path);
    SLAM_image_viewer_scale_ = static_cast<float>(rendered_image_width_) / image_width_;

    float fovy = graphics_utils::focal2fov(viewpointF_, im_size.height);
    cam_proj_ = glm::perspective(
        fovy < M_PIf32 ? fovy : M_PIf32, (float)glfw_window_width_ / (float)glfw_window_height_, 0.01f, 5000.0f);

    up_ = glm::vec3(0.0f, -1.0f, 0.0f);
    up_aligned_ = glm::vec4(up_, 1.0f);
    behind_ = glm::vec4(0.0f, 0.0f, -camera_watch_dist_, 1.0f);
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, up_);
    cam_trans_ = cam_proj_ * cam_view_;

    // ORB-SLAM3 viewer components removed for VINS-only mode
}

void ImGuiViewer::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened())
       throw std::runtime_error("[ImGuiViewer]Failed to open settings file at: " + cfg_path.string());
    std::cout << "[ImGuiViewer]Reading parameters from " << cfg_path << std::endl;

    glfw_window_width_ =
        settings_file["GaussianViewer.glfw_window_width"].operator int();
    glfw_window_height_ =
        settings_file["GaussianViewer.glfw_window_height"].operator int();
    main_cx_ = glfw_window_width_ / 2;
    main_cy_ = glfw_window_height_ / 2;

    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_height_ = image_height_ * rendered_image_viewer_scale_;
    rendered_image_width_ = image_width_ * rendered_image_viewer_scale_;

    int temp = rendered_image_width_ % 4;
    padded_sub_image_width_ = rendered_image_width_ + 4 - (temp == 0 ? 4 : temp);

    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
    rendered_image_height_main_ = image_height_ * rendered_image_viewer_scale_main_;
    rendered_image_width_main_ = image_width_ * rendered_image_viewer_scale_main_;

    temp = rendered_image_width_main_ % 4;
    padded_main_image_width_ = rendered_image_width_main_ + 4 - (temp == 0 ? 4 : temp); 

    camera_watch_dist_ =
        settings_file["GaussianViewer.camera_watch_dist"].operator float();

    // Initialize configurations same as the GaussianMapper
    position_lr_init_ = pGausMapper_->positionLearningRateInit();
    feature_lr_ = pGausMapper_->featureLearningRate();
    opacity_lr_ = pGausMapper_->opacityLearningRate();
    scaling_lr_ = pGausMapper_->scalingLearningRate();
    rotation_lr_ = pGausMapper_->rotationLearningRate();
    percent_dense_ = pGausMapper_->percentDense();
    lambda_dssim_ = pGausMapper_->lambdaDssim();
    opacity_reset_interval_ = pGausMapper_->opacityResetInterval();
    densify_grad_th_ = pGausMapper_->densifyGradThreshold();
    densify_interval_ = pGausMapper_->densifyInterval();
    new_kf_times_of_use_ = pGausMapper_->newKeyframeTimesOfUse();
    stable_num_iter_existence_ = pGausMapper_->stableNumIterExistence();

    do_gaus_pyramid_training_ = pGausMapper_->isdoingGausPyramidTraining();
    do_inactive_geo_densify_ = pGausMapper_->isdoingInactiveGeoDensify();
}

void ImGuiViewer::run()
{
    // Initialize glfw
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("[ImGuiViewer]Fails to initialize!");

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create window with graphics context
    GLFWwindow* window =
        glfwCreateWindow(glfw_window_width_, glfw_window_height_,
                         "VIWGS", nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error("[ImGuiViewer]Fails to create window!");
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    glEnable(GL_DEPTH_TEST); // Enable 3D Mouse handler

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Variables for tracking
    Sophus::SE3f Tcw, TcwInit;
    TcwInit = Sophus::SE3f();
    Tcw_main_ = TcwInit;
    Twc_main_ = Tcw_main_.inverse();
    cam_pos_ = glm::vec3(viewpointX_, viewpointY_, viewpointZ_);
    glm::vec4 cam_pos_aligned = glm::vec4(cam_pos_, 1.0f);
    cam_target_ = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cam_direction = cam_pos_ - cam_target_;
    glm::vec3 cam_right = glm::normalize(glm::cross(up_, cam_direction));
    glm::vec3 cam_up = glm::cross(cam_direction, cam_right);
    glm::vec4 cam_up_aligned = glm::vec4(cam_up, 1.0f);
    cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
    glm::mat4 glmTwc, Twr, glmTwcInit;
    glmTwc = glm::mat4(1.0f);
    glmTwcInit = glm::mat4(1.0f);
    glmTwc_main_ = glm::mat4(1.0f);
    glm::mat4 Ow, OwInit;
    Ow = glm::mat4(1.0f);
    OwInit = glm::mat4(1.0f);

    // Variables for showing images
    cv::Rect image_rect_sub(0, 0, rendered_image_width_, rendered_image_height_);
    cv::Rect image_rect_main(0, 0, rendered_image_width_main_, rendered_image_height_main_);

    GLuint main_img_texture;
    GLuint gt_img_texture;
    GLuint depth_img_texture;

    glGenTextures(1, &main_img_texture);
    glBindTexture(GL_TEXTURE_2D, main_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &gt_img_texture);
    glBindTexture(GL_TEXTURE_2D, gt_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &depth_img_texture);
    glBindTexture(GL_TEXTURE_2D, depth_img_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Main loop
    while(ros::ok() && !pGausMapper_->isStopped() && !isStopped() && !glfwWindowShouldClose(window))
    {
        //--------------Poll and handle events (inputs, window resize, etc.)--------------
        glfwPollEvents();

        //--------------Start the Dear ImGui frame--------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (tracking_vision_)
        {
            const auto& keyframes = pGausMapper_->scene_->keyframes();
            if (!keyframes.empty()) {
                auto last_kf = keyframes.rbegin()->second;
                Tcw_main_ = last_kf->getPosef();
                init_Twc_set_ = true;
            }
        }

        Sophus::SE3f Twc_display = Tcw_main_.inverse();

        // Latest keyframe pose (independent from viewer tracking mode)
        Sophus::SE3f Twc_latest = Twc_display;
        std::shared_ptr<GaussianKeyframe> latest_pkf = nullptr;
        std::size_t latest_kf_id = 0;
        {
            const auto& kfs = pGausMapper_->scene_->keyframes();
            if (!kfs.empty()) {
                auto it_last = kfs.rbegin();
                if (it_last->second) {
                    latest_pkf = it_last->second;
                    latest_kf_id = it_last->second->fid_;
                    Twc_latest = it_last->second->getPosef().inverse();
                }
            }
        }
        Eigen::Vector3f p_latest = Twc_latest.translation();
        Eigen::Quaternionf q_latest(Twc_latest.rotationMatrix());
        q_latest.normalize();

        // No ORB-SLAM3 tracking; use free-view camera only.
        if (tracking_vision_)
        {
            glm::vec3 cam_target = glm::vec3(Ow[3][0], Ow[3][1], Ow[3][2]);
            cam_pos_aligned = glmTwc * behind_;
            glm::vec3 cam_pos = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos - cam_target;
            // cam_right = glm::normalize(glm::cross(up_, cam_direction));
            // cam_up = glm::cross(cam_direction, cam_right);
            cam_up_aligned = glmTwc * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos, cam_target, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }
        else
        {
            if (reset_main_to_init_ || !init_Twc_set_)
            {
                cam_target_ = glm::vec3(OwInit[3][0], OwInit[3][1], OwInit[3][2]);
                cam_pos_aligned = glmTwcInit * behind_;
                glmTwc_main_ = glmTwcInit;
                Tcw_main_ = TcwInit;
                Twc_main_ = Tcw_main_.inverse();
                init_Twc_set_ = true;
                reset_main_to_init_ = false;
            }
            else
            {
                handleUserInput();
                glmTwc_main_ = eigenMat4ToGlm(Tcw_main_.inverse().matrix());
                cam_target_ = glm::vec3(glmTwc_main_[3][0], glmTwc_main_[3][1], glmTwc_main_[3][2]);
                cam_pos_aligned = glmTwc_main_ * behind_;
            }
            cam_pos_ = glm::vec3(cam_pos_aligned.x, cam_pos_aligned.y, cam_pos_aligned.z);
            cam_direction = cam_pos_ - cam_target_;
            // cam_right = glm::normalize(glm::cross(up_, cam_direction));
            // cam_up = glm::cross(cam_direction, cam_right);
            cam_up_aligned = glmTwc_main_ * up_aligned_;
            cam_up = glm::normalize(glm::vec3(cam_up_aligned.x, cam_up_aligned.y, cam_up_aligned.z) - cam_target_);
            cam_right = glm::normalize(glm::cross(cam_up, cam_direction));
            cam_up = glm::cross(cam_direction, cam_right);
            cam_view_ = glm::lookAt(cam_pos_, cam_target_, cam_up);
            cam_trans_ = cam_proj_ * cam_view_;
        }

        // No SLAM frame window in VINS-only mode.

        //--------------Draw main window image--------------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Draw main window image
        if (show_main_rendered_)
        {
            auto drawlist = ImGui::GetBackgroundDrawList();
            cv::Mat main_img = pGausMapper_->renderFromPose(
                Tcw_main_, rendered_image_width_main_, rendered_image_height_main_, true);

            // Robust fallback: draw trajectory/frustums directly on rendered image.
            // This avoids OpenGL fixed-pipeline state issues and guarantees visibility.
            const auto& kfs_overlay = pGausMapper_->scene_->keyframes();
            std::vector<Eigen::Vector3f> centers_overlay;
            centers_overlay.reserve(kfs_overlay.size());
            for (const auto& kv : kfs_overlay) {
                const auto& pkf = kv.second;
                if (!pkf) continue;
                centers_overlay.push_back(pkf->getPosef().inverse().translation());
            }

            const auto& cam0 = pGausMapper_->scene_->cameras_.begin()->second;
            const float fx_s = static_cast<float>(cam0.params_[0]) * rendered_image_viewer_scale_main_;
            const float fy_s = static_cast<float>(cam0.params_[1]) * rendered_image_viewer_scale_main_;
            const float cx_s = static_cast<float>(cam0.params_[2]) * rendered_image_viewer_scale_main_;
            const float cy_s = static_cast<float>(cam0.params_[3]) * rendered_image_viewer_scale_main_;

            if (show_camera_trajectory_ && centers_overlay.size() >= 2) {
                cv::Point p_prev;
                bool has_prev = false;
                for (const auto& pw : centers_overlay) {
                    cv::Point p_cur;
                    bool ok = projectWorldToImage(
                        Tcw_main_, pw,
                        fx_s, fy_s, cx_s, cy_s,
                        rendered_image_width_main_, rendered_image_height_main_,
                        p_cur);
                    if (!ok) continue;
                    if (has_prev) {
                        cv::line(main_img, p_prev, p_cur, cv::Scalar(0.10f, 0.50f, 1.00f), 2, cv::LINE_AA);
                    }
                    p_prev = p_cur;
                    has_prev = true;
                }
            }

            if (show_camera_frustums_ && !centers_overlay.empty()) {
                // Draw each camera as a rectangle with diagonals to reveal orientation.
                float marker_w = 0.04f;
                float marker_h = marker_w * 0.75f;
                float marker_z = marker_w * 1.8f;
                for (const auto& kv : kfs_overlay) {
                    const auto& pkf = kv.second;
                    if (!pkf) continue;
                    Sophus::SE3f Twc_kf = pkf->getPosef().inverse();
                    drawCameraRectMarkerOnImage(
                        main_img,
                        Twc_kf,
                        Tcw_main_,
                        fx_s, fy_s, cx_s, cy_s,
                        rendered_image_width_main_, rendered_image_height_main_,
                        marker_w, marker_h, marker_z,
                        cv::Scalar(1.0f, 0.8f, 0.2f),
                        2);
                }

                // Highlight latest camera marker.
                drawCameraRectMarkerOnImage(
                    main_img,
                    Twc_latest,
                    Tcw_main_,
                    fx_s, fy_s, cx_s, cy_s,
                    rendered_image_width_main_, rendered_image_height_main_,
                    marker_w * 1.3f, marker_h * 1.3f, marker_z * 1.3f,
                    cv::Scalar(0.2f, 1.0f, 1.0f),
                    3);
            }

            cv::Mat main_img_to_show = cv::Mat(rendered_image_height_main_, padded_main_image_width_, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
            main_img.copyTo(main_img_to_show(image_rect_main));
            glBindTexture(GL_TEXTURE_2D, main_img_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, main_img_to_show.cols, main_img_to_show.rows,
                 0, GL_RGB, GL_FLOAT, (float*)main_img_to_show.data);
            drawlist->AddImage((void *)(intptr_t)main_img_texture, ImVec2(0, 0),
                               ImVec2(glfw_window_width_, glfw_window_height_));
        }
        //--------------Get current parameters--------------
        VariableParameters params_in = pGausMapper_->getVaribleParameters();
        position_lr_init_ = params_in.position_lr_init;
        feature_lr_ = params_in.feature_lr;
        opacity_lr_ = params_in.opacity_lr;
        scaling_lr_ = params_in.scaling_lr;
        rotation_lr_ = params_in.rotation_lr;
        percent_dense_ = params_in.percent_dense;
        lambda_dssim_ = params_in.lambda_dssim;
        opacity_reset_interval_ = params_in.opacity_reset_interval;
        densify_grad_th_ = params_in.densify_grad_th;
        densify_interval_ = params_in.densify_interval;
        new_kf_times_of_use_ = params_in.new_kf_times_of_use;
        stable_num_iter_existence_ = params_in.stable_num_iter_existence;
        keep_training_ = params_in.keep_training;
        do_gaus_pyramid_training_ = params_in.do_gaus_pyramid_training;
        do_inactive_geo_densify_ = params_in.do_inactive_geo_densify;

        //--------------Display/Training/Camera panels (adaptive right-side layout)--------------
        const float side_margin = 8.0f;
        const float panel_gap = 8.0f;
        const float panel_w = std::min(460.0f, std::max(320.0f, static_cast<float>(glfw_window_width_) * 0.27f));
        const float panel_x = static_cast<float>(glfw_window_width_) - panel_w - side_margin;
        const float panel_top = side_margin;
        const float total_h = std::max(300.0f, static_cast<float>(glfw_window_height_) - 2.0f * side_margin);
        const float display_panel_h = std::min(420.0f, std::max(260.0f, total_h * (training_ ? 0.38f : 0.55f)));
        const float camera_h = std::min(220.0f, std::max(150.0f, total_h * (training_ ? 0.18f : 0.25f)));
        const float training_h = training_ ? std::max(220.0f, total_h - display_panel_h - camera_h - 2.0f * panel_gap) : 0.0f;

        ImGui::SetNextWindowPos(ImVec2(panel_x, panel_top), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(panel_w, display_panel_h), ImGuiCond_Always);
        {
            ImGui::Begin("Display Mode", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

            const auto poseopt_info = pGausMapper_->getPoseOptDebugInfo();

            if (training_)
            {
                ImGui::Text("VINS status: %s", pGausMapper_->isStopped() ? "Stopped" : "Running");
                ImGui::Text("Iteration: %d", pGausMapper_->getIteration());
                ImGui::Text("Loss (EMA): %.6f", pGausMapper_->getEmaLossForLog());
                ImGui::Separator();
                ImGui::Text("PoseOpt: %s", poseopt_info.enabled ? "Enabled" : "Disabled");
                ImGui::Text("PoseOpt status: %s", poseOptStatusToText(poseopt_info.last_status_code));
                ImGui::Text("PoseOpt queue: %d / %d", poseopt_info.queue_size, poseopt_info.queue_limit);
                ImGui::Text("PoseOpt last run: steps=%d chunks=%d time=%dms",
                            poseopt_info.last_run_steps,
                            poseopt_info.last_run_chunks,
                            poseopt_info.last_run_elapsed_ms);
                ImGui::Text("PoseOpt total optimize calls: %d", poseopt_info.total_optimize_calls);
                ImGui::Text("PoseOpt load: loop_ema=%.1fms batch=%.3f",
                            poseopt_info.mapping_loop_ema_ms,
                            poseopt_info.last_batch_ratio);
                ImGui::Text("PoseOpt flags: skip_load=%s lock_busy=%s",
                            poseopt_info.last_skipped_by_load ? "Yes" : "No",
                            poseopt_info.last_try_lock_failed ? "Yes" : "No");
            }
            ImGui::Checkbox("Show main window rendered", &show_main_rendered_);
            ImGui::Checkbox("Show current KF GT", &show_current_kf_gt_);
            ImGui::Checkbox("Show current KF depth", &show_current_kf_depth_);
            ImGui::Checkbox("Track latest pose", &tracking_vision_);
            ImGui::Checkbox("Show camera frustums", &show_camera_frustums_);
            ImGui::Checkbox("Show camera trajectory", &show_camera_trajectory_);

            ImGui::Separator();
            const auto kf_count = pGausMapper_->scene_->keyframes().size();
            ImGui::Text("Keyframes: %zu", kf_count);
            ImGui::Text("Latest KF id: %zu", latest_kf_id);
            ImGui::Text("Latest Twc position:");
            ImGui::Text("x=%.3f  y=%.3f  z=%.3f", p_latest.x(), p_latest.y(), p_latest.z());
            ImGui::Text("Latest Twc quaternion:");
            ImGui::Text("qx=%.4f qy=%.4f qz=%.4f qw=%.4f",
                        q_latest.x(), q_latest.y(), q_latest.z(), q_latest.w());

            ImGui::Text("Viewer average FPS %.1f", io.Framerate);
            ImGui::End();
        }

        //--------------Training options panel--------------
        if (training_)
        {
            ImGui::SetNextWindowPos(ImVec2(panel_x, panel_top + display_panel_h + panel_gap), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(panel_w, training_h), ImGuiCond_Always);
            {
                ImGui::Begin("Training Options", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

                ImGui::Text("Iteration: %d", pGausMapper_->getIteration());

                ImGui::Checkbox("Gaussian-pyramid-based training", &do_gaus_pyramid_training_);
                ImGui::Checkbox("Densify with inactive geometries", &do_inactive_geo_densify_);
                ImGui::Checkbox("Keep training after stop", &keep_training_);

                if (ImGui::Button("Enter Tail Optimization")) {
                    keep_training_ = true;
                    pGausMapper_->setKeepTraining(true);
                    pGausMapper_->signalStop(true);
                }
                ImGui::SameLine();
                if (ImGui::Button("Stop Tail Optimization")) {
                    keep_training_ = false;
                    pGausMapper_->setKeepTraining(false);
                    pGausMapper_->signalStop(true);
                }
                ImGui::Text("Tail request pending: %s",
                            pGausMapper_->isTailOptimizationRequested() ? "Yes" : "No");

                ImGui::SliderFloat("Position l.r.", &position_lr_init_, 0.00001f, 0.00100f, "%.5f");
                ImGui::SliderFloat("Feature l.r.", &feature_lr_, 0.0001f, 0.0050f, "%.5f");
                ImGui::SliderFloat("Opacity l.r.", &opacity_lr_, 0.01f, 0.10f, "%.5f");
                ImGui::SliderFloat("Scaling l.r.", &scaling_lr_, 0.001f, 0.010f, "%.5f");
                ImGui::SliderFloat("Rotation l.r.", &rotation_lr_, 0.0001f, 0.0100f, "%.5f");
                ImGui::SliderFloat("Percent dense", &percent_dense_, 0.001f, 0.100f, "%.3f");
                ImGui::SliderFloat("Lambda dssim", &lambda_dssim_, 0.01f, 0.40f, "%.2f");
                ImGui::SliderInt("Opacity reset", &opacity_reset_interval_, 0, 6000);
                ImGui::SliderFloat("Densify grad th.", &densify_grad_th_, 0.0001f, 0.0020f, "%.5f");
                ImGui::SliderInt("Densify int.", &densify_interval_, 1, 400);
                ImGui::SliderInt("New kf. using", &new_kf_times_of_use_, 0, 10);
                ImGui::SliderInt("Stable iter.", &stable_num_iter_existence_, 0, 100);

                ImGui::End();
            }
        }
        VariableParameters params_out;
        params_out.position_lr_init = position_lr_init_;
        params_out.feature_lr = feature_lr_;
        params_out.opacity_lr = opacity_lr_;
        params_out.scaling_lr = scaling_lr_;
        params_out.rotation_lr = rotation_lr_;
        params_out.percent_dense = percent_dense_;
        params_out.lambda_dssim = lambda_dssim_;
        params_out.opacity_reset_interval = opacity_reset_interval_;
        params_out.densify_grad_th = densify_grad_th_;
        params_out.densify_interval = densify_interval_;
        params_out.new_kf_times_of_use = new_kf_times_of_use_;
        params_out.stable_num_iter_existence = stable_num_iter_existence_;
        params_out.keep_training = keep_training_;
        params_out.do_gaus_pyramid_training = do_gaus_pyramid_training_;
        params_out.do_inactive_geo_densify = do_inactive_geo_densify_;
        pGausMapper_->setVaribleParameters(params_out);

        //--------------Camera view panel--------------
        ImGui::SetNextWindowPos(
            ImVec2(panel_x, panel_top + display_panel_h + (training_ ? (training_h + panel_gap) : panel_gap)),
            ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(panel_w, camera_h), ImGuiCond_Always);
        {
            ImGui::Begin("Camera View Velocity", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

            ImGui::SliderFloat("Mouse Left", &mouse_left_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Right", &mouse_right_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Mouse Middle", &mouse_middle_sensitivity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard t", &keyboard_velocity_, 0.01f, 1.0f, "%.2f");
            ImGui::SliderFloat("Keyboard R", &keyboard_anglular_velocity_, 0.01f, 1.0f, "%.2f");

            ImGui::End();
        }

        //--------------Overlay draw: camera frustums and trajectory--------------
        const glm::mat4 cam_trans_overlay = cam_proj_ * cam_view_;
        ImDrawList* fg = ImGui::GetForegroundDrawList();

        // Draw camera frustums and trajectory for Gaussian keyframes.
        const auto& keyframes = pGausMapper_->scene_->keyframes();
        std::vector<Eigen::Vector3f> centers;
        centers.reserve(keyframes.size());
        // Make frustum size adaptive to trajectory scale so it stays visible in both small and large scenes.
        Eigen::Vector3f bb_min(1e9f, 1e9f, 1e9f), bb_max(-1e9f, -1e9f, -1e9f);

        for (const auto& it : keyframes) {
            const auto& pkf = it.second;
            if (!pkf) continue;
            Sophus::SE3f Twc = pkf->getPosef().inverse();
            const Eigen::Vector3f c = Twc.translation();
            centers.push_back(c);
            bb_min = bb_min.cwiseMin(c);
            bb_max = bb_max.cwiseMax(c);
        }

        float traj_span = 0.0f;
        if (!centers.empty()) {
            traj_span = (bb_max - bb_min).norm();
        }
        const float frustum_w = std::max(0.03f, std::min(1.5f, traj_span * 0.01f));
        const float frustum_h = frustum_w * 0.75f;
        const float frustum_z = frustum_w * 0.6f;

        for (const auto& it : keyframes) {
            const auto& pkf = it.second;
            if (!pkf) continue;
            Sophus::SE3f Twc = pkf->getPosef().inverse();
            if (show_camera_frustums_) {
                drawFrustumProjected(
                    fg,
                    cam_trans_overlay,
                    Twc,
                    frustum_w,
                    frustum_h,
                    frustum_z,
                    glfw_window_width_,
                    glfw_window_height_,
                    IM_COL32(51, 204, 255, 255),
                    1.5f);
            }
        }

        if (show_camera_trajectory_) {
            drawTrajectoryProjected(
                fg,
                cam_trans_overlay,
                centers,
                glfw_window_width_,
                glfw_window_height_,
                IM_COL32(255, 128, 26, 255),
                2.0f);
        }

        // Highlight the current tracked/free-view camera pose.
        if (show_camera_frustums_) {
            drawFrustumProjected(
                fg,
                cam_trans_overlay,
                Twc_latest,
                frustum_w * 1.25f,
                frustum_h * 1.25f,
                frustum_z * 1.25f,
                glfw_window_width_,
                glfw_window_height_,
                IM_COL32(255, 255, 51, 255),
                2.5f);
        }

        //--------------Left-bottom keyframe GT / depth windows--------------
        const float lb_padding = 12.0f;
        const float preview_w = std::min(420.0f, std::max(220.0f, static_cast<float>(glfw_window_width_) * 0.25f));
        const float preview_h = preview_w * static_cast<float>(std::max(1, image_height_)) / static_cast<float>(std::max(1, image_width_));

        if (show_current_kf_gt_) {
            ImGui::SetNextWindowPos(
                ImVec2(lb_padding, static_cast<float>(glfw_window_height_) - preview_h - lb_padding),
                ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(preview_w, preview_h + 34.0f), ImGuiCond_Always);
            ImGui::Begin("Current KF GT", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            if (latest_pkf && latest_pkf->original_image_.defined() && latest_pkf->original_image_.numel() > 0) {
                auto gt_tensor = latest_pkf->original_image_.detach();
                cv::Mat gt_img = tensor_utils::torchTensor2CvMat_Float32(gt_tensor);
                if (!gt_img.empty()) {
                    glBindTexture(GL_TEXTURE_2D, gt_img_texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gt_img.cols, gt_img.rows,
                                 0, GL_RGB, GL_FLOAT, reinterpret_cast<float*>(gt_img.data));
                    ImGui::Image((void*)(intptr_t)gt_img_texture, ImVec2(preview_w - 16.0f, preview_h));
                } else {
                    ImGui::Text("GT unavailable");
                }
            } else {
                ImGui::Text("GT unavailable");
            }
            ImGui::End();
        }

        if (show_current_kf_depth_) {
            const float depth_offset = show_current_kf_gt_ ? (preview_h + 34.0f + 8.0f) : 0.0f;
            ImGui::SetNextWindowPos(
                ImVec2(lb_padding, static_cast<float>(glfw_window_height_) - preview_h - lb_padding - depth_offset),
                ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(preview_w, preview_h + 34.0f), ImGuiCond_Always);
            ImGui::Begin("Current KF Depth", nullptr,
                         ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
            if (latest_pkf && !latest_pkf->img_auxiliary_undist_.empty()) {
                cv::Mat depth_vis = makeDepthPreviewImage(latest_pkf->img_auxiliary_undist_);
                if (!depth_vis.empty()) {
                    glBindTexture(GL_TEXTURE_2D, depth_img_texture);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, depth_vis.cols, depth_vis.rows,
                                 0, GL_RGB, GL_FLOAT, reinterpret_cast<float*>(depth_vis.data));
                    ImGui::Image((void*)(intptr_t)depth_img_texture, ImVec2(preview_w - 16.0f, preview_h));
                } else {
                    ImGui::Text("Depth unavailable");
                }
            } else {
                ImGui::Text("Depth unavailable");
            }
            ImGui::End();
        }

        //--------------ImGui Rendering--------------
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (!keep_training_  && pGausMapper_->isStopped())
            signalStop();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteTextures(1, &main_img_texture);
    glDeleteTextures(1, &gt_img_texture);
    glDeleteTextures(1, &depth_img_texture);

    glfwDestroyWindow(window);
    glfwTerminate();

    pGausMapper_->signalStop();

    if (pGausMapper_->isKeepingTraining())
        pGausMapper_->setKeepTraining(false);
}

bool ImGuiViewer::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void ImGuiViewer::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

/**
 * We modify Twc_main_ then Tcw_main_ (Sophus::SE3f) to handle mouse and keyboard inputs
 */
void ImGuiViewer::handleUserInput()
{
    if (tracking_vision_)
    {
        free_view_enabled_ = false;
        return;
    }
    else
    {
        free_view_enabled_ = true;
    }

    Twc_main_ = Tcw_main_.inverse();

    // Only respond to mouse inputs when not interacting with ImGui
    if (!ImGui::IsAnyItemActive() && !ImGui::GetIO().WantCaptureMouse)
    {
        mouseWheel();
        mouseDrag();
    }

    // Respond to keyboard inputs
    keyboardEvent();

    Tcw_main_ = Twc_main_.inverse();
}

void ImGuiViewer::mouseWheel()
{
    float delta = ImGui::GetIO().MouseWheel;

    if (delta == 0)
        return;

    float scale_factor = std::pow(1.1f, -delta);

    // float mouse_x = ImGui::GetMousePos().x;
    // float mouse_y = ImGui::GetMousePos().y;

    // float focus3D_x = (mouse_x - main_cx_) / main_fx_;
    // float focus3D_y = (mouse_y - main_cy_) / main_fy_;

    //---Translation---
    // Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // translating.x() += focus3D_x * (1 - scale_factor);
    // translating.y() += focus3D_y * (1 - scale_factor);
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() *= scale_factor;
    // Twc_main_.translation() += (R * translating);
}

void ImGuiViewer::mouseDrag()
{
    float delta_rel_x = ImGui::GetIO().MouseDelta.x / glfw_window_width_;
    float delta_rel_y = ImGui::GetIO().MouseDelta.y / glfw_window_height_;
    float delta_l = delta_rel_x * delta_rel_x + delta_rel_y * delta_rel_y;

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // Left held
    if (ImGui::GetIO().MouseDown[0])
    {
        // Upward (delta_y < 0): pitch upward (Rx+)
        eulars.x() -= M_PI * delta_rel_y;
        // Leftward (delta_x < 0): yaw leftward (Ry-)
        eulars.y() += M_PI * delta_rel_x;
        // Angular Velocity
        eulars.x() *= mouse_left_sensitivity_;
        eulars.y() *= mouse_left_sensitivity_;
    }

    // Right held
    if (ImGui::GetIO().MouseDown[1])
    {
        // Leftward (delta_x < 0): roll counterclockwise (Rz-)
        eulars.z() += M_PI * (delta_rel_x < 0.0f ? -delta_l : delta_l);
        // Angular Velocity
        eulars.z() *= mouse_right_sensitivity_;
    }
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //-------------

    //---Translation---
    // Middle held
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    if (ImGui::GetIO().MouseDown[2])
    {
        translating.x() += delta_rel_x;
        translating.y() -= delta_rel_y;
        // Velocity
        translating *= mouse_middle_sensitivity_;
    }
    //-------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() += (R * translating);
    Twc_main_.setRotationMatrix(R * rotating);
}

void ImGuiViewer::keyboardEvent()
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    // R: Reset camera view to init
    if (ImGui::IsKeyPressed(ImGuiKey_R))
        reset_main_to_init_ = true;

    //---Translation---
    Eigen::Vector3f translating = Eigen::Vector3f::Zero();
    // W: forward
    if (ImGui::IsKeyDown(ImGuiKey_W))
        translating.z() += 1.0f;
    // S: backward
    if (ImGui::IsKeyDown(ImGuiKey_S))
        translating.z() -= 1.0f;
    // A: leftward
    if (ImGui::IsKeyDown(ImGuiKey_A))
        translating.x() -= 1.0f;
    // D: rightward
    if (ImGui::IsKeyDown(ImGuiKey_D))
        translating.x() += 1.0f;
    // Velocity
    translating *= keyboard_velocity_;
    //-------------

    //---Rotation---
    Eigen::Vector3f eulars = Eigen::Vector3f::Zero();
    // I: pitch upward (Rx+)
    if (ImGui::IsKeyDown(ImGuiKey_I))
        eulars.x() += M_PI;
    // K: pitch downward (Rx-)
    if (ImGui::IsKeyDown(ImGuiKey_K))
        eulars.x() -= M_PI;
    // J: yaw leftward (Ry-)
    if (ImGui::IsKeyDown(ImGuiKey_J))
        eulars.y() -= M_PI;
    // L: yaw rightward (Ry+)
    if (ImGui::IsKeyDown(ImGuiKey_L))
        eulars.y() += M_PI;
    // U: roll counterclockwise (Rz-)
    if (ImGui::IsKeyDown(ImGuiKey_U))
        eulars.z() -= M_PI;
    // O: roll clockwise (Rz+)
    if (ImGui::IsKeyDown(ImGuiKey_O))
        eulars.z() += M_PI;
    // Angular Velocity
    eulars *= keyboard_anglular_velocity_;
    // To rotation matrix
    Eigen::AngleAxisf roll_angle(eulars.z(), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yaw_angle(eulars.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitch_angle(eulars.x(), Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = roll_angle * yaw_angle * pitch_angle;
    Eigen::Matrix3f rotating = q.matrix();
    //--------------

    //---Apply---
    Eigen::Matrix3f R = Twc_main_.rotationMatrix();
    Twc_main_.translation() += (R * translating);
    Twc_main_.setRotationMatrix(R * rotating);
}