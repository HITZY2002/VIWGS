/**
 * ROS数据提供者实现
 */

#include "ros_data_provider.h"
#include "depth_anything_v2_estimator.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <cstring>
#include <random>

namespace gaussian_mapping {

ROSDataProvider::ROSDataProvider(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
        : nh_(nh),
            nh_private_(nh_private),
            topic_keyframe_pose_("/vins_estimator/keyframe_pose"),
            topic_keyframe_image_("/vins_estimator/keyframe_image"),
            topic_keyframe_point_("/vins_estimator/keyframe_point"),
            last_processed_index_(0),
            fx_(0), fy_(0), cx_(0), cy_(0),
            width_(0), height_(0),
            intrinsics_set_(false),
            initialized_(false),
            finished_(false),
        sync_time_threshold_(0.03),  // 默认30ms同步阈值，兼顾不同数据集时间抖动
    max_buffer_size_(300),
    min_keyframes_to_start_(5),
    min_point_range_(0.1f),
    max_point_range_(50.0f),
    pointcloud_in_camera_frame_(false),
    pose_is_Tcw_(false),
    undistort_keyframes_(false),
    alignment_debug_(true),
    alignment_debug_interval_(30),
    alignment_debug_count_(0),
    scale_stable_(false),
    scale_check_window_(20),
    scale_std_ratio_th_(0.3),
    scale_jump_ratio_th_(2.0),
    scale_min_translation_(0.01),
    scale_debug_(true),
    scale_debug_interval_(30),
    scale_debug_count_(0),
    last_cam_pos_(Eigen::Vector3d::Zero()),
    last_cam_pos_valid_(false),
    drop_first_keyframes_(30),
    dropped_keyframes_(0),
    max_keyframes_in_memory_(50),
    enable_kf_laplacian_filter_(true),
    enable_kf_pose_filter_(true),
    laplacian_threshold_(100.0),
    kf_min_translation_(0.05),
    kf_min_rotation_deg_(3.0),
    last_accepted_pose_valid_(false),
    last_accepted_cam_pos_(Eigen::Vector3d::Zero()),
    last_accepted_cam_rot_(Eigen::Matrix3d::Identity()),
    enable_scale_check_(true),
    depth_fusion_enable_(false),
    depth_anything_model_path_(""),
    depth_anything_output_type_("inverse_depth"),
    depth_anything_input_size_(518),
    depth_anything_device_("cuda"),
    depth_fusion_target_points_per_kf_(3000),
    depth_fusion_max_new_points_(3500),
    depth_fusion_min_sparse_matches_(30),
    depth_fusion_scale_trust_min_sparse_matches_(30),
    depth_fusion_scale_ema_fallback_min_sparse_matches_(8),
    depth_fusion_min_depth_(0.2f),
    depth_fusion_max_depth_(6.0f),
    depth_fusion_edge_grad_threshold_(0.03f),
    depth_fusion_scale_ema_enable_(true),
    depth_fusion_scale_ema_alpha_(0.20f),
    depth_fusion_scale_ema_initialized_(false),
    depth_fusion_scale_ema_value_(1.0f),
    depth_fusion_debug_log_(false),
    depth_fusion_async_enable_(true),
    depth_fusion_every_n_keyframes_(3),
    depth_fusion_async_max_queue_(8),
    accepted_keyframe_count_(0),
    depth_fusion_worker_stop_(false)
{
    // 从参数服务器读取话题名（可在 yaml/launch 中覆盖默认值）
    nh_private_.param("topic_keyframe_pose",  topic_keyframe_pose_,  topic_keyframe_pose_);
    nh_private_.param("topic_keyframe_image", topic_keyframe_image_, topic_keyframe_image_);
    nh_private_.param("topic_keyframe_point", topic_keyframe_point_, topic_keyframe_point_);
    const bool use_raw_image_topic = (topic_keyframe_image_.find("image_raw") != std::string::npos);

    // 订阅VINS话题
    sub_keyframe_pose_ = nh_.subscribe(
        topic_keyframe_pose_, 100,
        &ROSDataProvider::keyframePoseCallback, this);

    sub_keyframe_image_ = nh_.subscribe(
        topic_keyframe_image_, 100,
        &ROSDataProvider::keyframeImageCallback, this);

    sub_keyframe_pointcloud_ = nh_.subscribe(
        topic_keyframe_point_, 100,
        &ROSDataProvider::keyframePointCloudCallback, this);
    
    nh_private_.param("point_min_range", min_point_range_, min_point_range_);
    nh_private_.param("point_max_range", max_point_range_, max_point_range_);
    nh_private_.param("sync_time_threshold", sync_time_threshold_, sync_time_threshold_);
    int max_buffer_size_param = static_cast<int>(max_buffer_size_);
    nh_private_.param("max_buffer_size", max_buffer_size_param, max_buffer_size_param);
    max_buffer_size_ = static_cast<size_t>(std::max(50, max_buffer_size_param));
    const bool sync_threshold_overridden = nh_private_.hasParam("sync_time_threshold");
    const bool max_buffer_overridden = nh_private_.hasParam("max_buffer_size");
    if (use_raw_image_topic) {
        if (!sync_threshold_overridden) {
            sync_time_threshold_ = std::max(sync_time_threshold_, 0.08);
        }
        if (!max_buffer_overridden) {
            max_buffer_size_ = std::max(max_buffer_size_, static_cast<size_t>(1200));
        }
    }
    nh_private_.param("drop_first_keyframes", drop_first_keyframes_, drop_first_keyframes_);
    nh_private_.param("min_keyframes_to_start", min_keyframes_to_start_, min_keyframes_to_start_);
    nh_private_.param("pointcloud_in_camera_frame", pointcloud_in_camera_frame_, pointcloud_in_camera_frame_);
    nh_private_.param("pose_is_Tcw", pose_is_Tcw_, pose_is_Tcw_);
    nh_private_.param("undistort_keyframes", undistort_keyframes_, undistort_keyframes_);
    nh_private_.getParam("dist_coeffs", dist_coeffs_);
    nh_private_.param("alignment_debug", alignment_debug_, alignment_debug_);
    nh_private_.param("alignment_debug_interval", alignment_debug_interval_, alignment_debug_interval_);
    nh_private_.param("scale_check_window", scale_check_window_, scale_check_window_);
    nh_private_.param("scale_std_ratio_th", scale_std_ratio_th_, scale_std_ratio_th_);
    nh_private_.param("scale_jump_ratio_th", scale_jump_ratio_th_, scale_jump_ratio_th_);
    nh_private_.param("scale_min_translation", scale_min_translation_, scale_min_translation_);
    nh_private_.param("scale_debug", scale_debug_, scale_debug_);
    nh_private_.param("scale_debug_interval", scale_debug_interval_, scale_debug_interval_);
    nh_private_.param("max_keyframes_in_memory", max_keyframes_in_memory_, max_keyframes_in_memory_);
    nh_private_.param("enable_scale_check",         enable_scale_check_,         enable_scale_check_);
    nh_private_.param("enable_kf_laplacian_filter",  enable_kf_laplacian_filter_,  enable_kf_laplacian_filter_);
    nh_private_.param("enable_kf_pose_filter",       enable_kf_pose_filter_,       enable_kf_pose_filter_);
    nh_private_.param("laplacian_threshold",         laplacian_threshold_,         laplacian_threshold_);
    nh_private_.param("kf_min_translation",          kf_min_translation_,          kf_min_translation_);
    nh_private_.param("kf_min_rotation_deg",         kf_min_rotation_deg_,         kf_min_rotation_deg_);
    nh_private_.param("depth_fusion_enable", depth_fusion_enable_, depth_fusion_enable_);
    nh_private_.param("depth_anything_model_path", depth_anything_model_path_, depth_anything_model_path_);
    nh_private_.param("depth_anything_output_type", depth_anything_output_type_, depth_anything_output_type_);
    nh_private_.param("depth_anything_input_size", depth_anything_input_size_, depth_anything_input_size_);
    nh_private_.param("depth_anything_device", depth_anything_device_, depth_anything_device_);
    nh_private_.param("depth_fusion_target_points_per_kf", depth_fusion_target_points_per_kf_, depth_fusion_target_points_per_kf_);
    nh_private_.param("depth_fusion_max_new_points", depth_fusion_max_new_points_, depth_fusion_max_new_points_);
    nh_private_.param("depth_fusion_min_sparse_matches", depth_fusion_min_sparse_matches_, depth_fusion_min_sparse_matches_);
    nh_private_.param("depth_fusion_scale_trust_min_sparse_matches", depth_fusion_scale_trust_min_sparse_matches_, depth_fusion_min_sparse_matches_);
    const int default_ema_fallback_sparse_matches = std::max(1, depth_fusion_scale_trust_min_sparse_matches_ / 3);
    nh_private_.param("depth_fusion_scale_ema_fallback_min_sparse_matches", depth_fusion_scale_ema_fallback_min_sparse_matches_, default_ema_fallback_sparse_matches);
    nh_private_.param("depth_fusion_min_depth", depth_fusion_min_depth_, depth_fusion_min_depth_);
    nh_private_.param("depth_fusion_max_depth", depth_fusion_max_depth_, depth_fusion_max_depth_);
    nh_private_.param("depth_fusion_edge_grad_threshold", depth_fusion_edge_grad_threshold_, depth_fusion_edge_grad_threshold_);
    nh_private_.param("depth_fusion_scale_ema_enable", depth_fusion_scale_ema_enable_, depth_fusion_scale_ema_enable_);
    nh_private_.param("depth_fusion_scale_ema_alpha", depth_fusion_scale_ema_alpha_, depth_fusion_scale_ema_alpha_);
    nh_private_.param("depth_fusion_debug_log", depth_fusion_debug_log_, depth_fusion_debug_log_);
    nh_private_.param("depth_fusion_async_enable", depth_fusion_async_enable_, depth_fusion_async_enable_);
    nh_private_.param("depth_fusion_every_n_keyframes", depth_fusion_every_n_keyframes_, depth_fusion_every_n_keyframes_);
    nh_private_.param("depth_fusion_async_max_queue", depth_fusion_async_max_queue_, depth_fusion_async_max_queue_);

    if (depth_fusion_every_n_keyframes_ <= 0) {
        depth_fusion_every_n_keyframes_ = 1;
    }
    if (depth_fusion_async_max_queue_ <= 0) {
        depth_fusion_async_max_queue_ = 1;
    }
    depth_fusion_scale_trust_min_sparse_matches_ = std::max(1, depth_fusion_scale_trust_min_sparse_matches_);
    depth_fusion_scale_ema_fallback_min_sparse_matches_ = std::max(
        1,
        std::min(depth_fusion_scale_ema_fallback_min_sparse_matches_,
                 depth_fusion_scale_trust_min_sparse_matches_));
    depth_fusion_scale_ema_alpha_ = std::max(0.0f, std::min(1.0f, depth_fusion_scale_ema_alpha_));

    if (depth_fusion_enable_) {
        depth_estimator_ = std::make_unique<DepthAnythingV2Estimator>();
        if (!depth_estimator_->loadModel(depth_anything_model_path_, depth_anything_input_size_, depth_anything_device_)) {
            ROS_WARN("[ROSDataProvider] Depth fusion requested but model init failed, disable fusion.");
            depth_fusion_enable_ = false;
            depth_estimator_.reset();
        }
        if (depth_fusion_enable_ && depth_fusion_async_enable_) {
            depth_fusion_worker_ = std::thread(&ROSDataProvider::depthFusionWorkerLoop, this);
        }
    }

    ROS_INFO("[ROSDataProvider] Initialized, waiting for VINS keyframes...");
    ROS_INFO("[ROSDataProvider] Subscribing: pose=%s  image=%s  point=%s",
             topic_keyframe_pose_.c_str(), topic_keyframe_image_.c_str(), topic_keyframe_point_.c_str());
    if (use_raw_image_topic) {
        ROS_WARN("[ROSDataProvider] topic_keyframe_image=%s (raw). Ensure this is RGB source if you need color; sync threshold/buffer are tuned for raw stream.",
                 topic_keyframe_image_.c_str());
    }
    ROS_INFO("[ROSDataProvider] Point range filter: [%.2f, %.2f] meters", min_point_range_, max_point_range_);
    ROS_INFO("[ROSDataProvider] Sync time threshold: %.6f s", sync_time_threshold_);
    ROS_INFO("[ROSDataProvider] Pointcloud frame: %s", pointcloud_in_camera_frame_ ? "camera" : "world");
    ROS_INFO("[ROSDataProvider] Pose convention: %s", pose_is_Tcw_ ? "T_cw" : "T_wc");
    ROS_INFO("[ROSDataProvider] Min keyframes to start: %d", min_keyframes_to_start_);
    if (undistort_keyframes_) {
        ROS_WARN("[ROSDataProvider] undistort_keyframes is enabled but undistortion is handled in GaussianMapper. Disable undistort here.");
        undistort_keyframes_ = false;
    }
    if (alignment_debug_) {
        ROS_INFO("[ROSDataProvider] Alignment debug enabled, interval=%d", alignment_debug_interval_);
    }
    if (scale_debug_) {
        ROS_INFO("[ROSDataProvider] Scale debug enabled, window=%d std_th=%.3f jump_th=%.3f min_trans=%.3f interval=%d",
                 scale_check_window_, scale_std_ratio_th_, scale_jump_ratio_th_, scale_min_translation_, scale_debug_interval_);
    }
    ROS_INFO("[ROSDataProvider] Drop first keyframes: %d", drop_first_keyframes_);
    ROS_INFO("[ROSDataProvider] Max keyframes in memory: %d", max_keyframes_in_memory_);
    ROS_INFO("[ROSDataProvider] Scale check: %s | KF laplacian filter: %s | KF pose filter: %s",
             enable_scale_check_ ? "ON" : "OFF",
             enable_kf_laplacian_filter_ ? "ON" : "OFF",
             enable_kf_pose_filter_ ? "ON" : "OFF");
    ROS_INFO("[ROSDataProvider] KF quality filter: laplacian_th=%.1f min_trans=%.3fm min_rot=%.1fdeg",
             laplacian_threshold_, kf_min_translation_, kf_min_rotation_deg_);
    ROS_INFO("[ROSDataProvider] Depth fusion: %s target=%d max_new=%d depth=[%.2f, %.2f]",
             depth_fusion_enable_ ? "ON" : "OFF",
             depth_fusion_target_points_per_kf_,
             depth_fusion_max_new_points_,
             depth_fusion_min_depth_, depth_fusion_max_depth_);
    ROS_INFO("[ROSDataProvider] Depth fusion edge reject: grad_th=%.4f", depth_fusion_edge_grad_threshold_);
    ROS_INFO("[ROSDataProvider] Depth fusion scale EMA: %s alpha=%.3f",
             depth_fusion_scale_ema_enable_ ? "ON" : "OFF",
             depth_fusion_scale_ema_alpha_);
    ROS_INFO("[ROSDataProvider] Depth fusion scale sparse thresholds: trust_raw>=%d, ema_fallback<%d",
             depth_fusion_scale_trust_min_sparse_matches_,
             depth_fusion_scale_ema_fallback_min_sparse_matches_);
    ROS_INFO("[ROSDataProvider] Depth fusion mode: %s every_n=%d async_queue=%d",
             depth_fusion_async_enable_ ? "ASYNC" : "SYNC",
             depth_fusion_every_n_keyframes_, depth_fusion_async_max_queue_);
    ROS_INFO("[ROSDataProvider] DepthAnything output type: %s", depth_anything_output_type_.c_str());

}

ROSDataProvider::~ROSDataProvider()
{
    {
        std::lock_guard<std::mutex> lock(depth_fusion_mutex_);
        depth_fusion_worker_stop_ = true;
    }
    depth_fusion_cv_.notify_all();
    if (depth_fusion_worker_.joinable()) {
        depth_fusion_worker_.join();
    }
}

bool ROSDataProvider::buildUndistortParams(cv::Mat& K, cv::Mat& dist) const
{
    if (!intrinsics_set_ || dist_coeffs_.size() < 4) {
        return false;
    }

    K = (cv::Mat_<double>(3, 3) <<
        fx_, 0.0, cx_,
        0.0, fy_, cy_,
        0.0, 0.0, 1.0);

    if (dist_coeffs_.size() >= 5) {
        dist = (cv::Mat_<double>(1, 5) <<
            dist_coeffs_[0], dist_coeffs_[1], dist_coeffs_[2], dist_coeffs_[3], dist_coeffs_[4]);
    } else {
        dist = (cv::Mat_<double>(1, 4) <<
            dist_coeffs_[0], dist_coeffs_[1], dist_coeffs_[2], dist_coeffs_[3]);
    }

    return true;
}

// ==================================================================================
//  ROS 回调函数
// ==================================================================================

void ROSDataProvider::keyframePoseCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double timestamp = msg->header.stamp.toSec();
    pose_buffer_[timestamp] = *msg;
    
    // 尝试组装关键帧
    assembleKeyFrame(timestamp);
    
    // 清理过期缓存
    while (pose_buffer_.size() > max_buffer_size_) {
        pose_buffer_.erase(pose_buffer_.begin());
    }
}

void ROSDataProvider::keyframeImageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double timestamp = msg->header.stamp.toSec();
    image_buffer_[timestamp] = *msg;
    
    // 尝试组装关键帧
    assembleKeyFrame(timestamp);
    
    // 清理过期缓存
    while (image_buffer_.size() > max_buffer_size_) {
        image_buffer_.erase(image_buffer_.begin());
    }
}

void ROSDataProvider::keyframePointCloudCallback(const sensor_msgs::PointCloud::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    double timestamp = msg->header.stamp.toSec();
    pointcloud_buffer_[timestamp] = *msg;
    
    // 尝试组装关键帧
    assembleKeyFrame(timestamp);
    
    // 清理过期缓存
    while (pointcloud_buffer_.size() > max_buffer_size_) {
        pointcloud_buffer_.erase(pointcloud_buffer_.begin());
    }
}

// ==================================================================================
//  数据同步与组装
// ==================================================================================

void ROSDataProvider::assembleKeyFrame(double timestamp)
{
    // mutex_ 已在调用函数中加锁
    
    // 检查是否有对应的pose、image、pointcloud
    auto pose_it = pose_buffer_.find(timestamp);
    auto image_it = image_buffer_.find(timestamp);
    auto pc_it = pointcloud_buffer_.find(timestamp);
    
    // 如果三者都不存在，尝试时间对齐
    if (pose_it == pose_buffer_.end() || 
        image_it == image_buffer_.end() || 
        pc_it == pointcloud_buffer_.end()) {

        auto findNearestImage = [&](double t_ref) {
            auto best = image_buffer_.end();
            double best_diff = std::numeric_limits<double>::infinity();
            auto lb = image_buffer_.lower_bound(t_ref);
            if (lb != image_buffer_.end()) {
                double d = std::abs(lb->first - t_ref);
                if (d < best_diff) { best_diff = d; best = lb; }
            }
            if (lb != image_buffer_.begin()) {
                auto prev = std::prev(lb);
                double d = std::abs(prev->first - t_ref);
                if (d < best_diff) { best_diff = d; best = prev; }
            }
            return std::make_pair(best, best_diff);
        };

        auto findNearestPoint = [&](double t_ref) {
            auto best = pointcloud_buffer_.end();
            double best_diff = std::numeric_limits<double>::infinity();
            auto lb = pointcloud_buffer_.lower_bound(t_ref);
            if (lb != pointcloud_buffer_.end()) {
                double d = std::abs(lb->first - t_ref);
                if (d < best_diff) { best_diff = d; best = lb; }
            }
            if (lb != pointcloud_buffer_.begin()) {
                auto prev = std::prev(lb);
                double d = std::abs(prev->first - t_ref);
                if (d < best_diff) { best_diff = d; best = prev; }
            }
            return std::make_pair(best, best_diff);
        };

        // 在所有 pose 中寻找“误差最小”的一组匹配，而不是找到第一组就返回
        double best_score = std::numeric_limits<double>::infinity();
        auto best_pose_it = pose_buffer_.end();
        auto best_image_it = image_buffer_.end();
        auto best_pc_it = pointcloud_buffer_.end();

        for (auto it_pose = pose_buffer_.begin(); it_pose != pose_buffer_.end(); ++it_pose) {
            const double t_pose = it_pose->first;
            auto image_pair = findNearestImage(t_pose);
            auto pc_pair = findNearestPoint(t_pose);
            if (image_pair.first == image_buffer_.end() || pc_pair.first == pointcloud_buffer_.end()) {
                continue;
            }
            if (image_pair.second > sync_time_threshold_ || pc_pair.second > sync_time_threshold_) {
                continue;
            }

            const double score = image_pair.second + pc_pair.second;
            if (score < best_score) {
                best_score = score;
                best_pose_it = it_pose;
                best_image_it = image_pair.first;
                best_pc_it = pc_pair.first;
            }
        }

        if (best_pose_it != pose_buffer_.end()) {
            pose_it = best_pose_it;
            image_it = best_image_it;
            pc_it = best_pc_it;
            timestamp = pose_it->first;
        }
    }
    
    // 如果找不到匹配数据，返回
    if (pose_it == pose_buffer_.end() || 
        image_it == image_buffer_.end() || 
        pc_it == pointcloud_buffer_.end()) {
        static size_t sync_miss_counter = 0;
        ++sync_miss_counter;
        if (sync_miss_counter % 200 == 0) {
            auto nearestDiff = [&](const auto& buffer) {
                if (buffer.empty()) return std::numeric_limits<double>::infinity();
                double best = std::numeric_limits<double>::infinity();
                auto lb = buffer.lower_bound(timestamp);
                if (lb != buffer.end()) best = std::min(best, std::abs(lb->first - timestamp));
                if (lb != buffer.begin()) best = std::min(best, std::abs(std::prev(lb)->first - timestamp));
                return best;
            };
            const double d_pose = nearestDiff(pose_buffer_);
            const double d_img = nearestDiff(image_buffer_);
            const double d_pc = nearestDiff(pointcloud_buffer_);
            ROS_WARN("[ROSDataProvider][Sync] misses=%zu threshold=%.4f nearest_dt(pose/img/pc)=%.4f/%.4f/%.4f buf(pose/img/pc)=%zu/%zu/%zu",
                     sync_miss_counter, sync_time_threshold_, d_pose, d_img, d_pc,
                     pose_buffer_.size(), image_buffer_.size(), pointcloud_buffer_.size());
        }
        return;
    }
    
    // ========== 组装关键帧 ==========
    const double t_pose = pose_it->first;
    const double t_img = image_it->first;
    const double t_pc = pc_it->first;
    auto kf = std::make_shared<KeyFrameData>();
    kf->timestamp = timestamp;
    
    // 提取位姿
    Eigen::Vector3d position(
        pose_it->second.pose.pose.position.x,
        pose_it->second.pose.pose.position.y,
        pose_it->second.pose.pose.position.z
    );
    Eigen::Quaterniond quaternion(
        pose_it->second.pose.pose.orientation.w,
        pose_it->second.pose.pose.orientation.x,
        pose_it->second.pose.pose.orientation.y,
        pose_it->second.pose.pose.orientation.z
    );
    
    // 转换为 SE3f (world to camera)
    Sophus::SE3d T_msg(quaternion, position);
    Sophus::SE3d T_cw = pose_is_Tcw_ ? T_msg : T_msg.inverse();
    kf->pose = T_cw.cast<float>();
    
    // 提取图像
    std::string saved_filename;
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(
            boost::make_shared<sensor_msgs::Image>(image_it->second));
        cv::Mat img = cv_ptr->image;
        if (img.empty()) {
            ROS_WARN("[ROSDataProvider] Empty keyframe image at t=%.3f", timestamp);
            return;
        }

        cv::Mat img_rgb;
        if (img.channels() == 1) {
            cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);
        } else if (img.channels() == 3) {
            // 默认 VINS 输出为 BGR，统一转换为 RGB
            cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
        } else if (img.channels() == 4) {
            cv::cvtColor(img, img_rgb, cv::COLOR_BGRA2RGB);
        } else {
            ROS_WARN("[ROSDataProvider] Unsupported image channels: %d", img.channels());
            return;
        }

        if (undistort_keyframes_ && intrinsics_set_) {
            cv::Mat K = (cv::Mat_<double>(3, 3) <<
                fx_, 0.0, cx_,
                0.0, fy_, cy_,
                0.0, 0.0, 1.0);
            cv::Mat dist;
            if (dist_coeffs_.size() >= 5) {
                dist = (cv::Mat_<double>(1, 5) <<
                    dist_coeffs_[0], dist_coeffs_[1], dist_coeffs_[2], dist_coeffs_[3], dist_coeffs_[4]);
            } else {
                dist = (cv::Mat_<double>(1, 4) <<
                    dist_coeffs_[0], dist_coeffs_[1], dist_coeffs_[2], dist_coeffs_[3]);
            }
            cv::Mat img_undist;
            cv::undistort(img_rgb, img_undist, K, dist);
            img_rgb = img_undist;
        }

        if (img_rgb.type() != CV_32FC3) {
            img_rgb.convertTo(img_rgb, CV_32FC3, 1.0 / 255.0);
        }

        kf->image = img_rgb;
        kf->width = kf->image.cols;
        kf->height = kf->image.rows;

        (void)saved_filename;
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("[ROSDataProvider] cv_bridge exception: %s", e.what());
        return;
    }

    // 提取点云（过滤NaN与异常远点）
    size_t total_points = 0;
    size_t filtered_points = 0;
    float outlier_max = 0.0f;
    const auto& pc_msg = pc_it->second;
    const auto& channels = pc_msg.channels;
    auto channelIndex = [&](const std::string& name) -> int {
        for (size_t i = 0; i < channels.size(); ++i) {
            if (channels[i].name == name) return static_cast<int>(i);
        }
        return -1;
    };
    const int idx_rgb = channelIndex("rgb");
    const int idx_rgba = channelIndex("rgba");
    const int idx_r = channelIndex("r");
    const int idx_g = channelIndex("g");
    const int idx_b = channelIndex("b");
    const int idx_intensity = channelIndex("intensity");
    auto decodePackedRGB = [](float packed, Eigen::Vector3f& color) {
        uint32_t rgb = 0;
        std::memcpy(&rgb, &packed, sizeof(float));
        uint8_t r = static_cast<uint8_t>((rgb >> 16) & 0xFF);
        uint8_t g = static_cast<uint8_t>((rgb >> 8) & 0xFF);
        uint8_t b = static_cast<uint8_t>(rgb & 0xFF);
        color = Eigen::Vector3f(r, g, b) / 255.0f;
    };
    auto decodeScalar = [](float v) -> float {
        if (!std::isfinite(v)) return 0.5f;
        return (v > 1.0f) ? (v / 255.0f) : v;
    };

    for (size_t i = 0; i < pc_msg.points.size(); ++i) {
        const auto& point = pc_msg.points[i];
        ++total_points;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            ++filtered_points;
            continue;
        }
        Eigen::Vector3f p(point.x, point.y, point.z);
        Eigen::Vector3f p_for_range = p;
        if (!pointcloud_in_camera_frame_) {
            p_for_range = kf->pose * p;
        }
        float n = p_for_range.norm();
        if (n < min_point_range_ || n > max_point_range_) {
            ++filtered_points;
            outlier_max = std::max(outlier_max, n);
            continue;
        }
        Eigen::Vector3f color(0.5f, 0.5f, 0.5f);
        if (idx_rgb >= 0 && i < channels[idx_rgb].values.size()) {
            decodePackedRGB(channels[idx_rgb].values[i], color);
        } else if (idx_rgba >= 0 && i < channels[idx_rgba].values.size()) {
            decodePackedRGB(channels[idx_rgba].values[i], color);
        } else if (idx_r >= 0 && idx_g >= 0 && idx_b >= 0
                   && i < channels[idx_r].values.size()
                   && i < channels[idx_g].values.size()
                   && i < channels[idx_b].values.size()) {
            color.x() = decodeScalar(channels[idx_r].values[i]);
            color.y() = decodeScalar(channels[idx_g].values[i]);
            color.z() = decodeScalar(channels[idx_b].values[i]);
        } else if (idx_intensity >= 0 && i < channels[idx_intensity].values.size()) {
            float v = decodeScalar(channels[idx_intensity].values[i]);
            color = Eigen::Vector3f(v, v, v);
        }

        if (pointcloud_in_camera_frame_) {
            Eigen::Vector3f p_w = kf->pose.inverse() * p;
            kf->points.emplace_back(p_w);
            kf->colors.emplace_back(color);
        } else {
            kf->points.emplace_back(p);
            kf->colors.emplace_back(color);
        }
    }

    if (alignment_debug_ && intrinsics_set_) {
        ++alignment_debug_count_;
        if (alignment_debug_interval_ <= 0) {
            alignment_debug_interval_ = 1;
        }
        if (alignment_debug_count_ % alignment_debug_interval_ == 0 && !kf->points.empty()) {
            size_t proj_total = 0;
            size_t proj_in = 0;
            double sum_z = 0.0;
            double sum_z2 = 0.0;
            double z_min = std::numeric_limits<double>::infinity();
            double z_max = 0.0;
            for (const auto& p : kf->points) {
                Eigen::Vector3f pc = kf->pose * p;
                if (pc.z() <= 0.0f) {
                    continue;
                }
                float u = fx_ * pc.x() / pc.z() + cx_;
                float v = fy_ * pc.y() / pc.z() + cy_;
                ++proj_total;
                sum_z += pc.z();
                sum_z2 += static_cast<double>(pc.z()) * pc.z();
                z_min = std::min(z_min, static_cast<double>(pc.z()));
                z_max = std::max(z_max, static_cast<double>(pc.z()));
                if (u >= 0.0f && u < kf->width && v >= 0.0f && v < kf->height) {
                    ++proj_in;
                }
            }
            double proj_ratio = proj_total ? static_cast<double>(proj_in) / proj_total : 0.0;
            double z_mean = proj_total ? sum_z / proj_total : 0.0;
            double z_var = proj_total ? (sum_z2 / proj_total - z_mean * z_mean) : 0.0;
            double z_std = z_var > 0.0 ? std::sqrt(z_var) : 0.0;
            double dt_img = std::abs(t_pose - t_img);
            double dt_pc = std::abs(t_pose - t_pc);
            ROS_INFO("[ROSDataProvider][Align] proj_in=%zu/%zu ratio=%.3f z[min/mean/max/std]=%.3f/%.3f/%.3f/%.3f dt_img=%.4f dt_pc=%.4f pts=%zu filt=%zu",
                     proj_in, proj_total, proj_ratio, z_min, z_mean, z_max, z_std, dt_img, dt_pc, total_points, filtered_points);
        }
    }

    // 调试输出已移除
    
    // 设置相机内参
    if (intrinsics_set_) {
        kf->fx = fx_;
        kf->fy = fy_;
        kf->cx = cx_;
        kf->cy = cy_;
    }
    
    // // 丢弃前 N 帧，避免早期尺度不稳定
    // if (dropped_keyframes_ < drop_first_keyframes_) {
    //     ++dropped_keyframes_;
    //     return;
    // }

    // ======== 关键帧质量筛选（可选，由配置开关控制） ========

    // 1. 模糊检测：计算灰度图的拉普拉斯方差（enable_kf_laplacian_filter: true 时生效）
    if (enable_kf_laplacian_filter_) {
        cv::Mat gray;
        cv::Mat img_u8;
        // kf->image 是 CV_32FC3 [0,1] 范围
        kf->image.convertTo(img_u8, CV_8UC3, 255.0);
        cv::cvtColor(img_u8, gray, cv::COLOR_RGB2GRAY);
        cv::Mat lap;
        cv::Laplacian(gray, lap, CV_64F);
        cv::Scalar mu, sigma;
        cv::meanStdDev(lap, mu, sigma);
        double laplacian_var = sigma.val[0] * sigma.val[0];
        if (laplacian_var < laplacian_threshold_) {
            ROS_DEBUG("[ROSDataProvider][KFFilter] Blurry frame (lap_var=%.1f < %.1f), dropped.",
                      laplacian_var, laplacian_threshold_);
            return;
        }
    }

    // 2 & 3. 位姿相对变化筛选（enable_kf_pose_filter: true 时生效）
    if (enable_kf_pose_filter_) {
        Sophus::SE3d T_cw_d = kf->pose.cast<double>();
        Eigen::Vector3d cam_pos = T_cw_d.inverse().translation();
        Eigen::Matrix3d cam_rot = T_cw_d.inverse().rotationMatrix();

        if (last_accepted_pose_valid_) {
            // 2. 平移筛选
            double trans_diff = (cam_pos - last_accepted_cam_pos_).norm();
            if (trans_diff < kf_min_translation_) {
                ROS_DEBUG("[ROSDataProvider][KFFilter] Too close (trans=%.4fm < %.3fm), dropped.",
                          trans_diff, kf_min_translation_);
                return;
            }

            // 3. 旋转筛选
            Eigen::Matrix3d R_rel = last_accepted_cam_rot_.transpose() * cam_rot;
            double cos_angle = (R_rel.trace() - 1.0) * 0.5;
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            double rot_deg = std::acos(cos_angle) * 180.0 / M_PI;
            if (rot_deg < kf_min_rotation_deg_) {
                ROS_DEBUG("[ROSDataProvider][KFFilter] Rotation too small (rot=%.2fdeg < %.1fdeg), dropped.",
                          rot_deg, kf_min_rotation_deg_);
                return;
            }
        }

        // 通过筛选，更新上一个被接收的位姿
        last_accepted_cam_pos_ = cam_pos;
        last_accepted_cam_rot_ = cam_rot;
        last_accepted_pose_valid_ = true;
    }

    // 补点按帧率限流；异步模式下先上传稀疏帧，补点完成后再增量回推
    ++accepted_keyframe_count_;
    const bool do_depth_fusion = depth_fusion_enable_
        && (depth_fusion_every_n_keyframes_ <= 1
            || (accepted_keyframe_count_ % depth_fusion_every_n_keyframes_ == 0));
    if (do_depth_fusion) {
        if (depth_fusion_async_enable_) {
            enqueueDepthFusionTask(kf);
        } else {
            augmentSparsePointsWithDepth(kf);
        }
    }

    // 添加到关键帧列表
    keyframes_.push_back(kf);

    // 超出上限则丢弃最旧关键帧
    if (max_keyframes_in_memory_ > 0 && keyframes_.size() > static_cast<size_t>(max_keyframes_in_memory_)) {
        size_t remove_count = keyframes_.size() - static_cast<size_t>(max_keyframes_in_memory_);
        keyframes_.erase(keyframes_.begin(), keyframes_.begin() + static_cast<long>(remove_count));
        if (last_processed_index_ >= remove_count) {
            last_processed_index_ -= remove_count;
        } else {
            last_processed_index_ = 0;
        }
    }
    
    // 从缓存中移除已使用的数据
    pose_buffer_.erase(pose_it);
    image_buffer_.erase(image_it);
    pointcloud_buffer_.erase(pc_it);
    
    // ======== 尺度稳定性检查 ========
    {
        Sophus::SE3d T_cw_d = kf->pose.cast<double>();
        Eigen::Vector3d cam_pos = T_cw_d.inverse().translation();
        if (last_cam_pos_valid_) {
            double delta = (cam_pos - last_cam_pos_).norm();
            if (delta > scale_min_translation_) {
                recent_translations_.push_back(delta);
                if (recent_translations_.size() > static_cast<size_t>(scale_check_window_)) {
                    recent_translations_.pop_front();
                }
            }
        }
        last_cam_pos_ = cam_pos;
        last_cam_pos_valid_ = true;

        if (recent_translations_.size() >= static_cast<size_t>(scale_check_window_)) {
            double sum = 0.0;
            double max_v = 0.0;
            for (double v : recent_translations_) {
                sum += v;
                max_v = std::max(max_v, v);
            }
            double mean = sum / recent_translations_.size();
            double var = 0.0;
            for (double v : recent_translations_) {
                double d = v - mean;
                var += d * d;
            }
            var /= recent_translations_.size();
            double std = std::sqrt(var);
            if (mean > 1e-9) {
                double std_ratio = std / mean;
                double max_ratio = max_v / mean;
                scale_stable_ = (std_ratio <= scale_std_ratio_th_) && (max_ratio <= scale_jump_ratio_th_);

                if (scale_debug_) {
                    ++scale_debug_count_;
                    int interval = scale_debug_interval_ <= 0 ? 1 : scale_debug_interval_;
                    if (scale_debug_count_ % interval == 0) {
                        ROS_INFO("[ROSDataProvider][Scale] mean=%.4f std=%.4f std_ratio=%.3f max=%.4f max_ratio=%.3f stable=%d",
                                 mean, std, std_ratio, max_v, max_ratio, scale_stable_ ? 1 : 0);
                    }
                }
            } else {
                scale_stable_ = false;
            }
        } else {
            scale_stable_ = false;
        }
    }

    // 标记为已初始化
    // enable_scale_check_=true 时需要尺度稳定才初始化，false 时帧数足够即可
    bool scale_ok = !enable_scale_check_ || scale_stable_;
    if (!initialized_ && keyframes_.size() >= static_cast<size_t>(min_keyframes_to_start_) && scale_ok) {
        initialized_ = true;
        ROS_INFO("[ROSDataProvider] Initialized with %zu keyframes (%s)",
                 keyframes_.size(),
                 enable_scale_check_ ? "scale stable" : "scale check disabled");
    }
    
    ROS_DEBUG("[ROSDataProvider] Assembled keyframe %zu at t=%.3f", 
              keyframes_.size(), timestamp);
}

void ROSDataProvider::augmentSparsePointsWithDepth(std::shared_ptr<KeyFrameData>& kf)
{
    std::vector<Eigen::Vector3f> added_points;
    std::vector<Eigen::Vector3f> added_colors;
    if (!generateDepthFusionPoints(kf, added_points, added_colors)) {
        return;
    }

    kf->points.insert(kf->points.end(), added_points.begin(), added_points.end());
    kf->colors.insert(kf->colors.end(), added_colors.begin(), added_colors.end());
}

bool ROSDataProvider::generateDepthFusionPoints(
    const std::shared_ptr<KeyFrameData>& kf,
    std::vector<Eigen::Vector3f>& out_points,
    std::vector<Eigen::Vector3f>& out_colors)
{
    out_points.clear();
    out_colors.clear();

    if (!depth_fusion_enable_ || !depth_estimator_ || !depth_estimator_->isReady()) {
        return false;
    }
    if (!kf || kf->image.empty() || !intrinsics_set_) {
        return false;
    }

    const int target_points = std::max(0, depth_fusion_target_points_per_kf_);
    if (static_cast<int>(kf->points.size()) >= target_points) {
        return false;
    }

    const size_t base_sparse_count = kf->points.size();

    cv::Mat image_for_fusion = kf->image;
    cv::Mat K_undist, dist_undist;
    if (buildUndistortParams(K_undist, dist_undist)) {
        cv::Mat image_undist;
        cv::undistort(kf->image, image_undist, K_undist, dist_undist);
        if (!image_undist.empty()) {
            image_for_fusion = image_undist;
        }
    }

    cv::Mat depth_rel;
    if (!depth_estimator_->inferDepth(image_for_fusion, depth_rel) || depth_rel.empty()) {
        return false;
    }

    cv::Mat depth_for_fusion = depth_rel;
    if (buildUndistortParams(K_undist, dist_undist)) {
        cv::Mat depth_undist;
        cv::undistort(depth_rel, depth_undist, K_undist, dist_undist);
        if (!depth_undist.empty()) {
            depth_for_fusion = depth_undist;
        }
    }
    kf->depth_image = depth_for_fusion.clone();

    cv::Mat valid_depth_mask = (depth_for_fusion > 1e-6f) & (depth_for_fusion == depth_for_fusion);
    cv::Mat depth_for_grad = depth_for_fusion.clone();
    cv::patchNaNs(depth_for_grad, 0.0f);
    cv::Mat invalid_depth_mask;
    cv::bitwise_not(valid_depth_mask, invalid_depth_mask);
    depth_for_grad.setTo(0.0f, invalid_depth_mask);
    cv::GaussianBlur(depth_for_grad, depth_for_grad, cv::Size(3, 3), 0.0, 0.0, cv::BORDER_REPLICATE);
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(depth_for_grad, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(depth_for_grad, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    std::string out_type = depth_anything_output_type_;
    std::transform(out_type.begin(), out_type.end(), out_type.begin(), ::tolower);
    const bool is_inverse_depth = (out_type == "inverse" || out_type == "inv" || out_type == "inverse_depth");

    std::vector<float> scale_candidates;
    scale_candidates.reserve(kf->points.size());

    size_t sparse_total = 0;
    size_t sparse_behind = 0;
    size_t sparse_oob = 0;
    size_t sparse_invalid_depth = 0;
    size_t sparse_valid = 0;

    for (const auto& pw : kf->points) {
        ++sparse_total;
        Eigen::Vector3f pc = kf->pose * pw;
        if (pc.z() <= 1e-4f) {
            ++sparse_behind;
            continue;
        }

        const int u = static_cast<int>(std::lround(fx_ * pc.x() / pc.z() + cx_));
        const int v = static_cast<int>(std::lround(fy_ * pc.y() / pc.z() + cy_));
        if (u < 0 || v < 0 || u >= depth_for_fusion.cols || v >= depth_for_fusion.rows) {
            ++sparse_oob;
            continue;
        }

        const float d = depth_for_fusion.at<float>(v, u);
        if (!std::isfinite(d) || d <= 1e-6f) {
            ++sparse_invalid_depth;
            continue;
        }
        ++sparse_valid;
        if (is_inverse_depth) {
            scale_candidates.push_back(pc.z() * d);
        } else {
            scale_candidates.push_back(pc.z() / d);
        }
    }

    const int sparse_matches = static_cast<int>(scale_candidates.size());
    if (sparse_matches <= 0) {
        if (depth_fusion_debug_log_) {
            ROS_WARN("[ROSDataProvider][DepthFusion] No valid sparse matches for scale estimation | sparse(total=%zu valid=%zu behind=%zu oob=%zu invalid_depth=%zu) intrinsics_set=%d image=%dx%d depth=%dx%d",
                     sparse_total, sparse_valid, sparse_behind, sparse_oob, sparse_invalid_depth,
                     intrinsics_set_ ? 1 : 0,
                     kf->width, kf->height,
                     depth_for_fusion.cols, depth_for_fusion.rows);
        }
        return false;
    }

    std::nth_element(
        scale_candidates.begin(),
        scale_candidates.begin() + scale_candidates.size() / 2,
        scale_candidates.end());
    const float scale_raw = scale_candidates[scale_candidates.size() / 2];
    if (!std::isfinite(scale_raw) || scale_raw <= 1e-6f) {
        return false;
    }

    float scale = scale_raw;
    const bool sparse_is_trusted =
        (sparse_matches >= depth_fusion_scale_trust_min_sparse_matches_);
    const bool sparse_is_extremely_low =
        (sparse_matches < depth_fusion_scale_ema_fallback_min_sparse_matches_);

    if (depth_fusion_scale_ema_enable_) {
        std::lock_guard<std::mutex> lock(depth_fusion_scale_ema_mutex_);
        if (!depth_fusion_scale_ema_initialized_) {
            depth_fusion_scale_ema_value_ = scale_raw;
            depth_fusion_scale_ema_initialized_ = true;
        }

        if (sparse_is_extremely_low) {
            depth_fusion_scale_ema_value_ =
                depth_fusion_scale_ema_alpha_ * scale_raw +
                (1.0f - depth_fusion_scale_ema_alpha_) * depth_fusion_scale_ema_value_;
            scale = depth_fusion_scale_ema_value_;
        } else {
            // 内点不极少时直接信任当前帧尺度，允许大视角切换带来的尺度突变。
            scale = scale_raw;
            // 同步EMA状态，确保后续极少内点帧有可靠兜底基线。
            depth_fusion_scale_ema_value_ = scale_raw;
        }
    }

    if (depth_fusion_debug_log_ && sparse_is_extremely_low) {
        ROS_WARN("[ROSDataProvider][DepthFusion] Extremely low sparse matches (%d < %d), using EMA fallback scale=%.6f (raw=%.6f)",
                 sparse_matches, depth_fusion_scale_ema_fallback_min_sparse_matches_, scale, scale_raw);
    }

    if (depth_fusion_debug_log_ && !sparse_is_trusted) {
        ROS_INFO("[ROSDataProvider][DepthFusion] Sparse matches in non-trust zone (%d < %d), use %s scale=%.6f (raw=%.6f)",
                 sparse_matches,
                 depth_fusion_scale_trust_min_sparse_matches_,
                 sparse_is_extremely_low ? "EMA fallback" : "raw",
                 scale,
                 scale_raw);
    }

    const int need = std::min(
        std::max(0, target_points - static_cast<int>(kf->points.size())),
        std::max(0, depth_fusion_max_new_points_));
    if (need <= 0) {
        return false;
    }

    const Sophus::SE3f T_wc = kf->pose.inverse();
    const int h = depth_for_fusion.rows;
    const int w = depth_for_fusion.cols;
    const int stride = std::max(1, static_cast<int>(std::sqrt(static_cast<float>(w * h) / static_cast<float>(need * 2 + 1))));

    std::vector<std::pair<int, int>> samples;
    samples.reserve((h / stride + 1) * (w / stride + 1));
    for (int v = stride / 2; v < h; v += stride) {
        for (int u = stride / 2; u < w; u += stride) {
            samples.emplace_back(u, v);
        }
    }

    int added = 0;
    const int n_samples = static_cast<int>(samples.size());
    if (n_samples <= 0) {
        return false;
    }

    std::vector<uint8_t> visited(static_cast<size_t>(n_samples), static_cast<uint8_t>(0));

    auto try_add_sample = [&](int sample_idx) {
        if (sample_idx < 0 || sample_idx >= n_samples) {
            return;
        }
        if (visited[static_cast<size_t>(sample_idx)]) {
            return;
        }
        visited[static_cast<size_t>(sample_idx)] = static_cast<uint8_t>(1);
        if (added >= need) {
            return;
        }

        const auto& uv = samples[static_cast<size_t>(sample_idx)];
        const int u = uv.first;
        const int v = uv.second;
        if (valid_depth_mask.at<uint8_t>(v, u) == 0) {
            return;
        }
        if (grad_mag.at<float>(v, u) > depth_fusion_edge_grad_threshold_) {
            return;
        }
        const float d_rel = depth_for_fusion.at<float>(v, u);
        if (!std::isfinite(d_rel) || d_rel <= 1e-6f) {
            return;
        }

        float z = 0.0f;
        if (is_inverse_depth) {
            z = scale / std::max(1e-6f, d_rel);
        } else {
            z = scale * d_rel;
        }
        if (z < depth_fusion_min_depth_ || z > depth_fusion_max_depth_) {
            return;
        }

        const float x = (static_cast<float>(u) - cx_) * z / fx_;
        const float y = (static_cast<float>(v) - cy_) * z / fy_;
        Eigen::Vector3f pc(x, y, z);
        Eigen::Vector3f pw = T_wc * pc;

        if (!std::isfinite(pw.x()) || !std::isfinite(pw.y()) || !std::isfinite(pw.z())) {
            return;
        }

        const cv::Vec3f rgb = image_for_fusion.at<cv::Vec3f>(v, u);
        Eigen::Vector3f color(
            std::min(1.0f, std::max(0.0f, rgb[0])),
            std::min(1.0f, std::max(0.0f, rgb[1])),
            std::min(1.0f, std::max(0.0f, rgb[2])));

        out_points.emplace_back(pw);
        out_colors.emplace_back(color);
        ++added;
    };

    // 第一阶段：按均匀索引在全图网格上取样，避免随机选择导致局部过密。
    const int uniform_targets = std::min(need, n_samples);
    if (uniform_targets > 0) {
        const double step_idx = static_cast<double>(n_samples) / static_cast<double>(uniform_targets);
        for (int i = 0; i < uniform_targets && added < need; ++i) {
            int idx = static_cast<int>(std::floor((static_cast<double>(i) + 0.5) * step_idx));
            if (idx >= n_samples) {
                idx = n_samples - 1;
            }
            try_add_sample(idx);
        }
    }

    // 第二阶段：若第一阶段受无效深度/边缘拒绝导致数量不足，顺序遍历剩余样本补齐。
    for (int i = 0; i < n_samples && added < need; ++i) {
        try_add_sample(i);
    }

    if (depth_fusion_debug_log_) {
        ROS_INFO("[ROSDataProvider][DepthFusion] sparse=%zu added=%d total=%zu scale_raw=%.4f scale_ema=%.4f mode=%s",
                 base_sparse_count,
                 added,
                 base_sparse_count + static_cast<size_t>(added),
                 scale_raw,
                 scale,
                 is_inverse_depth ? "inverse_depth" : "depth");
    }

    return added > 0;
}

void ROSDataProvider::enqueueDepthFusionTask(const std::shared_ptr<KeyFrameData>& kf)
{
    if (!kf) {
        return;
    }

    auto task = std::make_shared<KeyFrameData>();
    task->timestamp = kf->timestamp;
    task->pose = kf->pose;
    task->image = kf->image;
    task->width = kf->width;
    task->height = kf->height;
    task->fx = kf->fx;
    task->fy = kf->fy;
    task->cx = kf->cx;
    task->cy = kf->cy;
    task->points = kf->points;
    task->colors = kf->colors;
    task->depth_image = kf->depth_image;

    {
        std::lock_guard<std::mutex> lock(depth_fusion_mutex_);
        if (static_cast<int>(depth_fusion_tasks_.size()) >= depth_fusion_async_max_queue_) {
            depth_fusion_tasks_.pop_front();
            if (depth_fusion_debug_log_) {
                ROS_WARN("[ROSDataProvider][DepthFusion] Async queue full, drop oldest task.");
            }
        }
        depth_fusion_tasks_.push_back(task);
    }
    depth_fusion_cv_.notify_one();
}

void ROSDataProvider::depthFusionWorkerLoop()
{
    while (true) {
        std::shared_ptr<KeyFrameData> task;
        {
            std::unique_lock<std::mutex> lock(depth_fusion_mutex_);
            depth_fusion_cv_.wait(lock, [&]() {
                return depth_fusion_worker_stop_ || !depth_fusion_tasks_.empty();
            });

            if (depth_fusion_worker_stop_ && depth_fusion_tasks_.empty()) {
                break;
            }

            task = depth_fusion_tasks_.front();
            depth_fusion_tasks_.pop_front();
        }

        std::vector<Eigen::Vector3f> added_points;
        std::vector<Eigen::Vector3f> added_colors;
        if (!generateDepthFusionPoints(task, added_points, added_colors)) {
            continue;
        }

        auto update = std::make_shared<KeyFrameData>();
        update->timestamp = task->timestamp;
        update->pose = task->pose;
        update->depth_fusion_update = true;
        update->image = task->image;
        update->width = task->width;
        update->height = task->height;
        update->fx = task->fx;
        update->fy = task->fy;
        update->cx = task->cx;
        update->cy = task->cy;
        update->points = std::move(added_points);
        update->colors = std::move(added_colors);
        update->depth_image = task->depth_image;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            depth_fusion_updates_.push_back(update);
        }
    }
}

// ==================================================================================
//  GaussianDataProvider 接口实现
// ==================================================================================

size_t ROSDataProvider::getKeyFrameCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.size();
}

std::shared_ptr<KeyFrameData> ROSDataProvider::getKeyFrame(size_t index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= keyframes_.size()) {
        return nullptr;
    }
    return keyframes_[index];
}

std::shared_ptr<KeyFrameData> ROSDataProvider::getLatestKeyFrame() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (keyframes_.empty()) {
        return nullptr;
    }
    return keyframes_.back();
}

std::vector<std::shared_ptr<KeyFrameData>> ROSDataProvider::getAllKeyFrames() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_;
}

std::vector<std::shared_ptr<KeyFrameData>> ROSDataProvider::getNewKeyFrames()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::shared_ptr<KeyFrameData>> new_keyframes;
    
    for (size_t i = last_processed_index_; i < keyframes_.size(); ++i) {
        new_keyframes.push_back(keyframes_[i]);
    }

    for (const auto& update_kf : depth_fusion_updates_) {
        new_keyframes.push_back(update_kf);
    }
    depth_fusion_updates_.clear();
    
    last_processed_index_ = keyframes_.size();
    
    return new_keyframes;
}

bool ROSDataProvider::isInitialized() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_;
}

bool ROSDataProvider::isFinished() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
}

bool ROSDataProvider::hasNewKeyFrame() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_processed_index_ < keyframes_.size() || !depth_fusion_updates_.empty();
}

bool ROSDataProvider::getCameraIntrinsics(float& fx, float& fy, float& cx, float& cy) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!intrinsics_set_) {
        return false;
    }
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;
    return true;
}

bool ROSDataProvider::getImageSize(int& width, int& height) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!intrinsics_set_) {
        return false;
    }
    width = width_;
    height = height_;
    return true;
}

void ROSDataProvider::setCameraIntrinsics(float fx, float fy, float cx, float cy, 
                                          int width, int height)
{
    std::lock_guard<std::mutex> lock(mutex_);
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    width_ = width;
    height_ = height;
    intrinsics_set_ = true;
    
    ROS_INFO("[ROSDataProvider] Camera intrinsics set: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f, %dx%d",
             fx, fy, cx, cy, width, height);
}

} // namespace gaussian_mapping
