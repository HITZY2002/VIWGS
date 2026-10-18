/**
 * ROS数据提供者实现
 * 通过ROS话题订阅VINS数据，实现GaussianDataProvider接口
 */

#ifndef ROS_DATA_PROVIDER_H
#define ROS_DATA_PROVIDER_H

#include "gaussian_data_provider.h"
#include "depth_anything_v2_estimator.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>

#include <mutex>
#include <deque>
#include <map>
#include <fstream>
#include <memory>
#include <thread>
#include <condition_variable>
#include <Eigen/Core>

namespace gaussian_mapping {

/**
 * ROS数据提供者类
 * 订阅VINS话题，缓存关键帧数据，提供给GaussianMapper使用
 */
class ROSDataProvider : public GaussianDataProvider
{
public:
    ROSDataProvider(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
    virtual ~ROSDataProvider();
    
    // ========== GaussianDataProvider 接口实现 ==========
    
    size_t getKeyFrameCount() const override;
    std::shared_ptr<KeyFrameData> getKeyFrame(size_t index) const override;
    std::shared_ptr<KeyFrameData> getLatestKeyFrame() const override;
    std::vector<std::shared_ptr<KeyFrameData>> getAllKeyFrames() const override;
    std::vector<std::shared_ptr<KeyFrameData>> getNewKeyFrames() override;
    
    bool isInitialized() const override;
    bool isFinished() const override;
    bool hasNewKeyFrame() const override;
    
    bool getCameraIntrinsics(float& fx, float& fy, float& cx, float& cy) const override;
    bool getImageSize(int& width, int& height) const override;
    
    // ========== 设置相机内参 ==========
    void setCameraIntrinsics(float fx, float fy, float cx, float cy, int width, int height);
    
private:
    // ========== ROS 回调函数 ==========
    void keyframePoseCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void keyframeImageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void keyframePointCloudCallback(const sensor_msgs::PointCloud::ConstPtr& msg);
    
    // ========== 数据同步与组装 ==========
    void assembleKeyFrame(double timestamp);
    bool buildUndistortParams(cv::Mat& K, cv::Mat& dist) const;
    void augmentSparsePointsWithDepth(std::shared_ptr<KeyFrameData>& kf);
    bool generateDepthFusionPoints(const std::shared_ptr<KeyFrameData>& kf,
                                   std::vector<Eigen::Vector3f>& out_points,
                                   std::vector<Eigen::Vector3f>& out_colors);
    void enqueueDepthFusionTask(const std::shared_ptr<KeyFrameData>& kf);
    void depthFusionWorkerLoop();
    
    // ========== ROS 相关 ==========
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    ros::Subscriber sub_keyframe_pose_;
    ros::Subscriber sub_keyframe_image_;
    ros::Subscriber sub_keyframe_pointcloud_;

    // 话题名（可通过 ROS param 配置）
    std::string topic_keyframe_pose_;
    std::string topic_keyframe_image_;
    std::string topic_keyframe_point_;
    
    // ========== 数据缓存 ==========
    mutable std::mutex mutex_;
    
    // 临时缓存（等待同步的数据）
    std::map<double, nav_msgs::Odometry> pose_buffer_;
    std::map<double, sensor_msgs::Image> image_buffer_;
    std::map<double, sensor_msgs::PointCloud> pointcloud_buffer_;
    
    // 关键帧数据（已同步完成）
    std::vector<std::shared_ptr<KeyFrameData>> keyframes_;
    size_t last_processed_index_;  // 上次getNewKeyFrames()处理到的索引
    
    // ========== 相机内参 ==========
    float fx_, fy_, cx_, cy_;
    int width_, height_;
    bool intrinsics_set_;
    
    // ========== 状态标志 ==========
    bool initialized_;
    bool finished_;
    
    // ========== 参数配置 ==========
    double sync_time_threshold_;  // 时间同步阈值（秒）
    size_t max_buffer_size_;      // 最大缓存大小

    // 初始化阈值
    int min_keyframes_to_start_;

    // 点云过滤参数
    float min_point_range_;       // 最小距离
    float max_point_range_;       // 最大距离

    // 点云坐标系配置
    bool pointcloud_in_camera_frame_;  // true: 点云在相机系，需要转换到世界系

    // 位姿坐标系配置
    bool pose_is_Tcw_;                 // true: pose消息即T_cw；false: pose消息为T_wc

    // 关键帧去畸变配置
    bool undistort_keyframes_;
    std::vector<double> dist_coeffs_;

    // 对齐调试（可开关）
    bool alignment_debug_;
    int alignment_debug_interval_;
    int alignment_debug_count_;

    // 尺度稳定性检查
    bool scale_stable_;
    int scale_check_window_;
    double scale_std_ratio_th_;
    double scale_jump_ratio_th_;
    double scale_min_translation_;
    bool scale_debug_;
    int scale_debug_interval_;
    int scale_debug_count_;
    Eigen::Vector3d last_cam_pos_;
    bool last_cam_pos_valid_;
    std::deque<double> recent_translations_;

    // 丢弃前 N 帧
    int drop_first_keyframes_;
    int dropped_keyframes_;

    // 关键帧内存上限
    int max_keyframes_in_memory_;

    // ========== 关键帧质量筛选参数 ==========
    bool enable_kf_laplacian_filter_;   // 是否启用模糊检测
    bool enable_kf_pose_filter_;        // 是否启用最小平移/旋转筛选
    // 1. 拉普拉斯模糊检测阈值（低于此值丢弃）
    double laplacian_threshold_;
    // 2. 与上一帧之间最小平移距离（m），低于则丢弃
    double kf_min_translation_;
    // 3. 与上一帧之间最小旋转角（度），低于则丢弃
    double kf_min_rotation_deg_;
    // 上一个被 3DGS 接收的关键帧位姿（世界坐标系下相机位置 + 旋转）
    bool last_accepted_pose_valid_;
    Eigen::Vector3d last_accepted_cam_pos_;
    Eigen::Matrix3d last_accepted_cam_rot_;

    // ========== 尺度稳定性检查开关 ==========
    bool enable_scale_check_;  // false 时跳过尺度稳定性要求，帧数足够即可初始化

    // ========== DepthAnythingV2 补点 ==========
    bool depth_fusion_enable_;
    std::string depth_anything_model_path_;
    std::string depth_anything_output_type_;
    int depth_anything_input_size_;
    std::string depth_anything_device_;
    int depth_fusion_target_points_per_kf_;
    int depth_fusion_max_new_points_;
    int depth_fusion_min_sparse_matches_;
    int depth_fusion_scale_trust_min_sparse_matches_;
    int depth_fusion_scale_ema_fallback_min_sparse_matches_;
    float depth_fusion_min_depth_;
    float depth_fusion_max_depth_;
    float depth_fusion_edge_grad_threshold_;
    bool depth_fusion_scale_ema_enable_;
    float depth_fusion_scale_ema_alpha_;
    mutable std::mutex depth_fusion_scale_ema_mutex_;
    bool depth_fusion_scale_ema_initialized_;
    float depth_fusion_scale_ema_value_;
    bool depth_fusion_debug_log_;
    bool depth_fusion_async_enable_;
    int depth_fusion_every_n_keyframes_;
    int depth_fusion_async_max_queue_;
    int accepted_keyframe_count_;
    std::unique_ptr<DepthAnythingV2Estimator> depth_estimator_;

    // Depth 补点异步任务队列（主线程不阻塞训练）
    std::mutex depth_fusion_mutex_;
    std::condition_variable depth_fusion_cv_;
    std::deque<std::shared_ptr<KeyFrameData>> depth_fusion_tasks_;
    std::thread depth_fusion_worker_;
    bool depth_fusion_worker_stop_;

    // 异步补点结果增量队列（在 getNewKeyFrames 中与新关键帧一起返回）
    std::vector<std::shared_ptr<KeyFrameData>> depth_fusion_updates_;

};

} // namespace gaussian_mapping

#endif // ROS_DATA_PROVIDER_H
