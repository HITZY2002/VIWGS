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

#pragma once  // 头文件保护，防止重复包含

// ========== LibTorch 相关头文件 ==========
#include <torch/torch.h>  // PyTorch C++ 库，用于张量操作和自动求导

// ========== C++ 标准库头文件 ==========
#include <iostream>      // 输入输出流（cout, cerr 等）
#include <algorithm>     // 算法库（sort, find 等）
#include <fstream>       // 文件流操作
#include <chrono>        // 时间相关（计时、时间戳）
#include <ctime>         // C 风格时间函数
#include <sstream>       // 字符串流
#include <thread>        // 多线程支持
#include <filesystem>    // 文件系统操作（路径、目录等）
#include <map>           // 关联容器（键值对）
#include <deque>         // 双端队列（任务调度）
#include <random>        // 随机数生成
#include <mutex>         // 互斥锁（线程同步）
#include <atomic>        // 原子变量（跨线程状态快照）

// ========== OpenCV 相关头文件 ==========
#include <opencv2/opencv.hpp>        // OpenCV 核心库（图像处理）
#include <opencv2/cudaimgproc.hpp>   // OpenCV CUDA 图像处理
#include <opencv2/cudastereo.hpp>    // OpenCV CUDA 立体视觉
#include <opencv2/cudawarping.hpp>   // OpenCV CUDA 图像变换

// ========== JSON 配置解析 ==========
#include <jsoncpp/json/json.h>  // JSON 配置文件解析库

// ========== slam 相关头文件 ==========
#include "gaussian_data_provider.h"        // 数据提供者接口（解耦SLAM系统）
#include <sophus/se3.hpp>    // Sophus 库：SE(3) 位姿表示
#include <Eigen/Core>
#include <Eigen/Geometry>
// ========== 3DGS 相关头文件 ==========
#include "operate_points.h"        // 点操作（CUDA 实现）
#include "stereo_vision.h"         // 立体视觉相关功能
#include "tensor_utils.h"          // 张量工具（OpenCV ↔ LibTorch 转换）
#include "gaussian_keyframe.h"     // 关键帧数据结构
#include "gaussian_scene.h"        // 场景管理（相机、关键帧、点云）
#include "gaussian_trainer.h"      // 训练器
#include "sliding_window_pose_optimizer.h"  // 滑动窗口主动位姿优化

// ========== 宏定义：目录检查和创建 ==========
// 如果目录不存在则创建，失败则抛出异常
#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());

// ========== 去畸变参数结构体 ==========
// 用于存储图像去畸变所需的参数
struct UndistortParams
{
    // 构造函数：初始化图像尺寸和畸变系数
    UndistortParams(
        const cv::Size& old_size,  // 原始图像尺寸
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))  // 畸变系数（默认无畸变）
        : old_size_(old_size)  // 初始化列表：设置原始尺寸
    {
        dist_coeff.copyTo(dist_coeff_);  // 深拷贝畸变系数
    }

    cv::Size old_size_;    // 原始图像尺寸（宽、高）
    cv::Mat dist_coeff_;   // 畸变系数矩阵（k1, k2, p1, p2）
};

// ========== 传感器类型枚举 ==========
// 定义系统支持的传感器类型
enum SystemSensorType
{
    INVALID = 0,   // 无效类型
    MONOCULAR = 1, // 单目相机
    STEREO = 2,    // 双目相机
    RGBD = 3       // RGB-D 相机（深度相机）
};

// ========== 可变参数结构体 ==========
// 存储训练过程中可以动态调整的参数
struct VariableParameters
{
    // ========== 学习率参数 ==========
    float position_lr_init;      // 位置学习率初始值
    float feature_lr;            // 特征（颜色）学习率
    float opacity_lr;            // 不透明度学习率
    float scaling_lr;            // 缩放学习率
    float rotation_lr;           // 旋转学习率
    
    // ========== 密度化参数 ==========
    float percent_dense;         // 密度化百分比阈值
    float densify_grad_th;       // 密度化梯度阈值
    int densify_interval;        // 密度化执行间隔（迭代次数）
    
    // ========== 损失函数参数 ==========
    float lambda_dssim;          // SSIM 损失权重（与 L1 损失的平衡）
    
    // ========== 不透明度参数 ==========
    int opacity_reset_interval;  // 不透明度重置间隔（迭代次数）
    
    // ========== 关键帧使用参数 ==========
    int new_kf_times_of_use;     // 新关键帧使用次数（训练中采样次数）
    int stable_num_iter_existence; ///< 稳定存在迭代数（用于回环闭合修正）
    
    // ========== 训练控制标志 ==========
    bool keep_training;          // 是否持续训练
    bool do_gaus_pyramid_training;  // 是否使用高斯金字塔训练（多分辨率）
    bool do_inactive_geo_densify;   // 是否对非活跃区域进行几何密度化
};

// ========== GaussianMapper 类 ==========
// 这是整个系统的核心控制器，负责：
// 1. 与 SLAM 系统交互，获取关键帧和地图点
// 2. 管理 3DGS 模型的训练和优化
// 3. 处理地图更新事件（局部 BA、回环闭合等）
// 4. 协调场景、模型、渲染器等组件
class GaussianMapper
{
public:
    struct PoseOptDebugInfo {
        bool enabled = false;
        bool has_optimizer = false;
        bool last_skipped_by_load = false;
        bool last_try_lock_failed = false;
        int last_status_code = 0;
        int queue_size = 0;
        int queue_limit = 0;
        int last_run_steps = 0;
        int last_run_chunks = 0;
        int last_run_elapsed_ms = 0;
        int total_optimize_calls = 0;
        int last_iteration = -1;
        float last_batch_ratio = 0.0f;
        float mapping_loop_ema_ms = 0.0f;
    };

    // ========== 构造函数 ==========
    // 初始化 GaussianMapper，建立与数据提供者的连接
    GaussianMapper(
        std::shared_ptr<gaussian_mapping::GaussianDataProvider> data_provider,  // 数据提供者接口
        std::filesystem::path gaussian_config_file_path,  // 3DGS 配置文件路径
        std::filesystem::path result_dir = "",      // 结果保存目录（可选）
        int seed = 0,                               // 随机数种子（用于可重复性）
        torch::DeviceType device_type = torch::kCUDA);  // 计算设备（GPU/CPU）

    // ========== 配置相关 ==========
    // 从配置文件读取参数
    void readConfigFromFile(std::filesystem::path cfg_path);

    // ========== 主运行接口 ==========
    void run();                    // 主训练循环（持续运行直到停止）
    void trainColmap();            // 使用 COLMAP 数据训练（离线模式）
    void trainForOneIteration();   // 执行一次训练迭代

    // ========== 数据同步（公开接口） ==========
    void syncNewKeyframes();       // 从数据提供者拉取新关键帧

    // ========== 控制接口 ==========
    bool isStopped();              // 检查是否已停止
    void signalStop(const bool going_to_stop = true);  // 发送停止信号

    // ========== 渲染接口 ==========
    // 从指定相机位姿渲染图像
    cv::Mat renderFromPose(
        const Sophus::SE3f &Tcw,   // 相机位姿（世界到相机）
        const int width,           // 渲染图像宽度
        const int height,          // 渲染图像高度
        const bool main_vision = false);  // 是否主视角（影响缩放）

    // ========== 迭代计数 ==========
    int getIteration();            // 获取当前迭代次数
    void increaseIteration(const int inc = 1);  // 增加迭代次数
    float getEmaLossForLog();      // 获取EMA损失
    PoseOptDebugInfo getPoseOptDebugInfo(); // 获取PoseOpt调度与执行状态（用于可视化）

    // ========== 参数获取接口（只读） ==========
    float positionLearningRateInit();      // 获取位置学习率初始值
    float featureLearningRate();           // 获取特征学习率
    float opacityLearningRate();           // 获取不透明度学习率
    float scalingLearningRate();           // 获取缩放学习率
    float rotationLearningRate();          // 获取旋转学习率
    float percentDense();                  // 获取密度化百分比
    float lambdaDssim();                   // 获取 SSIM 损失权重
    int opacityResetInterval();            // 获取不透明度重置间隔
    float densifyGradThreshold();          // 获取密度化梯度阈值
    int densifyInterval();                  // 获取密度化间隔
    int newKeyframeTimesOfUse();            // 获取新关键帧使用次数
    int stableNumIterExistence();          // 获取稳定存在迭代数
    bool isKeepingTraining();               // 是否持续训练
    bool isdoingGausPyramidTraining();     // 是否使用高斯金字塔训练
    bool isdoingInactiveGeoDensify();      // 是否进行非活跃区域密度化

    // ========== 参数设置接口（可写） ==========
    void setPositionLearningRateInit(const float lr);      // 设置位置学习率初始值
    void setFeatureLearningRate(const float lr);           // 设置特征学习率
    void setOpacityLearningRate(const float lr);           // 设置不透明度学习率
    void setScalingLearningRate(const float lr);          // 设置缩放学习率
    void setRotationLearningRate(const float lr);         // 设置旋转学习率
    void setPercentDense(const float percent_dense);      // 设置密度化百分比
    void setLambdaDssim(const float lambda_dssim);        // 设置 SSIM 损失权重
    void setOpacityResetInterval(const int interval);      // 设置不透明度重置间隔
    void setDensifyGradThreshold(const float th);         // 设置密度化梯度阈值
    void setDensifyInterval(const int interval);          // 设置密度化间隔
    void setNewKeyframeTimesOfUse(const int times);       // 设置新关键帧使用次数
    void setStableNumIterExistence(const int niter);      // 设置稳定存在迭代数
    void setKeepTraining(const bool keep);                 // 设置是否持续训练
    void requestTailOptimization(const bool keep_training = true);  // 手动请求进入尾部优化
    bool isTailOptimizationRequested();                    // 是否存在待处理的尾部优化请求
    void setDoGausPyramidTraining(const bool gaus_pyramid);  // 设置是否使用高斯金字塔训练
    void setDoInactiveGeoDensify(const bool inactive_geo_densify);  // 设置是否进行非活跃区域密度化

    // ========== 批量参数操作 ==========
    VariableParameters getVaribleParameters();  // 获取所有可变参数
    void setVaribleParameters(const VariableParameters &params);  // 设置所有可变参数

    // ========== 模型参数访问 ==========
    GaussianModelParams& getGaussianModelParams() { return this->model_params_; }  // 获取模型参数引用
    void setColmapDataPath(std::filesystem::path colmap_path) { this->model_params_.source_path_ = colmap_path; }  // 设置 COLMAP 数据路径
    void setSensorType(SystemSensorType sensor_type) { this->sensor_type_ = sensor_type; }  // 设置传感器类型

    // ========== 数据加载 ==========
    // 从 PLY 文件加载预训练的 Gaussian 模型
    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");

    // ========== COLMAP 导出 ==========
    void saveKeyframeToColmap(const std::shared_ptr<gaussian_mapping::KeyFrameData>& kf_data,
                              unsigned long kfid);

protected:
    // ========== 条件检查 ==========
    // 检查是否满足初始建图条件（需要足够的关键帧和地图点）
    bool hasMetInitialMappingConditions();
    // 检查是否满足增量建图条件（是否有新的地图更新）
    bool hasMetIncrementalMappingConditions();

    // ========== 数据同步 ==========
    // 与VINS滑动窗口同步（获取新的关键帧）
    void syncWithVinsSlidingWindow();

    // 初始训练前，对缓存关键帧做一次视觉 SfM 相对位姿精修
    bool refinePendingKeyframesBySfm(std::vector<std::shared_ptr<gaussian_mapping::KeyFrameData>>& keyframes);

    // ========== 地图操作处理 ==========
    // 处理来自 SLAM 系统的地图更新操作（局部 BA、回环闭合等）
    void combineMappingOperations();

    // ========== 关键帧处理 ==========
    // 处理新关键帧：创建 GaussianKeyframe 并添加到场景
    // 参数：关键帧元组（ID, 相机ID, 位姿, 图像, 是否回环, 辅助图像, 关键点像素, 关键点3D, 文件名）
    void handleNewKeyframe(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string> &kf);
    
    // ========== 关键帧采样和管理 ==========
    void generateKfidRandomShuffle();  // 生成关键帧 ID 的随机打乱序列
    std::shared_ptr<GaussianKeyframe> useOneRandomSlidingWindowKeyframe();  // 从滑动窗口随机选择一个关键帧
    std::shared_ptr<GaussianKeyframe> useOneRandomKeyframe();  // 随机选择一个关键帧
    void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframe> pkf, int times);  // 增加关键帧使用次数
    void boostOldFrameUsageEveryNIterations(int period_iters = 100);  // 每隔固定迭代随机提升老帧使用次数
    void boostHistoricalFrameUsageOnNewKeyframes(const std::vector<unsigned long>& new_kfids);  // 新关键帧到达后，筛选可见老点的历史帧并增加使用次数
    bool seesEnoughOldGaussiansByProjection(
        const std::shared_ptr<GaussianKeyframe>& cam,
        const torch::Tensor& old_xyz_cpu,
        int min_visible_old_pts) const;  // 纯投影判断该相机是否可见足够多老高斯
    void cullKeyframes();  // 剔除旧的关键帧（释放内存）

    // ========== PoseOpt 预算调度 ==========
    struct PoseOptTask {
        std::size_t task_id = 0;
        std::vector<unsigned long> frame_fids;
        std::vector<unsigned long> anchor_fids;
        int active_count = 0;
        int anchor_count = 0;
        int target_steps = 0;
        int steps_done = 0;
        bool completed = false;
        std::string source;
    };
    void enqueuePoseOptTask(int target_steps, const std::string& source);
    void runPoseOptWithBudget(bool force_run = false);
    void finishOnePoseOptTask(PoseOptTask& task);
    void boostUsageForOptimizedFids(const std::vector<unsigned long>& optimized_fids);
    bool shouldSkipPoseOptByLoad(bool force_run = false) const;
    float getPoseOptBatchRatioByLoad() const;

    // ========== 密度化相关 ==========
    // 从关键帧的非活跃区域提取几何信息进行密度化（单目模式）
    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<GaussianKeyframe> pkf);

    // ========== 已注释的接口（可能未来使用） ==========
    // bool needInterruptTraining();
    // void setInterruptTraining(const bool interrupt_training);

    // ========== 记录和保存 ==========
    // 记录关键帧的渲染结果（用于评估和可视化）
    void recordKeyframeRendered(
        torch::Tensor &rendered,      // 渲染图像张量
        torch::Tensor &ground_truth,  // 真实图像张量
        unsigned long kfid,           // 关键帧 ID
        std::filesystem::path result_img_dir,   // 渲染图像保存目录
        std::filesystem::path result_gt_dir,     // 真实图像保存目录
        std::filesystem::path result_loss_dir,   // 损失图像保存目录
        std::string name_suffix = "");  // 文件名后缀
    
    // 渲染并记录单个关键帧（计算 PSNR、SSIM 等指标）
    void renderAndRecordKeyframe(
        std::shared_ptr<GaussianKeyframe> pkf,  // 关键帧指针
        float &dssim,      // 输出：SSIM 距离
        float &psnr,       // 输出：PSNR
        float &psnr_gs,    // 输出：灰度 PSNR
        double &render_time,  // 输出：渲染时间
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    
    // 渲染并记录所有关键帧
    void renderAndRecordAllKeyframes(
        std::string name_suffix = "");

    // ========== 数据保存 ==========
    void savePly(std::filesystem::path result_dir);  // 保存 Gaussian 模型为 PLY 文件
    void keyframesToJson(std::filesystem::path result_dir);  // 将关键帧信息保存为 JSON
    void saveModelParams(std::filesystem::path result_dir);  // 保存模型参数
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");  // 记录关键帧使用次数

public:
    // ========== 公共成员变量 ==========
    
    // 配置路径
    std::filesystem::path config_file_path_;  // 配置文件路径

    // ========== 核心模型组件 ==========
    std::shared_ptr<GaussianModel> gaussians_;  // 3D Gaussian 模型（存储所有 Gaussian 点）
    std::shared_ptr<GaussianScene> scene_;      // 场景管理（相机、关键帧、点云）

    // ========== 数据提供者 ==========
    std::shared_ptr<gaussian_mapping::GaussianDataProvider> data_provider_;  // 数据提供者接口（用于获取关键帧和地图点）

    // ========== 设备设置 ==========
    torch::DeviceType device_type_;  // 计算设备类型（CUDA/CPU）
    bool opencv_cuda_available_ = false;  // OpenCV CUDA 是否可用（与 torch CUDA 独立）

    // ========== 高斯金字塔训练设置 ==========
    int num_gaus_pyramid_sub_levels_ = 0;  // 高斯金字塔子层级数量
    std::vector<int> kf_gaus_pyramid_times_of_use_;  // 每个金字塔层级的关键帧使用次数
    std::vector<float> kf_gaus_pyramid_factors_;    // 每个金字塔层级的缩放因子

    // ========== 可视化设置 ==========
    bool viewer_camera_id_set_ = false;  // 是否设置了查看器相机 ID
    std::uint32_t viewer_camera_id_ = 0;  // 查看器使用的相机 ID
    float rendered_image_viewer_scale_ = 1.0f;      // 查看器图像缩放比例
    float rendered_image_viewer_scale_main_ = 1.0f;  // 主视角图像缩放比例

    // ========== 深度范围 ==========
    float z_near_ = 0.01f;  // 近裁剪平面距离
    float z_far_ = 100.0f;  // 远裁剪平面距离

    // ========== 数据缓存 ==========
    bool kfid_shuffled_ = false;  // 关键帧 ID 是否已打乱
    std::vector<std::shared_ptr<gaussian_mapping::KeyFrameData>> pending_keyframes_;
    // 去畸变掩码（按相机 ID 索引）
    std::map<camera_id_t, torch::Tensor> undistort_mask_;           // 标准去畸变掩码
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;  // 查看器主视角去畸变掩码
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;   // 查看器子视角去畸变掩码

protected:
    // ========== 保护成员变量 ==========
    
    // ========== 参数配置 ==========
    GaussianModelParams model_params_;      // 模型参数（SH 阶数、分辨率等）
    GaussianOptimizationParams opt_params_; // 优化参数（学习率、迭代次数等）
    GaussianPipelineParams pipe_params_;    // 流水线参数（SH 转换、协方差计算等）

    // ========== 关键帧管理数据 ==========
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> viewpoint_sliding_window_;  // 滑动窗口中的关键帧（用于训练）
    std::vector<std::size_t> kfid_shuffle_;  // 关键帧 ID 的随机打乱序列
    std::size_t kfid_shuffle_idx_ = 0;       // 当前打乱序列索引
    std::map<std::size_t, int> kfs_used_times_;  // 每个关键帧的使用次数（用于采样）
    point3D_id_t next_point3d_id_ = 0;       // 点云缓存的自增 ID

    // ========== 状态标志 ==========
    bool initial_mapped_;          // 是否已完成初始建图
    bool interrupt_training_;       // 是否中断训练（未使用）
    bool stopped_;                  // 是否已停止
    int iteration_;                 // 当前迭代次数
    float ema_loss_for_log_;        // 指数移动平均损失（用于日志）
    bool SLAM_ended_;               // SLAM 是否已结束
    bool loop_closure_iteration_;   // 当前迭代是否是回环闭合迭代
    bool keep_training_ = false;    // 是否持续训练
    int default_sh_ = 0;            // 默认球谐函数阶数

    // ========== 传感器设置 ==========
    SystemSensorType sensor_type_;  // 传感器类型（单目/双目/RGB-D）
    Sophus::SE3d T_imu_cam_;  // IMU到相机的外参（body到camera）

    // ========== 单目相关参数 ==========
    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;  // 单目非活跃区域密度化最大像素距离

    // ========== 双目相关参数 ==========
    float stereo_baseline_length_ = 0.0f;  // 双目基线长度
    int stereo_min_disparity_ = 0;         // 最小视差
    int stereo_num_disparity_ = 128;       // 视差范围
    cv::Mat stereo_Q_;                     // 立体视觉重投影矩阵 Q
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;  // CUDA 立体匹配算法（Semi-Global Matching）

    // ========== RGB-D 相关参数 ==========
    float RGBD_min_depth_ = 0.0f;   // RGB-D 最小深度
    float RGBD_max_depth_ = 100.0f;  // RGB-D 最大深度

    // ========== 深度缓存（用于密度化） ==========
    bool inactive_geo_densify_ = true;  // 是否启用非活跃区域几何密度化
    int depth_cached_ = 0;              // 当前缓存的深度数量
    int max_depth_cached_ = 1;          // 最大缓存深度数量
    torch::Tensor depth_cache_points_;  // 缓存的深度点（3D 位置）
    torch::Tensor depth_cache_colors_;  // 缓存的深度点颜色

    // ========== 初始建图参数 ==========
    unsigned long min_num_initial_map_kfs_;  // 初始建图所需的最小关键帧数量

    // ========== 渲染相关 ==========
    torch::Tensor background_;  // 背景颜色张量（RGB）
    float large_rot_th_;        // 大旋转阈值（用于检测大运动）
    float large_trans_th_;      // 大平移阈值（用于检测大运动）
    torch::Tensor override_color_;  // 覆盖颜色（用于特殊渲染效果）

    // ========== 关键帧使用次数设置 ==========
    int new_keyframe_times_of_use_;           // 新关键帧使用次数
    int old_keyframe_times_of_use_ = 1;       // 老关键帧补充使用次数（滑窗关键帧用尽时）
    int local_BA_increased_times_of_use_;     // 局部 BA 后增加的使用次数
    int loop_closure_increased_times_of_use_; // 回环闭合后增加的使用次数

    // ========== 关键帧管理 ==========
    bool cull_keyframes_;              // 是否启用关键帧剔除
    int stable_num_iter_existence_;    // 稳定存在迭代数（用于判断关键帧是否稳定）

    // ========== 增量点云 ==========
    bool enable_incremental_pcd_ = true;  // 是否增量插入新点云

    // ========== 训练设置 ==========
    bool do_gaus_pyramid_training_;    // 是否使用高斯金字塔训练

    // ========== 结果保存设置 ==========
    std::filesystem::path result_dir_;  // 结果保存目录
    int keyframe_record_interval_;       // 关键帧记录间隔（迭代次数）
    int all_keyframes_record_interval_;  // 所有关键帧记录间隔
    bool record_rendered_image_;        // 是否记录渲染图像
    bool record_ground_truth_image_;     // 是否记录真实图像
    bool record_loss_image_;            // 是否记录损失图像

    // ========== COLMAP 导出 ==========
    bool save_colmap_ = false;
    std::filesystem::path colmap_dir_;
    std::ofstream colmap_cameras_;
    std::ofstream colmap_images_;
    std::ofstream colmap_points_;
    bool colmap_initialized_ = false;
    int colmap_camera_id_ = 1;
    int colmap_next_image_id_ = 1;
    long colmap_next_point_id_ = 1;

    // ========== 畸变参数 ==========
    std::vector<double> dist_coeffs_;

    // ========== 训练报告 ==========
    int training_report_interval_;  // 训练报告间隔（迭代次数）
    bool record_loop_ply_;          // 是否在回环闭合时记录 PLY 文件

    // ========== 点云剪枝 ==========
    int prune_big_point_after_iter_;  // 在此迭代后剪枝大点
    float densify_min_opacity_ = 20;   // 密度化最小不透明度阈值
    bool enable_safety_prune_ = true;  // 全程安全剪枝（防止后期远处大点爆发）
    int safety_prune_interval_ = 100;   // 安全剪枝执行间隔（迭代）
    float safety_prune_max_world_scale_ratio_ = 0.08f; // 单高斯最大世界尺度阈值（相对场景尺度）
    float safety_prune_max_center_dist_ratio_ = 6.0f;  // 点到场景中心最大距离阈值（相对场景尺度）
    float isotropic_scaling_reg_weight_ = 0.0f;  // 高斯各向同性正则权重（约束为近球形）
    float isotropic_scaling_reg_min_ratio_ = 0.10f;  // 各向同性正则最小保留比例
    float isotropic_scaling_reg_progress_power_ = 1.0f;  // 随训练进度衰减指数
    float isotropic_scaling_reg_loss_power_ = 1.0f;  // 随loss下降衰减指数
    float isotropic_scaling_reg_ref_ema_loss_ = -1.0f;  // 各向同性正则loss归一化参考值
    int isotropic_scaling_reg_new_point_iters_ = 500;  // 各向同性正则仅作用于新点的迭代窗口
    int multiview_old_frame_samples_ = 0;  // 每次迭代额外采样的历史老帧数量
    float multiview_old_frame_loss_weight_ = 0.0f;  // 历史老帧损失权重
    int multiview_old_gaussian_min_age_iters_ = 500;  // 认定老高斯的最小年龄（迭代）

    // ========== 滑动窗口主动位姿优化 ==========
    bool enable_pose_optimization_ = false;               ///< 是否启用位姿优化
    std::unique_ptr<SlidingWindowPoseOptimizer> pose_optimizer_;  ///< 滑动窗口位姿优化器
    SlidingWindowPoseOptimizer::Config pose_opt_cfg_;     ///< 位姿优化配置
    bool pose_opt_trigger_on_new_keyframe_ = true;        ///< 新关键帧到达时是否触发位姿优化
    int pose_opt_steps_on_new_keyframe_ = 1;              ///< 新关键帧触发时的小步数
    int pose_opt_max_chunks_per_loop_ = 1;                ///< 每次主循环最多处理的 PoseOpt chunk 数
    int pose_opt_max_steps_per_loop_ = 2;                 ///< 每次主循环最多执行的 PoseOpt step 数
    int pose_opt_max_millis_per_loop_ = 8;                ///< 每次主循环用于 PoseOpt 的最大毫秒预算
    std::size_t pose_opt_task_queue_limit_ = 2;           ///< PoseOpt 任务队列上限
    std::size_t pose_opt_next_task_id_ = 1;               ///< PoseOpt 任务自增 ID
    std::deque<PoseOptTask> pose_opt_task_queue_;         ///< PoseOpt 任务队列
    bool pose_opt_skip_when_new_keyframe_pending_ = true; ///< 新关键帧待处理时跳过 PoseOpt
    float pose_opt_busy_loop_ms_soft_ = 60.0f;            ///< 系统繁忙软阈值（毫秒）
    float pose_opt_busy_loop_ms_hard_ = 120.0f;           ///< 系统繁忙硬阈值（毫秒）
    float pose_opt_busy_batch_ratio_scale_ = 0.5f;        ///< 系统繁忙时 batch_ratio 缩放
    int pose_opt_extra_train_budget_per_round_ = 3;       ///< 每轮 PoseOpt 触发的固定训练补偿总预算
    float mapping_loop_ema_ms_ = 0.0f;                    ///< mapping loop 耗时 EMA（毫秒）
    std::atomic<int> pose_opt_dbg_queue_size_{0};
    std::atomic<int> pose_opt_dbg_last_status_code_{0};
    std::atomic<int> pose_opt_dbg_last_run_steps_{0};
    std::atomic<int> pose_opt_dbg_last_run_chunks_{0};
    std::atomic<int> pose_opt_dbg_last_run_elapsed_ms_{0};
    std::atomic<int> pose_opt_dbg_total_optimize_calls_{0};
    std::atomic<int> pose_opt_dbg_last_iteration_{-1};
    std::atomic<int> pose_opt_dbg_last_skipped_by_load_{0};
    std::atomic<int> pose_opt_dbg_last_try_lock_failed_{0};
    std::atomic<float> pose_opt_dbg_last_batch_ratio_{0.0f};

    // ========== 初始 SfM 精修 ==========
    bool do_initial_sfm_refine_ = true;                   ///< 初始建图前是否做一次 SfM 位姿精修
    int initial_sfm_min_inliers_ = 40;                    ///< 相邻帧 SfM 最小内点数

    bool tail_optimization_requested_ = false;            ///< 是否手动请求进入尾部优化

    // ========== 工具 ==========
    std::random_device rd_;  // 随机数设备（用于生成随机数种子）

    // ========== 互斥锁（线程安全） ==========
    std::mutex mutex_status_;   // 保护状态变量的互斥锁
    std::mutex mutex_settings_; // 保护设置参数的互斥锁
    std::mutex mutex_render_;   ///< 保护渲染相关数据的互斥锁（模型对外部应该是只读的）
};
