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

// 包含 GaussianMapper 类的头文件
#include "include/gaussian_mapper.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <unordered_set>
#define EIGEN_TO_TORCH(m, device) tensor_utils::EigenMatrix2TorchTensor(m, device)

namespace {
inline bool detectOpenCVCudaAvailable()
{
    try {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }
    catch (const cv::Exception&) {
        return false;
    }
}

inline Eigen::Vector3f transformWorldPointByPoseUpdate(
    const Eigen::Vector3f& pw_old,
    const Sophus::SE3f& Tcw_old,
    const Sophus::SE3f& Tcw_new)
{
    // 保持同一相机观测：p_c = Tcw_old * p_w_old = Tcw_new * p_w_new
    // => p_w_new = Tcw_new^{-1} * (Tcw_old * p_w_old)
    Eigen::Vector3f pc = Tcw_old.rotationMatrix() * pw_old + Tcw_old.translation();
    Eigen::Vector3f pw_new = Tcw_new.rotationMatrix().transpose() * (pc - Tcw_new.translation());
    return pw_new;
}
}
// ========== 构造函数：初始化 GaussianMapper ==========
// 参数：
//   data_provider: 数据提供者接口智能指针
//   gaussian_config_file_path: 3DGS 配置文件路径
//   result_dir: 结果保存目录
//   seed: 随机数种子（用于可重复性）
//   device_type: 计算设备类型（CUDA/CPU）
GaussianMapper::GaussianMapper(
    std::shared_ptr<gaussian_mapping::GaussianDataProvider> data_provider,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : data_provider_(data_provider),    // 初始化数据提供者接口
      initial_mapped_(false),           // 初始建图标志：未完成 
      interrupt_training_(false),        // 训练中断标志：未中断
      stopped_(false),                  // 停止标志：未停止
      iteration_(0),                     // 迭代次数：初始为 0
      ema_loss_for_log_(0.0f),          // 指数移动平均损失：初始为 0
      SLAM_ended_(false),                // SLAM 结束标志：未结束
      loop_closure_iteration_(false),   // 回环闭合迭代标志：否
      min_num_initial_map_kfs_(10UL),   // 初始建图所需最小关键帧数：10
      large_rot_th_(1e-1f),             // 大旋转阈值：0.1 弧度
      large_trans_th_(1e-2f),           // 大平移阈值：0.01 米
      training_report_interval_(0)       // 训练报告间隔：0（不报告）
{
    // ========== 设置随机数种子 ==========
    std::srand(seed);              // 设置 C++ 标准库随机数种子
    torch::manual_seed(seed);      // 设置 PyTorch 随机数种子（确保可重复性）

    // ========== 检测和设置计算设备 ==========
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        // CUDA 可用，使用 GPU
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;              // 设置设备类型为 CUDA
        model_params_.data_device_ = "cuda";       // 设置模型参数中的设备字符串
    }
    else {
        // CUDA 不可用，使用 CPU
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;                // 设置设备类型为 CPU
        model_params_.data_device_ = "cpu";        // 设置模型参数中的设备字符串
    }

    opencv_cuda_available_ = detectOpenCVCudaAvailable();
    if (device_type_ == torch::kCUDA && !opencv_cuda_available_) {
        std::cout << "[Gaussian Mapper] OpenCV CUDA unavailable at runtime, fallback OpenCV ops to CPU." << std::endl;
    }

    // ========== 设置结果目录和配置文件 ==========
    result_dir_ = result_dir;                      // 保存结果目录路径
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)  // 如果目录不存在则创建
    config_file_path_ = gaussian_config_file_path; // 保存配置文件路径
    readConfigFromFile(gaussian_config_file_path);  // 从配置文件读取参数

    if (save_colmap_ && colmap_dir_.empty()) {
        colmap_dir_ = result_dir_ / "colmap";
    }

    // ========== 设置背景颜色 ==========
    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};  // 白色背景 (RGB: 1,1,1)
    else
        bg_color = {0.0f, 0.0f, 0.0f};  // 黑色背景 (RGB: 0,0,0)
    // 将背景颜色转换为 Torch 张量，放在指定设备上
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    // ========== 初始化覆盖颜色（用于特殊渲染效果，初始为空） ==========
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // ========== 初始化场景和模型 ==========
    // 创建 Gaussian 模型（存储所有 3D Gaussian 点）
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    // 创建场景管理（相机、关键帧、点云）
    scene_ = std::make_shared<GaussianScene>(model_params_);

    // ========== 检查数据提供者 ==========
    if (!data_provider) {
        std::cerr << "[Gaussian Mapper] DataProvider is null!" << std::endl;
        // NO DATA PROVIDER
        return;
    }

    // ========== 设置传感器类型（默认为单目） ==========
    // 注：如果需要支持双目，可以在数据提供者中添加getSensorType()接口
    this->sensor_type_ = MONOCULAR;
    
    // IMU到相机的外参默认为单位矩阵（假设body=camera）
    // 如果需要外参，可在数据提供者中添加getExtrinsics()接口
    // T_imu_cam_ = Sophus::SE3d();  // 单位变换
    
    // 从数据提供者获取内参
    float fx, fy, cx, cy;
    int width, height;
    if (!data_provider->getCameraIntrinsics(fx, fy, cx, cy)) {
        std::cerr << "[Gaussian Mapper] Warning: Camera intrinsics not set in data provider!" << std::endl;
        // 使用默认值
        fx = fy = 460.0f;
        cx = 320.0f;
        cy = 240.0f;
    }
    
    if (!data_provider->getImageSize(width, height)) {
        std::cerr << "[Gaussian Mapper] Warning: Image size not set in data provider!" << std::endl;
        // 使用默认值
        width = 640;
        height = 480;
    }
    
    // 构建 Camera 对象
    Camera camera;
    camera.camera_id_ = 0;
    camera.setModelId(Camera::CameraModelType::PINHOLE);
    
    // 构建内参矩阵 K (用于去畸变初始化)
    cv::Mat K = (cv::Mat_<float>(3, 3) << 
        fx, 0.f, cx, 
        0.f, fy, cy, 
        0.f, 0.f, 1.f);
    cv::Mat K_new = K.clone();

    // 保存参数
    camera.width_ = width;
    camera.height_ = height;
    camera.params_[0] = fx; camera.params_[1] = fy;
    camera.params_[2] = cx; camera.params_[3] = cy;

            // ========== 设置高斯金字塔参数（用于多分辨率训练） ==========
            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;  // 金字塔子层级数量
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);     // 调整宽度数组大小
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);    // 调整高度数组大小
            // 为每个金字塔层级计算图像尺寸（按缩放因子）
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];   // 层级 l 的宽度
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l]; // 层级 l 的高度
            }

            // ========== 设置新的相机内参（考虑图像缩放） ==========
            // ========== 构建新的相机内参矩阵 K_new ==========
            // ========== 设置畸变系数（单目和 RGB-D 需要去畸变） ==========
            // ========== 初始化去畸变映射和掩码 ==========
            // 参数：原始 K、原始图像尺寸、新 K、是否进行高斯金字塔训练
            //camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);
            // 这里为单目去畸变初始化 undistort map（使用配置畸变系数）
            if (dist_coeffs_.size() >= 5) {
                camera.dist_coeff_ = (cv::Mat_<float>(1, 5) <<
                    static_cast<float>(dist_coeffs_[0]),
                    static_cast<float>(dist_coeffs_[1]),
                    static_cast<float>(dist_coeffs_[2]),
                    static_cast<float>(dist_coeffs_[3]),
                    static_cast<float>(dist_coeffs_[4]));
            } else if (dist_coeffs_.size() >= 4) {
                camera.dist_coeff_ = (cv::Mat_<float>(1, 4) <<
                    static_cast<float>(dist_coeffs_[0]),
                    static_cast<float>(dist_coeffs_[1]),
                    static_cast<float>(dist_coeffs_[2]),
                    static_cast<float>(dist_coeffs_[3]));
            } else {
                camera.dist_coeff_ = cv::Mat::zeros(4, 1, CV_32F);
            }
            const bool use_gaus_pyramid = do_gaus_pyramid_training_ && num_gaus_pyramid_sub_levels_ > 0;
            camera.initUndistortRectifyMapAndMask(K, cv::Size(width, height), K_new, use_gaus_pyramid);
            // ========== 将去畸变掩码转换为 Torch 张量（标准尺寸） ==========
            undistort_mask_[0] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);  // OpenCV Mat → Torch Tensor

            // ========== 创建查看器子视角的去畸变掩码（缩放后的尺寸） ==========
            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;      // 计算查看器图像高度
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;        // 计算查看器图像宽度
            // 将掩码缩放到查看器尺寸
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            // 转换为 Torch 张量并存储
            viewer_sub_undistort_mask_[0] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            // ========== 创建查看器主视角的去畸变掩码（主视角缩放后的尺寸） ==========
            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;  // 计算主视角图像高度
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;    // 计算主视角图像宽度
            // 将掩码缩放到主视角查看器尺寸
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            // 转换为 Torch 张量并存储
            viewer_main_undistort_mask_[0] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        // ========== 双目模式特殊处理 =========
        // ========== 处理鱼眼相机模型 ==========
        // ========== 无效的相机模型 ==========
        // ========== 设置查看器使用的相机 ID（使用第一个相机） ==========
        viewer_camera_id_ = 0;
        viewer_camera_id_set_ = true;
        // ========== 将相机添加到场景 ==========
        this->scene_->addCamera(camera);
}

// ========== 从配置文件读取参数 ==========
// 参数：cfg_path - 配置文件路径（YAML 格式）
void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    // ========== 打开配置文件 ==========
    std::string path_str = cfg_path.string();
    
    // 检查路径是否为空
    if (path_str.empty()) {
        std::error_code ec;
        std::cerr << "[Gaussian Mapper] ERROR: Config path is EMPTY!" << std::endl;
        exit(-1);
    }

    // 检查文件在操作系统层面是否存在
    if (!std::filesystem::exists(cfg_path)) {
        std::cerr << "[Gaussian Mapper] ERROR: File does not exist at: " << std::filesystem::absolute(cfg_path) << std::endl;
        std::cerr << "[Gaussian Mapper] Current Working Directory: " << std::filesystem::current_path() << std::endl;
        exit(-1);
    }
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       // 如果文件打开失败，输出错误信息并退出
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    // ========== 输出读取信息 ==========
    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    // ========== 加锁保护设置参数（线程安全） ==========
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // ========== 模型参数 ==========
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();  // 球谐函数阶数（通常为 3）
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();  // 分辨率（-1 表示使用原始分辨率）
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;  // 是否使用白色背景
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;  // 是否为评估模式

    // ========== 流水线参数（相机相关） ==========
    z_near_ =
        settings_file["Camera.z_near"].operator float();  // 近裁剪平面距离（默认 0.01）
    z_far_ =
        settings_file["Camera.z_far"].operator float();  // 远裁剪平面距离（默认 100.0）

    // ========== 畸变系数（用于去畸变） ==========
    dist_coeffs_.clear();
    {
        cv::FileNode dist_node = settings_file["dist_coeffs"];
        if (!dist_node.isNone()) {
            for (auto it = dist_node.begin(); it != dist_node.end(); ++it) {
                dist_coeffs_.push_back(static_cast<double>(*it));
            }
        }
    }

    // ========== 单目模式参数 ==========
    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();  // 单目非活跃区域密度化最大像素距离
    // ========== 双目模式参数 ==========
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();  // 最小视差
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();  // 视差范围（最大视差 - 最小视差）
    // ========== RGB-D 模式参数 ==========
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();  // RGB-D 最小深度（米）
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();  // RGB-D 最大深度（米）

    // ========== 映射器参数 ==========
    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;  // 是否启用非活跃区域几何密度化
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();  // 最大深度缓存数量（累积多少个关键帧的深度后添加到模型）
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());  // 初始建图所需最小关键帧数
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();  // 新关键帧使用次数（训练中采样次数）
    {
        cv::FileNode old_use_node = settings_file["Mapper.old_keyframe_times_of_use"];
        if (!old_use_node.isNone()) {
            old_keyframe_times_of_use_ = old_use_node.operator int();
        }
        old_keyframe_times_of_use_ = std::max(1, old_keyframe_times_of_use_);
    }
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();  // 局部 BA 后增加的使用次数
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();  // 回环闭合后增加的使用次数
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;  // 是否启用关键帧剔除
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();  // 大旋转阈值（用于检测回环闭合时的大位姿变化）
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();  // 大平移阈值（用于检测回环闭合时的大位姿变化）
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();  // 稳定存在迭代数（用于判断点是否稳定，回环闭合时使用）
    {
        cv::FileNode inc_node = settings_file["Mapper.enable_incremental_pcd"];
        if (!inc_node.isNone()) {
            enable_incremental_pcd_ = (inc_node.operator int()) != 0;
        }
    }
    {
        cv::FileNode sfm_node = settings_file["Mapper.do_initial_sfm_refine"];
        if (!sfm_node.isNone()) {
            do_initial_sfm_refine_ = (sfm_node.operator int()) != 0;
        }
    }
    {
        cv::FileNode sfm_inlier_node = settings_file["Mapper.initial_sfm_min_inliers"];
        if (!sfm_inlier_node.isNone()) {
            initial_sfm_min_inliers_ = sfm_inlier_node.operator int();
        }
    }

    // ========== 流水线参数 ==========
    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;  // 是否转换球谐系数
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;  // 是否计算 3D 协方差

    // ========== 高斯金字塔训练参数 ==========
    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;  // 是否使用高斯金字塔训练（多分辨率）
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();  // 金字塔子层级数量
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();  // 每个子层级的使用次数
    // 调整数组大小
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);  // 每个层级的使用次数数组
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);       // 每个层级的缩放因子数组
    // 为每个金字塔层级设置参数
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;  // 设置使用次数
        // 计算缩放因子：层级 l 的缩放因子 = 0.5^(总层级数 - l)
        // 例如：3 个层级，l=0 时 factor=0.5^3=0.125, l=1 时 factor=0.5^2=0.25, l=2 时 factor=0.5^1=0.5
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    // ========== 记录参数 ==========
    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();  // 关键帧记录间隔（迭代次数，0 表示不记录）
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();  // 所有关键帧记录间隔（迭代次数）
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;  // 是否记录渲染图像
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;  // 是否记录真实图像
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;  // 是否记录损失图像（渲染与真实的差异）
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();  // 训练报告间隔（迭代次数，0 表示不报告）
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;  // 是否在回环闭合时记录 PLY 文件
    {
        cv::FileNode colmap_node = settings_file["Record.save_colmap"];
        if (!colmap_node.isNone()) {
            save_colmap_ = (colmap_node.operator int()) != 0;
        }
    }
    {
        cv::FileNode colmap_dir_node = settings_file["Record.colmap_dir"];
        if (!colmap_dir_node.isNone()) {
            colmap_dir_ = colmap_dir_node.string();
        }
    }

    // ========== 优化参数 ==========
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();  // 最大迭代次数
    // ========== 位置学习率参数 ==========
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();  // 位置学习率初始值
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();  // 位置学习率最终值
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();  // 位置学习率延迟倍数
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();  // 位置学习率最大步数
    // ========== 其他参数学习率 ==========
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();  // 特征（颜色）学习率
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();  // 不透明度学习率
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();  // 缩放学习率
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();  // 旋转学习率

    // ========== 密度化参数 ==========
    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();  // 密度化百分比阈值
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();  // SSIM 损失权重（与 L1 损失的平衡，通常 0.2）
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();  // 密度化执行间隔（迭代次数，通常 100）
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();  // 不透明度重置间隔（迭代次数）
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();  // 从哪个迭代开始密度化
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();  // 到哪个迭代停止密度化
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();  // 密度化梯度阈值（超过此值进行分裂/克隆）

    // ========== 剪枝参数 ==========
    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();  // 在此迭代后剪枝大点
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();  // 密度化最小不透明度阈值
    {
        cv::FileNode n = settings_file["Optimization.lambda_isotropic_scaling"];
        if (!n.isNone()) {
            isotropic_scaling_reg_weight_ = n.operator float();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_isotropic_scaling_min_ratio"];
        if (!n.isNone()) {
            isotropic_scaling_reg_min_ratio_ = n.operator float();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_isotropic_scaling_progress_power"];
        if (!n.isNone()) {
            isotropic_scaling_reg_progress_power_ = n.operator float();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_isotropic_scaling_loss_power"];
        if (!n.isNone()) {
            isotropic_scaling_reg_loss_power_ = n.operator float();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.lambda_isotropic_scaling_new_point_iters"];
        if (!n.isNone()) {
            isotropic_scaling_reg_new_point_iters_ = n.operator int();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.multiview_old_frame_samples"];
        if (!n.isNone()) {
            multiview_old_frame_samples_ = n.operator int();
        }
        multiview_old_frame_samples_ = std::max(0, multiview_old_frame_samples_);
    }
    {
        cv::FileNode n = settings_file["Optimization.multiview_old_frame_loss_weight"];
        if (!n.isNone()) {
            multiview_old_frame_loss_weight_ = n.operator float();
        }
        multiview_old_frame_loss_weight_ = std::max(0.0f, multiview_old_frame_loss_weight_);
    }
    {
        cv::FileNode n = settings_file["Optimization.multiview_old_gaussian_min_age_iters"];
        if (!n.isNone()) {
            multiview_old_gaussian_min_age_iters_ = n.operator int();
        }
        multiview_old_gaussian_min_age_iters_ = std::max(0, multiview_old_gaussian_min_age_iters_);
    }
    {
        cv::FileNode n = settings_file["Optimization.enable_safety_prune"];
        if (!n.isNone()) {
            enable_safety_prune_ = (n.operator int()) != 0;
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.safety_prune_interval"];
        if (!n.isNone()) {
            safety_prune_interval_ = n.operator int();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.safety_prune_max_world_scale_ratio"];
        if (!n.isNone()) {
            safety_prune_max_world_scale_ratio_ = n.operator float();
        }
    }
    {
        cv::FileNode n = settings_file["Optimization.safety_prune_max_center_dist_ratio"];
        if (!n.isNone()) {
            safety_prune_max_center_dist_ratio_ = n.operator float();
        }
    }

    // ========== 滑动窗口主动位姿优化参数 ==========
    {
        cv::FileNode n = settings_file["PoseOpt.enable"];
        if (n.isNone()) {
            std::cout << "[GaussianMapper] PoseOpt.enable not found in config: "
                      << cfg_path << " (pose optimization disabled)" << std::endl;
        }
        enable_pose_optimization_ = !n.isNone() && (n.operator int() != 0);
    }
    if (enable_pose_optimization_) {
        cv::FileNode n;
        n = settings_file["PoseOpt.window_size"];
        if (!n.isNone()) pose_opt_cfg_.window_size    = n.operator int();
        n = settings_file["PoseOpt.slide_size"];
        if (!n.isNone()) pose_opt_cfg_.slide_size     = n.operator int();
        n = settings_file["PoseOpt.anchor_size"];
        if (!n.isNone()) pose_opt_cfg_.anchor_size    = n.operator int();
        n = settings_file["PoseOpt.batch_ratio"];
        if (!n.isNone()) pose_opt_cfg_.batch_ratio    = n.operator float();
        n = settings_file["PoseOpt.steps_per_call"];
        if (!n.isNone()) pose_opt_cfg_.steps_per_call = n.operator int();
        n = settings_file["PoseOpt.lr"];
        if (!n.isNone()) pose_opt_cfg_.lr             = n.operator float();
        n = settings_file["PoseOpt.call_interval"];
        if (!n.isNone()) pose_opt_cfg_.call_interval  = n.operator int();
        n = settings_file["PoseOpt.min_iteration"];
        if (!n.isNone()) pose_opt_cfg_.min_iteration  = n.operator int();
        n = settings_file["PoseOpt.lambda_ssim"];
        if (!n.isNone()) pose_opt_cfg_.lambda_ssim    = n.operator float();
        n = settings_file["PoseOpt.rot_reg_weight"];
        if (!n.isNone()) pose_opt_cfg_.rot_reg_weight = n.operator float();
        n = settings_file["PoseOpt.trans_reg_weight"];
        if (!n.isNone()) pose_opt_cfg_.trans_reg_weight = n.operator float();
        n = settings_file["PoseOpt.max_grad_norm"];
        if (!n.isNone()) pose_opt_cfg_.max_grad_norm  = n.operator float();
        n = settings_file["PoseOpt.absorb_rot_min_deg"];
        if (!n.isNone()) pose_opt_cfg_.absorb_rot_min_deg = n.operator float();
        n = settings_file["PoseOpt.absorb_trans_min"];
        if (!n.isNone()) pose_opt_cfg_.absorb_trans_min = n.operator float();
        n = settings_file["PoseOpt.max_optimized_fids_per_absorb"];
        if (!n.isNone()) pose_opt_cfg_.max_optimized_fids_per_absorb = n.operator int();
        n = settings_file["PoseOpt.cache_downsample"];
        if (!n.isNone()) pose_opt_cfg_.cache_downsample = n.operator int();
        n = settings_file["PoseOpt.use_lowres_cache"];
        if (!n.isNone()) pose_opt_cfg_.use_lowres_cache = (n.operator int()) != 0;
        n = settings_file["PoseOpt.trigger_on_new_keyframe"];
        if (!n.isNone()) pose_opt_trigger_on_new_keyframe_ = (n.operator int()) != 0;
        n = settings_file["PoseOpt.steps_on_new_keyframe"];
        if (!n.isNone()) pose_opt_steps_on_new_keyframe_ = n.operator int();
        n = settings_file["PoseOpt.max_chunks_per_loop"];
        if (!n.isNone()) pose_opt_max_chunks_per_loop_ = n.operator int();
        n = settings_file["PoseOpt.max_steps_per_loop"];
        if (!n.isNone()) pose_opt_max_steps_per_loop_ = n.operator int();
        n = settings_file["PoseOpt.max_millis_per_loop"];
        if (!n.isNone()) pose_opt_max_millis_per_loop_ = n.operator int();
        n = settings_file["PoseOpt.task_queue_limit"];
        if (!n.isNone()) pose_opt_task_queue_limit_ = static_cast<std::size_t>(std::max(1, n.operator int()));
        n = settings_file["PoseOpt.skip_when_new_keyframe_pending"];
        if (!n.isNone()) pose_opt_skip_when_new_keyframe_pending_ = (n.operator int()) != 0;
        n = settings_file["PoseOpt.busy_loop_ms_soft"];
        if (!n.isNone()) pose_opt_busy_loop_ms_soft_ = n.operator float();
        n = settings_file["PoseOpt.busy_loop_ms_hard"];
        if (!n.isNone()) pose_opt_busy_loop_ms_hard_ = n.operator float();
        n = settings_file["PoseOpt.busy_batch_ratio_scale"];
        if (!n.isNone()) pose_opt_busy_batch_ratio_scale_ = n.operator float();
        n = settings_file["PoseOpt.extra_train_budget_per_round"];
        if (!n.isNone()) pose_opt_extra_train_budget_per_round_ = n.operator int();

        pose_opt_max_chunks_per_loop_ = std::max(1, pose_opt_max_chunks_per_loop_);
        pose_opt_max_steps_per_loop_ = std::max(1, pose_opt_max_steps_per_loop_);
        pose_opt_max_millis_per_loop_ = std::max(1, pose_opt_max_millis_per_loop_);
        pose_opt_steps_on_new_keyframe_ = std::max(1, pose_opt_steps_on_new_keyframe_);
        pose_opt_busy_loop_ms_soft_ = std::max(1.0f, pose_opt_busy_loop_ms_soft_);
        pose_opt_busy_loop_ms_hard_ = std::max(pose_opt_busy_loop_ms_soft_, pose_opt_busy_loop_ms_hard_);
        pose_opt_busy_batch_ratio_scale_ = std::max(0.0f, std::min(1.0f, pose_opt_busy_batch_ratio_scale_));
        pose_opt_extra_train_budget_per_round_ = std::max(0, pose_opt_extra_train_budget_per_round_);

        pose_optimizer_ = std::make_unique<SlidingWindowPoseOptimizer>(
            pose_opt_cfg_, device_type_);

        std::cout << "[GaussianMapper] L1 SSIM PoseBA ："
                  << " window=" << pose_opt_cfg_.window_size
                  << " slide=" << pose_opt_cfg_.slide_size
                  << " anchor=" << pose_opt_cfg_.anchor_size
                  << " batch_ratio=" << pose_opt_cfg_.batch_ratio
                  << " steps=" << pose_opt_cfg_.steps_per_call
                  << " lr="    << pose_opt_cfg_.lr
                  << " interval=" << pose_opt_cfg_.call_interval
                  << " min_iter=" << pose_opt_cfg_.min_iteration
                  << " lambda_ssim=" << pose_opt_cfg_.lambda_ssim
                  << " rot_reg=" << pose_opt_cfg_.rot_reg_weight
                  << " trans_reg=" << pose_opt_cfg_.trans_reg_weight
                  << " grad_clip=" << pose_opt_cfg_.max_grad_norm
                  << " absorb_rot_min_deg=" << pose_opt_cfg_.absorb_rot_min_deg
                  << " absorb_trans_min=" << pose_opt_cfg_.absorb_trans_min
                  << " max_optimized_fids_per_absorb=" << pose_opt_cfg_.max_optimized_fids_per_absorb
                  << " cache_downsample=" << pose_opt_cfg_.cache_downsample
                  << " use_lowres_cache=" << (pose_opt_cfg_.use_lowres_cache ? 1 : 0)
                  << " trigger_on_new_kf=" << (pose_opt_trigger_on_new_keyframe_ ? 1 : 0)
                  << " steps_on_new_kf=" << pose_opt_steps_on_new_keyframe_
                  << " max_chunks_per_loop=" << pose_opt_max_chunks_per_loop_
                  << " max_steps_per_loop=" << pose_opt_max_steps_per_loop_
                  << " max_millis_per_loop=" << pose_opt_max_millis_per_loop_
                  << " task_queue_limit=" << pose_opt_task_queue_limit_
                  << " skip_when_new_kf_pending=" << (pose_opt_skip_when_new_keyframe_pending_ ? 1 : 0)
                  << " busy_loop_ms_soft=" << pose_opt_busy_loop_ms_soft_
                  << " busy_loop_ms_hard=" << pose_opt_busy_loop_ms_hard_
                  << " busy_batch_ratio_scale=" << pose_opt_busy_batch_ratio_scale_
                  << " extra_train_budget_per_round=" << pose_opt_extra_train_budget_per_round_
                  << std::endl;
    }

    // ========== 查看器参数 ==========
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();  // 查看器图像缩放比例（子视角）
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();  // 查看器图像缩放比例（主视角）
}

// ========== 主运行函数：执行完整的建图流程 ==========
// 包含三个阶段：
//   1. 初始建图：等待足够的关键帧和地图点，创建初始 Gaussian 模型
//   2. 增量建图：持续处理新的关键帧和地图点，优化模型
//   3. 尾部优化：SLAM 结束后继续优化模型
void GaussianMapper::run()
{
    // ========== 第一阶段：初始 Gaussian 建图 ==========
    while (!isStopped()) {
        if (!data_provider_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ========== 先等待数据提供者初始化 ==========
        if (!data_provider_->isInitialized()) {
            if (data_provider_->isFinished()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ========== 初始化完成后开始缓存关键帧 ==========
        auto new_keyframes = data_provider_->getNewKeyFrames();
        for (const auto& kf_data : new_keyframes) {
            if (!kf_data || kf_data->image.empty()) {
                continue;
            }
            pending_keyframes_.push_back(kf_data);
        }

        // ========== 缓存数量满足阈值后，直接在此处处理并添加 ==========
        if (pending_keyframes_.size() >= min_num_initial_map_kfs_) {
            std::cout << "[Gaussian Mapper] Data provider initialized. Starting Gaussian Mapping..." << std::endl;

            if (do_initial_sfm_refine_) {
                refinePendingKeyframesBySfm(pending_keyframes_);
            }

            for (const auto& kf_data : pending_keyframes_) {
                if (!kf_data || kf_data->image.empty()) {
                    continue;
                }

                unsigned long kfid = static_cast<unsigned long>(kf_data->timestamp * 1000);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                Sophus::SE3d T_cw = kf_data->pose.cast<double>();

                if (pkf) {
                    pkf->setPose(T_cw.unit_quaternion(), T_cw.translation());
                    pkf->computeTransformTensors();
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    std::vector<float> kps;
                    std::vector<float> kps3d;

                    if (!kf_data->points.empty()) {
                        kps3d.reserve(kf_data->points.size() * 3);
                        for (size_t i = 0; i < kf_data->points.size(); ++i) {
                            const auto& pt = kf_data->points[i];
                            Eigen::Vector3f color(0.5f, 0.5f, 0.5f);
                            if (i < kf_data->colors.size()) {
                                color = kf_data->colors[i];
                            }
                            kps3d.push_back(pt.x());
                            kps3d.push_back(pt.y());
                            kps3d.push_back(pt.z());

                            Point3D p3d;
                            p3d.xyz_ = Eigen::Vector3d(pt.x(), pt.y(), pt.z());
                            p3d.color_ = color;
                            p3d.color256_ = Eigen::Matrix<uint8_t, 3, 1>(
                                static_cast<uint8_t>(std::round(std::max(0.0f, std::min(1.0f, color.x())) * 255.0f)),
                                static_cast<uint8_t>(std::round(std::max(0.0f, std::min(1.0f, color.y())) * 255.0f)),
                                static_cast<uint8_t>(std::round(std::max(0.0f, std::min(1.0f, color.z())) * 255.0f)));
                            scene_->cachePoint3D(next_point3d_id_++, p3d);
                        }
                    }

                    std::tuple<unsigned long, unsigned long, Sophus::SE3f, cv::Mat, bool, cv::Mat, std::vector<float>, std::vector<float>, std::string>
                        new_kf_data(
                            kfid,
                            0,
                            T_cw.cast<float>(),
                            kf_data->image,
                            false,
                            kf_data->depth_image,
                            kps,
                            kps3d,
                            std::to_string(kfid));

                    handleNewKeyframe(new_kf_data);
                    saveKeyframeToColmap(kf_data, kfid);
                }
            }
            pending_keyframes_.clear();

            // ========== 准备训练：创建 Gaussian 模型 ==========
            {
                // ========== 加锁保护模型（线程安全） ==========
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // 计算场景范围（用于空间学习率缩放）
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());  // 获取场景半径
                // 从缓存的点云创建 Gaussian 模型
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                // ========== 加锁保护设置参数 ==========
                std::unique_lock<std::mutex> lock(mutex_settings_);
                // 设置优化器（Adam）和学习率
                gaussians_->trainingSetup(opt_params_);
            }

            // ========== 执行一次训练迭代 ==========
            trainForOneIteration();

            // ========== 完成初始建图循环 ==========
            initial_mapped_ = true;  // 标记初始建图已完成
            break;                   // 退出初始建图循环
        }

        if (data_provider_->isFinished()) {
            // 数据提供者标记为已结束，退出循环
            break;
        }

        // ========== 初始条件未满足，等待 ==========
        // Initial conditions not satisfied
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 休眠 1 毫秒，避免 CPU 占用过高
    }

    // Shutdown before initialization has no Gaussian model to export.
    if (!initial_mapped_) {
        signalStop();
        return;
    }

    // ========== 第二阶段：增量 Gaussian 建图 ==========
    int SLAM_stop_iter = 0;  // 记录 SLAM 停止时的迭代次数
    while (!isStopped()) {
        {
            std::unique_lock<std::mutex> lock_status(mutex_status_);
            if (tail_optimization_requested_) {
                tail_optimization_requested_ = false;
                SLAM_ended_ = true;
                std::cout << "[Gaussian Mapper] Tail optimization requested from viewer, leaving incremental loop." << std::endl;
                break;
            }
        }

        // ========== 检查是否满足增量建图条件 ==========
        // 每次训练前，同步 VINS 滑动窗口内的最新位姿
        // 这种"全量刷新"策略适合 VINS，因为 VINS 会不断优化窗口内的所有位姿
        if (initial_mapped_) {
            syncWithVinsSlidingWindow();
            
            // 如果开启了关键帧剔除
            if (cull_keyframes_) cullKeyframes();
        }
        // ========== 执行一次训练迭代 ==========
        trainForOneIteration();
        // ========== 检查数据源是否已结束 ==========
        if (data_provider_->isFinished()) {
            SLAM_stop_iter = getIteration();  // 记录停止时的迭代次数
            SLAM_ended_ = true;               // 标记数据源已结束
        }
        // ========== 检查是否应该退出增量建图循环 ==========
        // 条件： 达到最大迭代次数
        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // ========== 第三阶段：尾部 Gaussian 优化 ==========
    // 在 SLAM 结束后继续优化模型，直到密度化完成或达到停止条件
    int densify_interval = densifyInterval();        // 获取密度化间隔
    int n_delay_iters = densify_interval * 0.8;     // 延迟迭代数（密度化间隔的 80%）
    // 继续训练直到：
    //   1. SLAM 停止后的延迟迭代数已过，且
    //   2. 当前迭代不在密度化周期内，且
    //   3. 不保持训练
    while (!isStopped() && isKeepingTraining()) {
        trainForOneIteration();                      // 执行一次训练迭代
        densify_interval = densifyInterval();        // 重新获取密度化间隔（可能变化）
        n_delay_iters = densify_interval * 0.8;     // 重新计算延迟迭代数
    }

    // ========== 保存结果并清理 ==========
    renderAndRecordAllKeyframes("_shutdown");       // 渲染并记录所有关键帧（带 "_shutdown" 后缀）
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");  // 保存 PLY 文件
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");  // 记录关键帧使用次数（最终）

    signalStop();  // 发送停止信号
}

bool GaussianMapper::refinePendingKeyframesBySfm(
    std::vector<std::shared_ptr<gaussian_mapping::KeyFrameData>>& keyframes)
{
    if (keyframes.size() < 2 || scene_->cameras_.empty()) {
        return false;
    }

    Camera& cam = scene_->cameras_.at(0);
    const float fx = cam.params_[0];
    const float fy = cam.params_[1];
    const float cx = cam.params_[2];
    const float cy = cam.params_[3];
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                          0.0, fy, cy,
                                          0.0, 0.0, 1.0);

    auto orb = cv::ORB::create(2000);
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);

    int refined_pairs = 0;
    for (size_t i = 1; i < keyframes.size(); ++i) {
        auto kf_prev = keyframes[i - 1];
        auto kf_curr = keyframes[i];
        if (!kf_prev || !kf_curr || kf_prev->image.empty() || kf_curr->image.empty()) {
            continue;
        }

        cv::Mat img1_u8, img2_u8;
        if (kf_prev->image.type() == CV_32FC3) {
            kf_prev->image.convertTo(img1_u8, CV_8UC3, 255.0);
        } else {
            kf_prev->image.convertTo(img1_u8, CV_8UC3);
        }
        if (kf_curr->image.type() == CV_32FC3) {
            kf_curr->image.convertTo(img2_u8, CV_8UC3, 255.0);
        } else {
            kf_curr->image.convertTo(img2_u8, CV_8UC3);
        }

        cv::Mat gray1, gray2;
        cv::cvtColor(img1_u8, gray1, cv::COLOR_RGB2GRAY);
        cv::cvtColor(img2_u8, gray2, cv::COLOR_RGB2GRAY);

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat desc1, desc2;
        orb->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
        orb->detectAndCompute(gray2, cv::noArray(), kp2, desc2);
        if (desc1.empty() || desc2.empty()) {
            continue;
        }

        std::vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);
        if (matches.size() < 30) {
            continue;
        }

        std::sort(matches.begin(), matches.end(),
                  [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
        const size_t keep_n = std::min<size_t>(matches.size(), 300);

        std::vector<cv::Point2f> pts1, pts2;
        pts1.reserve(keep_n);
        pts2.reserve(keep_n);
        for (size_t m = 0; m < keep_n; ++m) {
            pts1.push_back(kp1[matches[m].queryIdx].pt);
            pts2.push_back(kp2[matches[m].trainIdx].pt);
        }
        if (pts1.size() < 20) {
            continue;
        }

        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(
            pts1, pts2, K, cv::RANSAC, 0.999, 1.0, inlier_mask);
        if (E.empty()) {
            continue;
        }

        cv::Mat R_cv, t_cv;
        int inliers = cv::recoverPose(E, pts1, pts2, K, R_cv, t_cv, inlier_mask);
        if (inliers < initial_sfm_min_inliers_) {
            continue;
        }

        Sophus::SE3f Tcw_prev = kf_prev->pose;
        Sophus::SE3f Tcw_curr_vins = kf_curr->pose;
        Eigen::Vector3f C_prev = -(Tcw_prev.rotationMatrix().transpose() * Tcw_prev.translation());
        Eigen::Vector3f C_curr = -(Tcw_curr_vins.rotationMatrix().transpose() * Tcw_curr_vins.translation());
        float baseline = (C_curr - C_prev).norm();
        if (baseline < 1e-5f) baseline = 1.0f;

        Eigen::Matrix3f R_21;
        Eigen::Vector3f t_21;
        for (int r = 0; r < 3; ++r) {
            t_21(r) = static_cast<float>(t_cv.at<double>(r, 0)) * baseline;
            for (int c = 0; c < 3; ++c) {
                R_21(r, c) = static_cast<float>(R_cv.at<double>(r, c));
            }
        }

        Sophus::SE3f T_21(R_21, t_21);          // cam(i-1) -> cam(i)
        kf_curr->pose = T_21 * Tcw_prev;        // Tworld->cam(i)

        // 位姿变了后，同步更新该关键帧携带的稀疏点世界坐标，避免与位姿不一致
        if (!kf_curr->points.empty()) {
            for (auto& pw : kf_curr->points) {
                pw = transformWorldPointByPoseUpdate(pw, Tcw_curr_vins, kf_curr->pose);
            }
        }
        ++refined_pairs;
    }

    if (refined_pairs > 0) {
        std::cout << "[Gaussian Mapper] Initial SfM refinement done, refined pairs: "
                  << refined_pairs << "/" << (keyframes.size() - 1) << std::endl;
        return true;
    }

    std::cout << "[Gaussian Mapper] Initial SfM refinement skipped (insufficient inliers)." << std::endl;
    return false;
}

void GaussianMapper::syncWithVinsSlidingWindow()
{
    struct PcdCandidate {
        std::shared_ptr<gaussian_mapping::KeyFrameData> kf_data;
        Sophus::SE3f Tcw_before;
        bool existed_in_scene;
    };

    // 使用数据提供者接口获取新关键帧
    auto new_keyframes = data_provider_->getNewKeyFrames();
    bool inserted_new_kf = false;
    std::vector<unsigned long> inserted_kfids;
    std::vector<PcdCandidate> pcd_candidates;
    
    if (new_keyframes.empty()) {
        return;  // 没有新关键帧
    }
    
    if (!initial_mapped_) {
        return;
    }

    // 处理每个新关键帧
    for (const auto& kf_data : new_keyframes) {
        if (!kf_data || kf_data->image.empty()) {
            continue;
        }

        unsigned long kfid = static_cast<unsigned long>(kf_data->timestamp * 1000);
        std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
        Sophus::SE3d T_cw = kf_data->pose.cast<double>();
        const bool is_depth_fusion_update = kf_data->depth_fusion_update;

        if (pkf) {
            if (!is_depth_fusion_update) {
                pkf->setPose(T_cw.unit_quaternion(), T_cw.translation());
                pkf->computeTransformTensors();
            }
            if (!kf_data->depth_image.empty()) {
                pkf->img_auxiliary_undist_ = kf_data->depth_image;
            }
            increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
        }
        else {
            if (is_depth_fusion_update) {
                // 异步补点增量到达时若关键帧尚未入场景，直接跳过，避免以旧位姿新建关键帧。
                continue;
            }

            std::vector<float> kps;
            std::vector<float> kps3d;

            if (!kf_data->points.empty()) {
                kps3d.reserve(kf_data->points.size() * 3);
                for (size_t i = 0; i < kf_data->points.size(); ++i) {
                    const auto& pt = kf_data->points[i];
                    Eigen::Vector3f color(0.5f, 0.5f, 0.5f);
                    if (i < kf_data->colors.size()) {
                        color = kf_data->colors[i];
                    }
                    kps3d.push_back(pt.x());
                    kps3d.push_back(pt.y());
                    kps3d.push_back(pt.z());
                }
            }

            std::tuple<unsigned long, unsigned long, Sophus::SE3f, cv::Mat, bool, cv::Mat, std::vector<float>, std::vector<float>, std::string>
                new_kf_data(
                    kfid,
                    0,
                    T_cw.cast<float>(),
                    kf_data->image,
                    false,
                    kf_data->depth_image,
                    kps,
                    kps3d,
                    std::to_string(kfid));

            handleNewKeyframe(new_kf_data);
            saveKeyframeToColmap(kf_data, kfid);
            inserted_new_kf = true;
            inserted_kfids.push_back(kfid);
        }

        if (kf_data->points.size() > 30) {
            pcd_candidates.push_back(PcdCandidate{kf_data, T_cw.cast<float>(), static_cast<bool>(pkf)});
        }
    }

    // 老帧筛选只在新关键帧插入时做一次，避免每步训练重复筛选造成性能损耗。
    if (inserted_new_kf) {
        boostHistoricalFrameUsageOnNewKeyframes(inserted_kfids);
    }

    // 根据最新关键帧位姿（可能已被PoseOpt优化）再上传对应点云，保证位姿-点云一致
    if (!pcd_candidates.empty()) {
        std::vector<float> new_points;
        std::vector<float> new_colors;

        for (const auto& item : pcd_candidates) {
            const auto& kf_data = item.kf_data;
            const auto& Tcw_before = item.Tcw_before;
            if (!kf_data) continue;

            unsigned long kfid = static_cast<unsigned long>(kf_data->timestamp * 1000);
            auto pkf_now = scene_->getKeyframe(kfid);
            if (!pkf_now) continue;

            // 异步补点通常对应已存在关键帧；提升使用次数以便新点更快参与训练采样。
            if (item.existed_in_scene) {
                increaseKeyframeTimesOfUse(pkf_now, local_BA_increased_times_of_use_);
            }

            Sophus::SE3f Tcw_after = pkf_now->getPosef();

            for (size_t i = 0; i < kf_data->points.size(); ++i) {
                Eigen::Vector3f pw = transformWorldPointByPoseUpdate(
                    kf_data->points[i], Tcw_before, Tcw_after);
                Eigen::Vector3f color(0.5f, 0.5f, 0.5f);
                if (i < kf_data->colors.size()) {
                    color = kf_data->colors[i];
                }

                new_points.push_back(pw.x());
                new_points.push_back(pw.y());
                new_points.push_back(pw.z());
                new_colors.push_back(color.x());
                new_colors.push_back(color.y());
                new_colors.push_back(color.z());
            }
        }

        if (!new_points.empty()) {
            torch::NoGradGuard no_grad;
            std::unique_lock<std::mutex> lock_render(mutex_render_);
            gaussians_->increasePcd(new_points, new_colors, getIteration());
        }
    }
}

void GaussianMapper::syncNewKeyframes()
{
    syncWithVinsSlidingWindow();
}

void GaussianMapper::boostUsageForOptimizedFids(const std::vector<unsigned long>& optimized_fids)
{
    int budget_left = std::max(0, pose_opt_extra_train_budget_per_round_);
    if (budget_left <= 0) {
        return;
    }

    for (const auto fid : optimized_fids) {
        if (budget_left <= 0) break;
        auto pkf_opt = scene_->getKeyframe(fid);
        if (pkf_opt) {
            increaseKeyframeTimesOfUse(pkf_opt, 1);
            --budget_left;
        }
    }
}

bool GaussianMapper::shouldSkipPoseOptByLoad(bool force_run) const
{
    if (!force_run && pose_opt_skip_when_new_keyframe_pending_ && data_provider_ && data_provider_->hasNewKeyFrame()) {
        return true;
    }
    if (mapping_loop_ema_ms_ >= pose_opt_busy_loop_ms_hard_) {
        return true;
    }
    return false;
}

float GaussianMapper::getPoseOptBatchRatioByLoad() const
{
    if (mapping_loop_ema_ms_ >= pose_opt_busy_loop_ms_soft_) {
        return pose_opt_cfg_.batch_ratio * pose_opt_busy_batch_ratio_scale_;
    }
    return pose_opt_cfg_.batch_ratio;
}

void GaussianMapper::enqueuePoseOptTask(int target_steps, const std::string& source)
{
    if (!enable_pose_optimization_ || !pose_optimizer_) return;
    if (getIteration() < pose_opt_cfg_.min_iteration) return;
    if (!pose_optimizer_->isChunkReady()) return;

    const auto chunk = pose_optimizer_->getCurrentChunkInfo();
    if (chunk.active_count <= chunk.anchor_count || chunk.frame_fids.empty()) return;

    if (!pose_opt_task_queue_.empty()) {
        const auto& back = pose_opt_task_queue_.back();
        if (!back.completed && back.frame_fids == chunk.frame_fids) {
            pose_opt_task_queue_.back().target_steps = std::max(
                pose_opt_task_queue_.back().target_steps,
                std::max(1, target_steps));
            return;
        }
    }

    while (pose_opt_task_queue_.size() >= pose_opt_task_queue_limit_) {
        pose_opt_task_queue_.pop_front();
    }
    pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);

    PoseOptTask task;
    task.task_id = pose_opt_next_task_id_++;
    task.frame_fids = chunk.frame_fids;
    task.anchor_fids = chunk.anchor_fids;
    task.active_count = chunk.active_count;
    task.anchor_count = chunk.anchor_count;
    task.target_steps = std::max(1, target_steps);
    task.steps_done = 0;
    task.completed = false;
    task.source = source;
    pose_opt_task_queue_.push_back(std::move(task));
    pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);
}

void GaussianMapper::finishOnePoseOptTask(PoseOptTask& task)
{
    if (!task.completed) {
        return;
    }
    if (pose_optimizer_) {
        pose_optimizer_->slideWindowAfterOptimize();
    }

    std::cout << "[PoseOptScheduler] finish task=" << task.task_id
              << " src=" << task.source
              << " steps=" << task.steps_done << "/" << task.target_steps
              << " frames=" << task.active_count
              << " anchors=" << task.anchor_count
              << std::endl;
}

void GaussianMapper::runPoseOptWithBudget(bool force_run)
{
    pose_opt_dbg_last_iteration_.store(getIteration(), std::memory_order_relaxed);
    pose_opt_dbg_last_try_lock_failed_.store(0, std::memory_order_relaxed);
    pose_opt_dbg_last_skipped_by_load_.store(0, std::memory_order_relaxed);
    if (!enable_pose_optimization_ || !pose_optimizer_) {
        pose_opt_dbg_last_status_code_.store(1, std::memory_order_relaxed);
        return;
    }
    if (getIteration() < pose_opt_cfg_.min_iteration) {
        if (getIteration() % 100 == 0) {
            std::cout << "[PoseOptScheduler] gated by min_iteration: iter=" << getIteration()
                      << " < " << pose_opt_cfg_.min_iteration << std::endl;
        }
        pose_opt_task_queue_.clear();
        pose_opt_dbg_queue_size_.store(0, std::memory_order_relaxed);
        pose_opt_dbg_last_status_code_.store(10, std::memory_order_relaxed);
        return;
    }
    if (shouldSkipPoseOptByLoad(force_run)) {
        pose_opt_dbg_last_skipped_by_load_.store(1, std::memory_order_relaxed);
        pose_opt_dbg_last_status_code_.store(2, std::memory_order_relaxed);
        return;
    }

    auto loop_start = std::chrono::steady_clock::now();
    const int max_chunks = std::max(1, pose_opt_max_chunks_per_loop_);
    const int max_steps = std::max(1, pose_opt_max_steps_per_loop_);
    const int max_millis = std::max(1, pose_opt_max_millis_per_loop_);

    int chunks_done = 0;
    int steps_done = 0;
    int status_code = 4;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - loop_start).count();
        if (elapsed_ms >= max_millis) {
            status_code = 7;
            break;
        }
        if (chunks_done >= max_chunks) {
            status_code = 8;
            break;
        }
        if (steps_done >= max_steps) {
            status_code = 9;
            break;
        }

        if (pose_opt_task_queue_.empty() && pose_optimizer_->isChunkReady()) {
            enqueuePoseOptTask(pose_opt_cfg_.steps_per_call, "auto_chunk_ready");
        }
        if (pose_opt_task_queue_.empty()) {
            status_code = 4;
            break;
        }
        pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);

        auto& task = pose_opt_task_queue_.front();
        if (task.completed) {
            finishOnePoseOptTask(task);
            pose_opt_task_queue_.pop_front();
            pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);
            ++chunks_done;
            continue;
        }

        const int task_left = std::max(0, task.target_steps - task.steps_done);
        const int budget_left = std::max(0, max_steps - steps_done);
        if (task_left <= 0 || budget_left <= 0) {
            task.completed = true;
            finishOnePoseOptTask(task);
            pose_opt_task_queue_.pop_front();
            pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);
            ++chunks_done;
            continue;
        }

        const int run_steps = std::max(1, std::min(task_left, budget_left));
        const float batch_ratio_override = getPoseOptBatchRatioByLoad();
        pose_opt_dbg_last_batch_ratio_.store(batch_ratio_override, std::memory_order_relaxed);
        auto poseopt_start_timing = std::chrono::steady_clock::now();
        {
            std::unique_lock<std::mutex> lock_render(mutex_render_, std::defer_lock);
            if (!lock_render.try_lock()) {
                pose_opt_dbg_last_try_lock_failed_.store(1, std::memory_order_relaxed);
                status_code = 3;
                break;
            }
            pose_optimizer_->optimize(gaussians_, pipe_params_, background_, run_steps, batch_ratio_override);
            pose_opt_dbg_total_optimize_calls_.fetch_add(1, std::memory_order_relaxed);
        }
        boostUsageForOptimizedFids(pose_optimizer_->getLastOptimizedFids());
        status_code = 0;

        auto poseopt_end_timing = std::chrono::steady_clock::now();
        const double poseopt_elapsed_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(poseopt_end_timing - poseopt_start_timing).count();
        task.steps_done += run_steps;
        steps_done += run_steps;

        std::size_t task_id_snapshot = task.task_id;
        std::string task_src_snapshot = task.source;
        int task_steps_done_snapshot = task.steps_done;
        int task_target_steps_snapshot = task.target_steps;

        if (task.steps_done >= task.target_steps) {
            task.completed = true;
            finishOnePoseOptTask(task);
            pose_opt_task_queue_.pop_front();
            pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);
            ++chunks_done;
        }

        std::cout << std::fixed << std::setprecision(8)
                  << "[PoseOptScheduler] iter=" << getIteration()
                  << " task=" << task_id_snapshot
                  << " src=" << task_src_snapshot
                  << " run_steps=" << run_steps
                  << " task_progress=" << task_steps_done_snapshot << "/" << task_target_steps_snapshot
                  << " loop_steps=" << steps_done << "/" << max_steps
                  << " loop_chunks=" << chunks_done << "/" << max_chunks
                  << " queue=" << pose_opt_task_queue_.size()
                  << " elapsed=" << poseopt_elapsed_s << "s"
                  << std::endl;
    }

    const auto end_now = std::chrono::steady_clock::now();
    const int elapsed_ms_total = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_now - loop_start).count());
    pose_opt_dbg_last_run_steps_.store(steps_done, std::memory_order_relaxed);
    pose_opt_dbg_last_run_chunks_.store(chunks_done, std::memory_order_relaxed);
    pose_opt_dbg_last_run_elapsed_ms_.store(elapsed_ms_total, std::memory_order_relaxed);
    pose_opt_dbg_last_status_code_.store(status_code, std::memory_order_relaxed);
    pose_opt_dbg_queue_size_.store(static_cast<int>(pose_opt_task_queue_.size()), std::memory_order_relaxed);
}

void GaussianMapper::saveKeyframeToColmap(
    const std::shared_ptr<gaussian_mapping::KeyFrameData>& kf_data,
    unsigned long kfid)
{
    if (!save_colmap_ || !kf_data) {
        return;
    }
    if (kf_data->image.empty()) {
        return;
    }
    if (scene_->cameras_.empty()) {
        return;
    }

    if (colmap_dir_.empty()) {
        colmap_dir_ = result_dir_ / "colmap";
    }

    Camera camera = scene_->cameras_.at(0);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(colmap_dir_)
    std::filesystem::path images_dir = colmap_dir_ / "images";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(images_dir)

    if (!colmap_initialized_) {
        colmap_cameras_.open(colmap_dir_ / "cameras.txt", std::ios::out);
        colmap_images_.open(colmap_dir_ / "images.txt", std::ios::out);
        colmap_points_.open(colmap_dir_ / "points3D.txt", std::ios::out);
        if (colmap_cameras_.is_open() && colmap_images_.is_open() && colmap_points_.is_open()) {
            colmap_cameras_ << "# Camera list with one line of data per camera:\n";
            colmap_cameras_ << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
            colmap_cameras_ << "# Number of cameras: 1\n";
            colmap_cameras_ << colmap_camera_id_ << " PINHOLE "
                            << camera.width_ << " " << camera.height_ << " "
                            << camera.params_[0] << " " << camera.params_[1] << " "
                            << camera.params_[2] << " " << camera.params_[3] << "\n";

            colmap_images_ << "# Image list with two lines of data per image:\n";
            colmap_images_ << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n";
            colmap_images_ << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";

            colmap_points_ << "# 3D point list with one line of data per point:\n";
            colmap_points_ << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n";

            colmap_initialized_ = true;
        }
    }

    if (!colmap_initialized_) {
        return;
    }

    cv::Mat img_rgb = kf_data->image;
    if (img_rgb.type() != CV_32FC3) {
        img_rgb.convertTo(img_rgb, CV_32FC3, 1.0 / 255.0);
    }

    cv::Mat img_undist;
    camera.undistortImage(img_rgb, img_undist);
    cv::Mat img_u8;
    img_undist.convertTo(img_u8, CV_8UC3, 255.0);
    cv::Mat img_bgr;
    cv::cvtColor(img_u8, img_bgr, cv::COLOR_RGB2BGR);

    const std::string image_name = "kf_" + std::to_string(kfid) + ".jpg";
    cv::imwrite((images_dir / image_name).string(), img_bgr);

    const int image_id = colmap_next_image_id_++;
    const auto T_cw_d = kf_data->pose.cast<double>();
    const Eigen::Quaterniond q = T_cw_d.unit_quaternion();
    const Eigen::Vector3d t = T_cw_d.translation();

    colmap_images_ << image_id << " "
                   << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                   << t.x() << " " << t.y() << " " << t.z() << " "
                   << colmap_camera_id_ << " " << image_name << "\n";
    colmap_images_ << "\n";

    for (size_t i = 0; i < kf_data->points.size(); ++i) {
        const auto& p = kf_data->points[i];
        int r = 128;
        int g = 128;
        int b = 128;
        if (i < kf_data->colors.size()) {
            const auto& c = kf_data->colors[i];
            r = static_cast<int>(std::round(std::max(0.0f, std::min(1.0f, c.x())) * 255.0f));
            g = static_cast<int>(std::round(std::max(0.0f, std::min(1.0f, c.y())) * 255.0f));
            b = static_cast<int>(std::round(std::max(0.0f, std::min(1.0f, c.z())) * 255.0f));
        }
        const long pid = colmap_next_point_id_++;
        colmap_points_ << pid << " "
                       << p.x() << " " << p.y() << " " << p.z() << " "
                       << r << " " << g << " " << b << " "
                       << 1.0 << "\n";
    }
    colmap_images_.flush();
    colmap_points_.flush();
}

void GaussianMapper::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA && opencv_cuda_available_) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

/**
 * @brief 训练迭代体：执行一次完整的训练迭代
 * 
 * 流程：
 *   1. 随机选择一个关键帧
 *   2. 渲染该关键帧视角的图像
 *   3. 计算损失（L1 + SSIM）
 *   4. 反向传播
 *   5. 密度化（分裂/克隆/剪枝）
 *   6. 优化器更新参数
 */
void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);  // 增加迭代次数
    auto iter_start_timing = std::chrono::steady_clock::now();  // 记录开始时间（用于计时）

    // 每100迭代随机补充一批老帧使用次数，避免老区域长期采样不足。
    boostOldFrameUsageEveryNIterations(100);

    // ========== 调试：打印 Gaussian 点数 ==========
    // if (gaussians_) {
        // std::cout << "[Gaussian Mapper][Debug] Gaussian count before train: "
        //           << gaussians_->getXYZ().size(0) << std::endl;
        // if (gaussians_->getXYZ().size(0) == 0) {
        //     std::cerr << "[Gaussian Mapper] Warning: no Gaussians yet, skip training iteration." << std::endl;
        //     increaseIteration(-1);
        //     return;
        // }
    // }

    // ========== 随机选择一个关键帧用于训练 ==========
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        // 如果没有可用的关键帧，回退迭代次数并返回
        increaseIteration(-1);
        return;
    }

    // ========== 记录关键帧使用次数（用于调试和分析） ==========
    writeKeyframeUsedTimes(result_dir_ / "used_times");

    // ========== 非活跃区域几何密度化（已注释，可选功能） ==========
    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    // ========== 确定训练层级和图像尺寸 ==========
    int training_level = num_gaus_pyramid_sub_levels_;  // 默认使用最高分辨率（原始尺寸）
    int image_height, image_width;                       // 图像尺寸
    torch::Tensor gt_image, mask;                        // 真实图像和掩码
    // ========== 如果启用高斯金字塔训练，选择当前层级 ==========
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();  // 获取当前应该使用的金字塔层级
    // ========== 根据训练层级选择图像和掩码 ==========
    if (training_level == num_gaus_pyramid_sub_levels_) {
        // ========== 使用原始分辨率 ==========
        image_height = viewpoint_cam->image_height_;     // 原始高度
        image_width = viewpoint_cam->image_width_;       // 原始宽度
        gt_image = viewpoint_cam->original_image_.to(device_type_);  // 原始图像（移动到目标设备）
        mask = undistort_mask_[viewpoint_cam->camera_id_];  // 原始尺寸的掩码
    }
    else {
        // ========== 使用金字塔子层级 ==========
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];  // 层级高度
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];     // 层级宽度
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].to(device_type_);  // 层级图像（移动到目标设备）
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];  // 层级掩码
    }

    // ========== 调试：打印渲染图像尺寸 ==========
    // std::cout << "[Gaussian Mapper][Debug] Render size: "
    //           << image_width << "x" << image_height
    //           << " (level " << training_level << ")" << std::endl;

    {
        // ========== 加锁保护 Gaussian 模型（线程安全） ==========
        // Mutex lock for usage of the gaussian model
        std::unique_lock<std::mutex> lock_render(mutex_render_);

    // ========== 每 1000 次迭代增加球谐函数阶数（渐进式训练） ==========
    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;  // 增加 SH 阶数（从 0 逐渐增加到最大阶数，提高颜色表达能力）
    // ========== 设置当前使用的 SH 阶数 ==========
    // 注释掉的代码：如果使用高斯金字塔训练，可以根据层级设置不同的 SH 阶数
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);  // 使用默认 SH 阶数

    // ========== 更新学习率 ==========
    // ========== 根据关键帧使用次数更新位置学习率 ==========
    int used_times = kfs_used_times_[viewpoint_cam->fid_];  // 获取该关键帧的使用次数
    // 步数 = min(使用次数, 最大步数)（限制步数上限）
    int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
    float position_lr = gaussians_->updateLearningRate(step);  // 根据步数更新学习率（指数衰减）
    setPositionLearningRateInit(position_lr);  // 更新初始位置学习率

    // ========== 设置其他参数的学习率（固定值） ==========
    gaussians_->setFeatureLearningRate(featureLearningRate());   // 特征学习率
    gaussians_->setOpacityLearningRate(opacityLearningRate());   // 不透明度学习率
    gaussians_->setScalingLearningRate(scalingLearningRate());   // 缩放学习率
    gaussians_->setRotationLearningRate(rotationLearningRate());  // 旋转学习率

    // ========== 渲染：从当前关键帧视角渲染 Gaussian 模型 ==========
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,        // 关键帧（包含位姿、相机参数等）
        image_height,          // 渲染图像高度
        image_width,           // 渲染图像宽度
        gaussians_,            // Gaussian 模型
        pipe_params_,          // 流水线参数
        background_,           // 背景颜色
        override_color_        // 覆盖颜色（通常为空）
    );
    // ========== 解包渲染结果 ==========
    auto rendered_image = std::get<0>(render_pkg);           // 渲染图像（RGB）
    auto viewspace_point_tensor = std::get<1>(render_pkg);   // 视图空间点张量（用于密度化统计）
    auto visibility_filter = std::get<2>(render_pkg);         // 可见性过滤器（哪些点可见）
    auto radii = std::get<3>(render_pkg);                    // 2D 半径（用于剪枝）

    // ========== 去除去畸变导致的黑色边缘 ==========
    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;  // 用掩码屏蔽边缘区域

    // ========== 计算损失 ==========
    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);  // L1 损失（像素级差异）
    float lambda_dssim = lambdaDssim();  // 获取 SSIM 损失权重
    // 组合损失 = (1 - λ) * L1 + λ * (1 - SSIM)
    // SSIM 值越大越好（接近 1），所以用 (1 - SSIM) 作为损失
    auto photo_loss = (1.0 - lambda_dssim) * Ll1
                      + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));

    // 各向同性正则：仅对新加入点的前 N 次迭代生效，老点不再施加该约束
    auto scaling_log = gaussians_->scaling_;  // [N,3]
    auto isotropic_reg = torch::zeros({}, scaling_log.options());
    if (isotropic_scaling_reg_new_point_iters_ > 0) {
        const int cur_iter = getIteration();
        if (gaussians_->exist_since_iter_.defined() &&
            gaussians_->exist_since_iter_.numel() == scaling_log.size(0)) {
            auto age = cur_iter - gaussians_->exist_since_iter_;
            auto young_mask = age < isotropic_scaling_reg_new_point_iters_;
            auto young_idx = torch::nonzero(young_mask).squeeze(1);
            if (young_idx.defined() && young_idx.numel() > 0) {
                auto young_scaling_log = scaling_log.index({young_idx});
                auto young_scaling_log_mean = young_scaling_log.mean(/*dim=*/1, /*keepdim=*/true);
                isotropic_reg = torch::mean(torch::pow(young_scaling_log - young_scaling_log_mean, 2));
            }
        }
    }

    // 自适应权重：随训练进度与EMA loss下降而减小，后期保留最小比例避免完全失去形状约束
    if (isotropic_scaling_reg_ref_ema_loss_ < 0.0f) {
        float bootstrap = std::max(ema_loss_for_log_, 1e-6f);
        if (bootstrap <= 1e-6f) {
            bootstrap = Ll1.item().toFloat();
        }
        isotropic_scaling_reg_ref_ema_loss_ = std::max(bootstrap, 1e-6f);
    }

    float iter_ratio = 1.0f - static_cast<float>(getIteration()) /
                                static_cast<float>(std::max(1, opt_params_.iterations_));
    iter_ratio = std::max(0.0f, std::min(1.0f, iter_ratio));
    float progress_factor = std::pow(iter_ratio, std::max(0.0f, isotropic_scaling_reg_progress_power_));

    float loss_ratio = ema_loss_for_log_ / std::max(1e-6f, isotropic_scaling_reg_ref_ema_loss_);
    loss_ratio = std::max(0.0f, std::min(1.0f, loss_ratio));
    float loss_factor = std::pow(loss_ratio, std::max(0.0f, isotropic_scaling_reg_loss_power_));

    float min_ratio = std::max(0.0f, std::min(1.0f, isotropic_scaling_reg_min_ratio_));
    float adaptive_ratio = std::max(min_ratio, progress_factor * std::max(min_ratio, loss_factor));
    float effective_iso_weight = isotropic_scaling_reg_weight_ * adaptive_ratio;

    auto loss = photo_loss + effective_iso_weight * isotropic_reg;
    // ========== 反向传播：计算梯度 ==========
    loss.backward();

    // ========== 同步 CUDA 操作（确保所有 GPU 操作完成） ==========
    if (device_type_ == torch::kCUDA) {
        torch::cuda::synchronize();
    }

    {
        // ========== 禁用梯度计算（用于非优化操作） ==========
        torch::NoGradGuard no_grad;
        // ========== 更新指数移动平均损失（用于日志） ==========
        // EMA = 0.4 * 当前损失 + 0.6 * 之前 EMA（平滑损失曲线）
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        // ========== 定期记录关键帧渲染结果 ==========
        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // ========== 密度化：增加/删除 Gaussian 点 ==========
        // Densification
        if (getIteration() < opt_params_.densify_until_iter_) {
            // ========== 保护：确保统计张量长度与高斯数量一致 ==========
            // auto gaussian_count = gaussians_->getXYZ().size(0);
            // if (gaussians_->max_radii2D_.numel() != gaussian_count ||
            //     gaussians_->xyz_gradient_accum_.size(0) != gaussian_count ||
            //     gaussians_->denom_.size(0) != gaussian_count) {
            //     std::cerr << "[Gaussian Mapper] Warning: densification stats size mismatch. "
            //               << "xyz=" << gaussian_count
            //               << ", max_radii2D_=" << gaussians_->max_radii2D_.numel()
            //               << ", xyz_gradient_accum_=" << gaussians_->xyz_gradient_accum_.size(0)
            //               << ", denom_=" << gaussians_->denom_.size(0)
            //               << ". Resetting stats tensors." << std::endl;
            //     gaussians_->max_radii2D_ = torch::zeros(
            //         {gaussian_count}, torch::TensorOptions().device(device_type_));
            //     gaussians_->xyz_gradient_accum_ = torch::zeros(
            //         {gaussian_count, 1}, torch::TensorOptions().device(device_type_));
            //     gaussians_->denom_ = torch::zeros(
            //         {gaussian_count, 1}, torch::TensorOptions().device(device_type_));
            // }

            // ========== 跟踪图像空间中的最大半径（用于剪枝） ==========
            // Keep track of max radii in image-space for pruning
            // 更新可见点的最大 2D 半径（取当前值和历史最大值中的较大者）
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},  // 只更新可见点
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // ========== 添加密度化统计信息（用于决定分裂/克隆） ==========
            // 注释掉的代码：如果使用高斯金字塔训练且是最高层级，可能跳过统计
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            // ========== 执行密度化和剪枝 ==========
            // 条件：迭代次数 > 开始密度化的迭代 && 到达密度化间隔
            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                // 大小阈值：如果超过剪枝大点的迭代，则剪枝大于 20 像素的点
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                // 执行密度化（分裂大梯度点、克隆小梯度点）和剪枝（删除低不透明度点、大点）
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),      // 密度化梯度阈值
                    densify_min_opacity_,        // 最小不透明度阈值（低于此值被剪枝）
                    scene_->cameras_extent_,     // 场景范围（用于判断点是否过大）
                    size_threshold,              // 大小阈值（像素，超过此值被剪枝）
                    getIteration()               // 新生高斯年龄起点
                );
            }

            // ========== 重置不透明度 ==========
            // 条件：设置了重置间隔 && (到达重置间隔 || (白色背景 && 刚开始密度化))
            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();  // 重置所有点的不透明度（用于重新优化）
        }

        // ========== 全程安全剪枝（防止后期在远处长出超大高斯） ==========
        if (enable_safety_prune_ && safety_prune_interval_ > 0 &&
            getIteration() > opt_params_.densify_from_iter_ &&
            getIteration() % safety_prune_interval_ == 0) {
            auto xyz = gaussians_->getXYZ();
            const int total_pts = static_cast<int>(xyz.size(0));
            if (total_pts > 0) {
                auto center = xyz.mean(/*dim=*/0, /*keepdim=*/false);
                auto center_expand = center.unsqueeze(0).expand_as(xyz);
                auto dists = torch::norm(xyz - center_expand, 2, 1);

                const float scene_extent = std::max(scene_->cameras_extent_, 1e-6f);
                const float max_ws = std::max(1e-6f, safety_prune_max_world_scale_ratio_ * scene_extent);
                const float max_dist = std::max(1e-6f, safety_prune_max_center_dist_ratio_ * scene_extent);

                auto max_scale = std::get<0>(gaussians_->getScalingActivation().max(/*dim=*/1));
                auto big_ws = max_scale > max_ws;
                auto far_away = dists > max_dist;
                auto low_opacity = (gaussians_->getOpacityActivation() < densify_min_opacity_).squeeze();

                auto prune_mask = torch::logical_or(torch::logical_or(big_ws, far_away), low_opacity);
                int prune_num = prune_mask.sum().item<int>();

                // 避免一次性删光，至少保留少量高斯稳定训练
                if (prune_num > 0 && prune_num < total_pts) {
                    gaussians_->prunePoints(prune_mask);
                    std::cout << "[Gaussian Mapper] safety-prune removed " << prune_num
                              << " / " << total_pts
                              << " (max_ws=" << max_ws
                              << ", max_dist=" << max_dist << ")"
                              << std::endl;
                }
            }
        }

        // ========== 每迭代删除 20 个随机 Gaussian（正则化，防止过拟合） ==========
        // Delete 20 random Gaussians per iteration
        //gaussians_->deleteRandomGaussians(5);

        // ========== 计算迭代时间 ==========
        auto iter_end_timing = std::chrono::steady_clock::now();  // 记录结束时间
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();  // 计算耗时（毫秒）
        if (mapping_loop_ema_ms_ <= 0.0f) {
            mapping_loop_ema_ms_ = static_cast<float>(iter_time);
        } else {
            mapping_loop_ema_ms_ = 0.9f * mapping_loop_ema_ms_ + 0.1f * static_cast<float>(iter_time);
        }

        // ========== 日志和保存 ==========
        // Log and save
        // ========== 定期输出训练报告 ==========
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),          // 当前迭代次数
                opt_params_.iterations_, // 最大迭代次数
                Ll1,                     // L1 损失
                loss,                    // 总损失
                ema_loss_for_log_,       // EMA 损失
                loss_utils::l1_loss,     // L1 损失函数
                iter_time,               // 迭代耗时
                *gaussians_,             // Gaussian 模型
                *scene_,                 // 场景
                pipe_params_,            // 流水线参数
                background_              // 背景颜色
            );
        // ========== 定期记录所有关键帧和保存模型 ==========
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_  // 注释掉：回环闭合时也记录
            )
        {
            renderAndRecordAllKeyframes();  // 渲染并记录所有关键帧
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");  // 保存 PLY 文件
        }

        // ========== 重置回环闭合迭代标志 ==========
        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // ========== 优化器更新参数 ==========
        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();        // 执行优化器步骤（更新参数）
            gaussians_->optimizer_->zero_grad(true);  // 清零梯度（为下一次迭代准备）
        }

    }

    // 释放训练阶段渲染锁后，再尝试执行 PoseOpt 预算调度。
    }

    // ========== 滑动窗口主动位姿优化 ==========
    // 注意：必须放在 NoGradGuard 作用域外，否则会禁用位姿优化链路的自动求导
    if (enable_pose_optimization_ && pose_optimizer_ && pose_optimizer_->isReady()) {
        if (!pose_opt_trigger_on_new_keyframe_ &&
            getIteration() >= pose_opt_cfg_.min_iteration &&
            pose_opt_cfg_.call_interval > 0 &&
            getIteration() % pose_opt_cfg_.call_interval == 0) {
            enqueuePoseOptTask(pose_opt_cfg_.steps_per_call, "train_interval");
        }
        runPoseOptWithBudget(false);
    }
}

// ========== 检查是否已停止 ==========
bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);  // 加锁保护状态
    return this->stopped_;  // 返回停止标志
}

// ========== 发送停止信号 ==========
void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);  // 加锁保护状态
    this->stopped_ = going_to_stop;  // 设置停止标志
}

// ========== 检查是否满足初始建图条件 ==========
bool GaussianMapper::hasMetInitialMappingConditions()
{
    // 条件：
    //   1. 关键帧数量 >= 最小初始建图关键帧数
    //   2. 数据提供者已初始化
    if (data_provider_ && 
        data_provider_->isInitialized() &&
        data_provider_->getKeyFrameCount() >= min_num_initial_map_kfs_)
        return true;

    return false;
}

// ========== 检查是否满足增量建图条件 ==========
bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    // 条件：有新的关键帧
    if (data_provider_ && data_provider_->hasNewKeyFrame())
        return true;

    return false;
}

// ========== 处理地图操作：处理来自数据提供者的新关键帧 ==========
void GaussianMapper::combineMappingOperations()
{
    // 从数据提供者获取新关键帧
    if (!data_provider_) return;
    
    // 处理所有新关键帧
    // 注：这里简化了原始的ORB-SLAM3集成代码
    // 原代码包含了复杂的局部BA、回环闭合处理
    // 现在只需要获取新关键帧并添加到场景
}

// ========== 处理新关键帧：创建 GaussianKeyframe 并添加到场景 ==========
// 参数：kf - 关键帧元组（ID, 相机ID, 位姿, 图像, 是否回环, 辅助图像, 关键点像素, 关键点3D, 文件名）
void GaussianMapper::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,              // 关键帧 ID
                unsigned long/*CameraId*/,         // 相机 ID
                Sophus::SE3f/*pose*/,             // 位姿（世界到相机）
                cv::Mat/*image*/,                  // 主图像（RGB）
                bool/*isLoopClosure*/,             // 是否是回环关键帧
                cv::Mat/*auxiliaryImage*/,         // 辅助图像（深度图或右目）
                std::vector<float>,                // 关键点像素坐标
                std::vector<float>,                // 关键点局部 3D 坐标
                std::string> &kf)                  // 图像文件名
{
    // ========== 创建新的 Gaussian 关键帧 ==========
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());  // 使用关键帧 ID 和当前迭代
    pkf->zfar_ = z_far_;   // 设置远裁剪平面
    pkf->znear_ = z_near_; // 设置近裁剪平面
    // ========== 设置关键帧位姿 ==========
    // Pose
    auto& pose = std::get<2>(kf);  // 获取位姿
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),  // 旋转（四元数）
        pose.translation().cast<double>());      // 平移
    cv::Mat imgRGB_undistorted, imgAux_undistorted;  // 去畸变后的图像
    try {
        // ========== 设置相机参数 ==========
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));  // 根据相机 ID 获取相机
        pkf->setCameraParams(camera);  // 设置相机参数

/*         // ========== 处理主图像（左目或单目/RGB-D 的 RGB） ==========
        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);  // 获取主图像
        if (this->sensor_type_ == STEREO)
            // 双目模式：图像已校正，不需要去畸变
            imgRGB_undistorted = imgRGB;
        else
            // 单目/RGB-D 模式：需要去畸变
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // ========== 处理辅助图像（右目或深度图） ==========
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);  // 获取辅助图像
        if (this->sensor_type_ == RGBD)
            // RGB-D 模式：深度图需要去畸变
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            // 双目模式：右目图像已校正，不需要去畸变
            imgAux_undistorted = imgAux;
 */
        // ========== 设置关键帧图像数据 ==========
        cv::Mat imgRGB = std::get<3>(kf);
        if (imgRGB.empty()) {
            std::cerr << "[Gaussian Mapper] Warning: Keyframe image is empty after extraction. Skip this keyframe." << std::endl;
            return;
        }
        camera.undistortImage(imgRGB, imgRGB_undistorted);
        cv::Mat imgAux = std::get<5>(kf);
        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);  // 转换为 Torch 张量
        pkf->img_filename_ = std::get<8>(kf);  // 保存文件名
        // ========== 设置高斯金字塔参数 ==========
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;  // 各层级高度
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;   // 各层级宽度
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;  // 各层级使用次数
        pkf->img_undist_ = imgRGB_undistorted;
        pkf->img_auxiliary_undist_ = imgAux;
    }
    catch (std::out_of_range) {
        // 如果找不到对应的相机，抛出异常
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // ========== 先计算变换张量，供后续 PoseOpt 使用 ==========
    pkf->computeTransformTensors();  // 计算变换矩阵

    // ========== 从新关键帧获取密集点云以加速训练 ==========
    // Get dense point cloud from the new keyframe to accelerate training
/*     pkf->img_undist_ = imgRGB_undistorted;           // 保存去畸变后的主图像
    pkf->img_auxiliary_undist_ = imgAux_undistorted;  // 保存去畸变后的辅助图像
    pkf->kps_pixel_ = std::move(std::get<6>(kf));    // 移动赋值关键点像素坐标（避免拷贝）
    pkf->kps_point_local_ = std::move(std::get<7>(kf));  // 移动赋值关键点局部 3D 坐标 */
    // ========== 如果启用非活跃区域几何密度化，从关键帧提取点 ==========
/*     if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);  // 从关键帧的非活跃区域提取几何信息
 */
    // ========== 为训练准备多分辨率图像（高斯金字塔） ==========
    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA && opencv_cuda_available_) {
        // ========== GPU 模式：使用 CUDA 加速 ==========
        cv::cuda::GpuMat img_gpu;                    // CUDA GpuMat
        img_gpu.upload(pkf->img_undist_);            // 上传到 GPU
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);  // 调整数组大小
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;            // 缩放后的 GPU 图像
            // CUDA 图像缩放
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            // 转换为 Torch 张量
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        // ========== CPU 模式：使用 OpenCV CPU 函数 ==========
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);  // 调整数组大小
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;                      // 缩放后的 CPU 图像
            // OpenCV CPU 图像缩放
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            // 转换为 Torch 张量
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }

    // ========== 将关键帧推入滑动窗口位姿优化器（先微调，再上传场景） ==========
    if (enable_pose_optimization_ && pose_optimizer_) {
        pose_optimizer_->push(pkf);

        if (pose_opt_trigger_on_new_keyframe_ &&
            getIteration() >= pose_opt_cfg_.min_iteration) {
            enqueuePoseOptTask(std::max(1, pose_opt_steps_on_new_keyframe_), "new_keyframe");
        }
    }

    // ========== 位姿微调后再将关键帧添加到场景 ==========
    scene_->addKeyframe(pkf, &kfid_shuffled_);  // 添加到场景（可能打乱顺序）

    // ========== 设置新关键帧的使用次数并添加到训练滑动窗口 ==========
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());  // 增加使用次数
}

// ========== 生成关键帧 ID 的随机打乱序列 ==========
// 用于随机采样关键帧进行训练（避免总是按顺序选择）
void GaussianMapper::generateKfidRandomShuffle()
{
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    // ========== 检查是否有关键帧 ==========
    if (scene_->keyframes().empty())
        return;  // 如果没有关键帧，直接返回

    // ========== 生成打乱序列 ==========
    std::size_t nkfs = scene_->keyframes().size();  // 获取关键帧数量
    kfid_shuffle_.resize(nkfs);                     // 调整打乱数组大小
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);  // 填充 0, 1, 2, ..., nkfs-1
    std::mt19937 g(rd_());                          // 创建随机数生成器（使用随机设备作为种子）
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);  // 打乱序列

    kfid_shuffled_ = true;  // 标记已打乱
}

// ========== 从滑动窗口随机选择一个关键帧用于训练 ==========
// 使用打乱的序列循环选择，确保所有关键帧都被使用
std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();  // 计时（已注释）
    // ========== 检查是否有关键帧 ==========
    if (scene_->keyframes().empty())
        return nullptr;  // 没有关键帧，返回空指针

    // ========== 如果未打乱，则生成打乱序列 ==========
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;  // 选中的关键帧
    int random_cam_idx;  // 随机相机索引

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;  // 记录起始索引
        do {
            // ========== 移动到下一个打乱的索引 ==========
            // Next shuffled idx
            ++kfid_shuffle_idx_;  // 增加索引
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;  // 如果超出范围，回到开头（循环）
            // ========== 如果所有关键帧都不可用，给所有关键帧增加一次使用次数 ==========
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                // 如果回到起始位置（说明遍历了一圈），给所有关键帧增加使用次数
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, old_keyframe_times_of_use_);
            // ========== 根据打乱序列获取关键帧 ==========
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];  // 获取打乱后的索引
            auto random_cam_it = scene_->keyframes().begin();   // 获取关键帧迭代器
            // 移动到对应的关键帧（因为 map 是无序的，需要遍历）
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;  // 获取关键帧指针
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);  // 循环直到找到有剩余使用次数的关键帧
    }

    // ========== 统计使用次数 ==========
    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;  // 获取关键帧 ID
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;  // 如果未记录过，初始化为 1
    else
        ++kfs_used_times_[viewpoint_fid];     // 否则增加计数
    
    // ========== 处理使用次数 ==========
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);  // 减少剩余使用次数

// auto t2 = std::chrono::steady_clock::now();  // 计时结束（已注释）
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();  // 计算耗时
// std::cout<<t21 <<" ns"<<std::endl;  // 输出耗时
    return viewpoint_cam;  // 返回选中的关键帧
}

// ========== 随机选择一个关键帧（不使用滑动窗口，完全随机） ==========
std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    // ========== 检查是否有关键帧 ==========
    if (scene_->keyframes().empty())
        return nullptr;  // 没有关键帧，返回空指针

    // ========== 完全随机选择 ==========
    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());  // 获取关键帧数量
    // 生成随机索引（使用 C 标准库的 rand，注意这不是均匀分布）
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();  // 获取迭代器
    // 移动到随机索引对应的关键帧
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;  // 获取关键帧指针

    // ========== 统计使用次数 ==========
    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;  // 获取关键帧 ID
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;  // 如果未记录过，初始化为 1
    else
        ++kfs_used_times_[viewpoint_fid];     // 否则增加计数

    return viewpoint_cam;  // 返回选中的关键帧
}

// ========== 增加关键帧使用次数 ==========
void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,  // 关键帧指针
    int times)                               // 增加的使用次数
{
    pkf->remaining_times_of_use_ += times;  // 增加剩余使用次数
}

void GaussianMapper::boostOldFrameUsageEveryNIterations(int period_iters)
{
    if (period_iters <= 0 || getIteration() % period_iters != 0) {
        return;
    }
    if (scene_->keyframes().empty() || old_keyframe_times_of_use_ <= 0) {
        return;
    }

    std::vector<std::shared_ptr<GaussianKeyframe>> old_frame_pool;
    old_frame_pool.reserve(scene_->keyframes().size());
    const int cur_iter = getIteration();
    for (const auto& kfit : scene_->keyframes()) {
        const auto& cam = kfit.second;
        if (!cam) {
            continue;
        }
        if (!cam->original_image_.defined() || cam->original_image_.numel() == 0) {
            continue;
        }
        // 将创建超过一个周期的关键帧视为老帧。
        if (cur_iter - cam->creation_iter_ >= period_iters) {
            old_frame_pool.push_back(cam);
        }
    }

    if (old_frame_pool.empty()) {
        return;
    }

    std::mt19937 rng(rd_());
    std::shuffle(old_frame_pool.begin(), old_frame_pool.end(), rng);
    const int sample_n = std::min(
        std::max(1, multiview_old_frame_samples_),
        static_cast<int>(old_frame_pool.size()));
    for (int i = 0; i < sample_n; ++i) {
        increaseKeyframeTimesOfUse(old_frame_pool[i], old_keyframe_times_of_use_);
    }
}

bool GaussianMapper::seesEnoughOldGaussiansByProjection(
    const std::shared_ptr<GaussianKeyframe>& cam,
    const torch::Tensor& old_xyz_cpu,
    int min_visible_old_pts) const
{
    if (!cam || !cam->set_pose_ || cam->intr_.size() < 4 || min_visible_old_pts <= 0) {
        return false;
    }
    if (!old_xyz_cpu.defined() || old_xyz_cpu.numel() == 0 || old_xyz_cpu.size(1) != 3) {
        return false;
    }

    const float fx = cam->intr_[0];
    const float fy = cam->intr_[1];
    const float cx = cam->intr_[2];
    const float cy = cam->intr_[3];
    const float width = static_cast<float>(std::max(1, cam->image_width_));
    const float height = static_cast<float>(std::max(1, cam->image_height_));

    const Sophus::SE3f Tcw = cam->getPosef();
    const Eigen::Matrix3f& Rcw = Tcw.rotationMatrix();
    const Eigen::Vector3f& tcw = Tcw.translation();

    auto xyz = old_xyz_cpu.contiguous();
    auto xyz_acc = xyz.accessor<float, 2>();
    int visible_cnt = 0;
    const int64_t n_pts = xyz.size(0);

    for (int64_t i = 0; i < n_pts; ++i) {
        Eigen::Vector3f pw(xyz_acc[i][0], xyz_acc[i][1], xyz_acc[i][2]);
        Eigen::Vector3f pc = Rcw * pw + tcw;
        if (pc.z() <= 1e-4f) {
            continue;
        }

        const float inv_z = 1.0f / pc.z();
        const float u = fx * pc.x() * inv_z + cx;
        const float v = fy * pc.y() * inv_z + cy;
        if (u >= 0.0f && u < width && v >= 0.0f && v < height) {
            ++visible_cnt;
            if (visible_cnt >= min_visible_old_pts) {
                return true;
            }
        }
    }

    return false;
}

void GaussianMapper::boostHistoricalFrameUsageOnNewKeyframes(const std::vector<unsigned long>& new_kfids)
{
    if (multiview_old_frame_samples_ <= 0 || old_keyframe_times_of_use_ <= 0 || new_kfids.empty()) {
        return;
    }
    if (!gaussians_ || scene_->keyframes().empty()) {
        return;
    }

    torch::Tensor xyz_all;
    torch::Tensor exist_since;
    {
        torch::NoGradGuard no_grad;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        xyz_all = gaussians_->getXYZ().detach();
        exist_since = gaussians_->exist_since_iter_.detach();
    }

    if (!xyz_all.defined() || xyz_all.numel() == 0 || xyz_all.size(1) != 3) {
        return;
    }
    if (!exist_since.defined() || exist_since.numel() != xyz_all.size(0)) {
        return;
    }

    const int cur_iter = getIteration();
    auto age = cur_iter - exist_since;
    auto old_mask = age >= multiview_old_gaussian_min_age_iters_;
    auto old_idx = torch::nonzero(old_mask).squeeze(1);
    if (!old_idx.defined() || old_idx.numel() == 0) {
        return;
    }

    constexpr int64_t kMaxProjectedOldPoints = 30000;
    if (old_idx.size(0) > kMaxProjectedOldPoints) {
        auto rand_perm = torch::randperm(
            old_idx.size(0),
            torch::TensorOptions().dtype(torch::kLong).device(old_idx.device()));
        old_idx = old_idx.index({rand_perm.index({torch::indexing::Slice(0, kMaxProjectedOldPoints)})});
    }

    auto old_xyz_cpu = xyz_all.index({old_idx}).to(torch::kCPU, torch::kFloat32).contiguous();
    const int old_pts_num = static_cast<int>(old_xyz_cpu.size(0));
    if (old_pts_num <= 0) {
        return;
    }

    int min_visible_old_pts = std::max(64, static_cast<int>(static_cast<float>(old_pts_num) * 0.005f));
    min_visible_old_pts = std::min(min_visible_old_pts, old_pts_num);

    std::unordered_set<unsigned long> new_kfid_set(new_kfids.begin(), new_kfids.end());
    std::vector<std::shared_ptr<GaussianKeyframe>> candidate_frames;
    candidate_frames.reserve(scene_->keyframes().size());
    for (const auto& kfit : scene_->keyframes()) {
        const auto& cam = kfit.second;
        if (!cam) {
            continue;
        }
        if (new_kfid_set.find(static_cast<unsigned long>(cam->fid_)) != new_kfid_set.end()) {
            continue;
        }
        if (!cam->original_image_.defined() || cam->original_image_.numel() == 0) {
            continue;
        }
        candidate_frames.push_back(cam);
    }

    if (candidate_frames.empty()) {
        return;
    }

    std::mt19937 rng(rd_());
    std::shuffle(candidate_frames.begin(), candidate_frames.end(), rng);

    int selected_num = 0;
    const int target_num = std::min(multiview_old_frame_samples_, static_cast<int>(candidate_frames.size()));
    for (const auto& cam : candidate_frames) {
        if (seesEnoughOldGaussiansByProjection(cam, old_xyz_cpu, min_visible_old_pts)) {
            increaseKeyframeTimesOfUse(cam, old_keyframe_times_of_use_);
            ++selected_num;
            // 一旦找到足够历史帧，立即退出，避免无效遍历。
            if (selected_num >= target_num) {
                break;
            }
        }
    }
}

// ========== 剔除关键帧：删除 SLAM 中已删除的关键帧 ==========
void GaussianMapper::cullKeyframes()
{
    // VINS 是滑动窗口，旧帧会被移出窗口
    // 原意是建图，保留所有历史关键帧用于构建完整的地图
    // 如果需要剔除关键帧以减少内存占用，可以根据使用次数或时间戳进行筛选
    // 目前保持为空，即保留所有历史帧
}

// ========== 从关键帧的非活跃区域提取几何信息进行密度化 ==========
// 用于单目模式：从特征点周围的非活跃区域提取密集点云
// 用于双目/RGB-D 模式：从深度图/视差图提取密集点云
void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)  // 关键帧指针
{
// auto start_timing = std::chrono::steady_clock::now();  // 计时开始（已注释）
    torch::NoGradGuard no_grad;  // 禁用梯度（提取点不需要梯度）

    if (!opencv_cuda_available_) {
        return;
    }

    // ========== 计算相机到世界的变换（用于将局部坐标转换为世界坐标） ==========
    Sophus::SE3f Twc = pkf->getPosef().inverse();  // Tcw 的逆 = Twc

    // ========== 根据传感器类型提取点云 ==========
    switch (this->sensor_type_)
    {
    // ========== 单目模式：从特征点周围提取点 ==========
    case MONOCULAR:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    // ========== 双目模式：从视差图提取点 ==========
    case STEREO:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        // ========== 上传图像到 GPU ==========
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;   // RGB 图像（GPU）
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu; // 灰度图像（GPU）

        rgb_left_gpu.upload(pkf->img_undist_);           // 上传左目图像
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_); // 上传右目图像

        // ========== 转换为灰度图 ==========
        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);   // 左目转灰度
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY); // 右目转灰度

        // ========== 转换为 8 位无符号整数（SGM 算法需要） ==========
        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);   // 左目：浮点 → 8 位（0-255）
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0); // 右目：浮点 → 8 位

        // ========== 计算视差图 ==========
        // Compute disparity
        cv::cuda::GpuMat cv_disp;  // 视差图（GPU）
        // 使用 Semi-Global Matching (SGM) 算法计算视差
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        // 转换为浮点型（SGM 输出是 16 位整数，需要除以 16）
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // ========== 从视差图重投影得到 3D 点 ==========
        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;  // 3D 点（GPU）
        // 使用 Q 矩阵将视差图转换为 3D 点（局部坐标系）
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // ========== 从 CUDA GpuMat 转换为 Torch 张量 ==========
        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);  // 视差图
        disp = disp.flatten(0, 1).contiguous();  // 展平为 1D（H*W）
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);  // 3D 点
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();  // 重排维度并展平（H*W, 3）
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);  // 颜色
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();  // 重排维度并展平（H*W, 3）
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    // ========== RGB-D 模式：从深度图提取点 ==========
    case RGBD:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        // ========== 上传图像到 GPU ==========
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;  // RGB 和深度图（GPU）
        img_rgb_gpu.upload(pkf->img_undist_);          // 上传 RGB 图像
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);  // 上传深度图

        // ========== 从 CUDA GpuMat 转换为 Torch 张量 ==========
        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);  // RGB 图像
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();  // 重排并展平（H*W, 3）
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);  // 深度图
        depth = depth.flatten(0, 1).contiguous();  // 展平为 1D（H*W）

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // ========== 从深度图重投影得到 3D 点 ==========
        // Reproject to get 3D points
        torch::Tensor points3D_valid;  // 有效的 3D 点
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);  // 获取相机
        // ========== 根据相机模型类型进行重投影 ==========
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            // ========== 针孔模型：使用内参重投影 ==========
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);  // 重投影深度图到 3D
        }
        break;
        case Camera::FISHEYE:
        {
            // ========== 鱼眼模型：暂不支持 ==========
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            // ========== 无效的相机模型 ==========
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        // ========== 只保留有效的点 ==========
        points3D_valid = points3D_valid.index({point_valid_flags});  // 根据有效标志筛选

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // ========== 标记该关键帧已完成非活跃区域密度化 ==========
    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;  // 增加缓存计数

    // ========== 如果达到最大缓存数量，将点添加到模型 ==========
    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;  // 重置缓存计数
        // ========== 将新点添加到模型 ==========
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);  // 加锁保护模型
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());  // 增加点云
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

// ========== 记录关键帧渲染结果（用于评估和可视化） ==========
void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,      // 渲染图像（Torch 张量）
        torch::Tensor &ground_truth,  // 真实图像（Torch 张量）
        unsigned long kfid,           // 关键帧 ID
        std::filesystem::path result_img_dir,   // 渲染图像保存目录
        std::filesystem::path result_gt_dir,     // 真实图像保存目录
        std::filesystem::path result_loss_dir,   // 损失图像保存目录
        std::string name_suffix)      // 文件名后缀
{
    // ========== 记录渲染图像 ==========
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);  // Torch → OpenCV
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);  // RGB → BGR（OpenCV 格式）
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);  // 浮点 [0,1] → 整数 [0,255]
        // 保存图像：迭代次数_关键帧ID_后缀.jpg
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    // ========== 记录真实图像 ==========
    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);  // Torch → OpenCV
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);  // RGB → BGR
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);  // 浮点 → 整数
        // 保存图像：迭代次数_关键帧ID_后缀_gt.jpg
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    // ========== 记录损失图像（渲染与真实的差异） ==========
    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);  // 计算绝对差异
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);  // Torch → OpenCV
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);  // RGB → BGR
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);  // 浮点 → 整数
        // 保存图像：迭代次数_关键帧ID_后缀_loss.jpg
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

// ========== 从指定相机位姿渲染图像 ==========
cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,  // 相机位姿（世界到相机）
    const int width,           // 渲染图像宽度
    const int height,          // 渲染图像高度
    const bool main_vision)    // 是否主视角（影响使用的掩码）
{
    // ========== 检查是否可以渲染 ==========
    if (!initial_mapped_ || getIteration() <= 0)
        // 如果未完成初始建图或迭代次数 <= 0，返回黑色图像
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    
    // ========== 创建临时关键帧用于渲染 ==========
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;   // 设置远裁剪平面
    pkf->znear_ = z_near_; // 设置近裁剪平面
    // ========== 设置位姿 ==========
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),  // 旋转
        Tcw.translation().cast<double>());      // 平移
    try {
        // ========== 设置相机参数 ==========
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);  // 获取查看器相机
        pkf->setCameraParams(camera);  // 设置相机参数
        // ========== 计算变换矩阵 ==========
        // Transformations
        pkf->computeTransformTensors();  // 计算世界到视图、投影矩阵等
    }
    catch (std::out_of_range) {
        // 如果找不到相机，抛出异常
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    // ========== 渲染 ==========
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;  // 渲染结果包
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);  // 加锁保护模型
        // ========== 执行渲染 ==========
        // Render
        render_pkg = GaussianRenderer::render(
            pkf,            // 关键帧
            height,         // 图像高度
            width,          // 图像宽度
            gaussians_,     // Gaussian 模型
            pipe_params_,   // 流水线参数
            background_,    // 背景颜色
            override_color_ // 覆盖颜色
        );
    }

    // ========== 应用掩码去除边缘 ==========
    // Result
    torch::Tensor masked_image;  // 掩码后的图像
    if (main_vision)
        // 主视角：使用主视角掩码
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        // 子视角：使用子视角掩码
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    // ========== 转换为 OpenCV Mat 并返回 ==========
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);  // Torch → OpenCV
}

// ========== 渲染并记录关键帧（计算评估指标） ==========
void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,  // 关键帧指针
    float &dssim,      // 输出：SSIM 距离（1 - SSIM）
    float &psnr,       // 输出：PSNR
    float &psnr_gs,    // 输出：Gaussian Splatting 风格的 PSNR
    double &render_time,  // 输出：渲染时间（毫秒）
    std::filesystem::path result_img_dir,   // 渲染图像保存目录
    std::filesystem::path result_gt_dir,     // 真实图像保存目录
    std::filesystem::path result_loss_dir,   // 损失图像保存目录
    std::string name_suffix)  // 文件名后缀
{
    // ========== 开始计时 ==========
    auto start_timing = std::chrono::steady_clock::now();
    // ========== 渲染关键帧 ==========
    auto render_pkg = GaussianRenderer::render(
        pkf,                    // 关键帧
        pkf->image_height_,     // 图像高度
        pkf->image_width_,      // 图像宽度
        gaussians_,             // Gaussian 模型
        pipe_params_,           // 流水线参数
        background_,            // 背景颜色
        override_color_         // 覆盖颜色
    );
    auto rendered_image = std::get<0>(render_pkg);  // 渲染图像
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];  // 应用掩码
    torch::cuda::synchronize();  // 同步 CUDA 操作
    // ========== 计算渲染时间 ==========
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;  // 纳秒 → 毫秒
    auto gt_image = pkf->original_image_;  // 获取真实图像

    // ========== 计算评估指标 ==========
    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();  // SSIM 距离
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();  // PSNR
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();  // GS 风格 PSNR

    // ========== 记录渲染结果 ==========
    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

// ========== 保存 PLY 文件（Gaussian 模型和关键帧信息） ==========
void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)  // 检查并创建目录
    keyframesToJson(result_dir);    // 保存关键帧信息为 JSON
    saveModelParams(result_dir);    // 保存模型参数

    // ========== 创建点云子目录 ==========
    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // ========== 创建迭代子目录 ==========
    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // ========== 保存 Gaussian 模型和输入点云 ==========
    gaussians_->savePly(ply_dir / "point_cloud.ply");        // 保存优化后的 Gaussian 点云
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");  // 保存输入的稀疏点云
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

// ========== 获取当前迭代次数 ==========
int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);  // 加锁保护状态
    return iteration_;  // 返回迭代次数
}
// ========== 获取EMA损失 ==========
float GaussianMapper::getEmaLossForLog()
{
    return ema_loss_for_log_;
}

GaussianMapper::PoseOptDebugInfo GaussianMapper::getPoseOptDebugInfo()
{
    PoseOptDebugInfo info;
    info.enabled = enable_pose_optimization_;
    info.has_optimizer = static_cast<bool>(pose_optimizer_);
    info.last_skipped_by_load = (pose_opt_dbg_last_skipped_by_load_.load(std::memory_order_relaxed) != 0);
    info.last_try_lock_failed = (pose_opt_dbg_last_try_lock_failed_.load(std::memory_order_relaxed) != 0);
    info.last_status_code = pose_opt_dbg_last_status_code_.load(std::memory_order_relaxed);
    info.queue_size = pose_opt_dbg_queue_size_.load(std::memory_order_relaxed);
    info.queue_limit = static_cast<int>(pose_opt_task_queue_limit_);
    info.last_run_steps = pose_opt_dbg_last_run_steps_.load(std::memory_order_relaxed);
    info.last_run_chunks = pose_opt_dbg_last_run_chunks_.load(std::memory_order_relaxed);
    info.last_run_elapsed_ms = pose_opt_dbg_last_run_elapsed_ms_.load(std::memory_order_relaxed);
    info.total_optimize_calls = pose_opt_dbg_total_optimize_calls_.load(std::memory_order_relaxed);
    info.last_iteration = pose_opt_dbg_last_iteration_.load(std::memory_order_relaxed);
    info.last_batch_ratio = pose_opt_dbg_last_batch_ratio_.load(std::memory_order_relaxed);
    info.mapping_loop_ema_ms = mapping_loop_ema_ms_;
    return info;
}
// ========== 增加迭代次数 ==========
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);  // 加锁保护状态
    iteration_ += inc;  // 增加迭代次数（可以是负数，用于回退）
}

// ========== 参数获取函数（只读，线程安全） ==========
float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    return opt_params_.position_lr_init_;  // 返回位置学习率初始值
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    return opt_params_.feature_lr_;  // 返回特征学习率
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    return opt_params_.opacity_lr_;  // 返回不透明度学习率
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    return opt_params_.scaling_lr_;  // 返回缩放学习率
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    return opt_params_.rotation_lr_;  // 返回旋转学习率
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

// ========== 参数设置函数（可写，线程安全） ==========
void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    opt_params_.position_lr_init_ = lr;  // 设置位置学习率初始值
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    opt_params_.feature_lr_ = lr;  // 设置特征学习率
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    opt_params_.opacity_lr_ = lr;  // 设置不透明度学习率
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    opt_params_.scaling_lr_ = lr;  // 设置缩放学习率
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);  // 加锁保护设置
    opt_params_.rotation_lr_ = lr;  // 设置旋转学习率
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}

void GaussianMapper::requestTailOptimization(const bool keep_training)
{
    {
        std::unique_lock<std::mutex> lock_status(mutex_status_);
        tail_optimization_requested_ = true;
    }
    {
        std::unique_lock<std::mutex> lock(mutex_settings_);
        keep_training_ = keep_training;
    }
}

bool GaussianMapper::isTailOptimizationRequested()
{
    std::unique_lock<std::mutex> lock_status(mutex_status_);
    return tail_optimization_requested_;
}

void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

// ========== 从 PLY 文件加载预训练的 Gaussian 模型 ==========
void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    // ========== 加载 Gaussian 模型 ==========
    this->gaussians_->loadPly(ply_path);  // 从 PLY 文件加载模型

    // ========== 加载相机参数（如果提供） ==========
    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        // ========== 打开相机配置文件 ==========
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        // ========== 创建相机对象 ==========
        Camera camera;
        camera.camera_id_ = 0;  // 设置相机 ID
        camera.width_ = camera_file["Camera.w"].operator int();   // 读取宽度
        camera.height_ = camera_file["Camera.h"].operator int();   // 读取高度

        // ========== 读取相机类型 ==========
        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);  // 设置为针孔模型

            // ========== 读取相机内参 ==========
            float fx = camera_file["Camera.fx"].operator float();  // 焦距 x
            float fy = camera_file["Camera.fy"].operator float();  // 焦距 y
            float cx = camera_file["Camera.cx"].operator float();  // 主点 x
            float cy = camera_file["Camera.cy"].operator float();  // 主点 y

            // ========== 读取畸变系数 ==========
            float k1 = camera_file["Camera.k1"].operator float();  // 径向畸变系数 1
            float k2 = camera_file["Camera.k2"].operator float();  // 径向畸变系数 2
            float p1 = camera_file["Camera.p1"].operator float();  // 切向畸变系数 1
            float p2 = camera_file["Camera.p2"].operator float();  // 切向畸变系数 2
            float k3 = camera_file["Camera.k3"].operator float();  // 径向畸变系数 3

            // ========== 构建相机内参矩阵 ==========
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,    // 第一行
                        0.f, fy, cy,   // 第二行
                        0.f, 0.f, 1.f  // 第三行
            );

            // ========== 设置相机参数 ==========
            camera.params_[0] = fx;  // 焦距 x
            camera.params_[1] = fy;  // 焦距 y
            camera.params_[2] = cx;  // 主点 x
            camera.params_[3] = cy;  // 主点 y

            // ========== 设置畸变系数 ==========
            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};  // 畸变系数向量
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());  // 转换为 OpenCV Mat
            // ========== 初始化去畸变映射和掩码 ==========
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            // ========== 将掩码转换为 Torch 张量 ==========
            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            // ========== 创建主视角掩码 ==========
            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;  // 计算主视角高度
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;     // 计算主视角宽度
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));  // 缩放掩码
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);  // 转换为张量

        }
        else {
            // ========== 不支持的相机模型 ==========
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        // ========== 设置查看器相机 ID ==========
        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;  // 设置查看器相机 ID
            viewer_camera_id_set_ = true;            // 标记已设置
        }
        // ========== 将相机添加到场景 ==========
        this->scene_->addCamera(camera);
    }

    // ========== 准备就绪 ==========
    // Ready
    this->initial_mapped_ = true;  // 标记初始建图已完成
    increaseIteration();            // 增加迭代次数（表示已加载模型）
}