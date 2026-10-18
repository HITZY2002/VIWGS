/**
 * 数据提供者接口
 * 用于解耦GaussianMapper和VinsSystem
 * GaussianMapper通过此接口获取关键帧数据，不再直接依赖VinsSystem
 */

#ifndef GAUSSIAN_DATA_PROVIDER_H
#define GAUSSIAN_DATA_PROVIDER_H

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace gaussian_mapping {

// ==================================================================================
//  关键帧数据结构
// ==================================================================================
struct KeyFrameData
{
    double timestamp;                      // 时间戳
    Sophus::SE3f pose;                     // 相机位姿 (world to camera, T_cw)
    bool depth_fusion_update;              // true: 仅深度补点增量（不应覆盖位姿）
    cv::Mat image;                         // 关键帧图像 (单目或左目)
    cv::Mat depth_image;                   // 与关键帧对齐的深度/逆深度图（可选，CV_32FC1）
    std::vector<Eigen::Vector3f> points;   // 世界坐标系下的稀疏3D点
    std::vector<Eigen::Vector3f> colors;   // 与points对应的颜色 (RGB, 0-1)
    
    // 可选：相机内参
    float fx, fy, cx, cy;
    int width, height;
    
    KeyFrameData() 
        : timestamp(0.0), depth_fusion_update(false), fx(0), fy(0), cx(0), cy(0), width(0), height(0) {}
    
        KeyFrameData(double t, const Sophus::SE3f& p, const cv::Mat& img)
                : timestamp(t), pose(p), image(img.clone()),
                    depth_fusion_update(false), fx(0), fy(0), cx(0), cy(0), width(img.cols), height(img.rows) {}
};

// ==================================================================================
//  数据提供者抽象接口
// ==================================================================================
class GaussianDataProvider
{
public:
    virtual ~GaussianDataProvider() = default;
    
    // ========== 关键帧数据访问 ==========
    
    /**
     * 获取所有关键帧数量
     */
    virtual size_t getKeyFrameCount() const = 0;
    
    /**
     * 获取指定索引的关键帧
     * @param index 关键帧索引 (0 到 getKeyFrameCount()-1)
     * @return 关键帧数据指针，若索引无效返回nullptr
     */
    virtual std::shared_ptr<KeyFrameData> getKeyFrame(size_t index) const = 0;
    
    /**
     * 获取最新的关键帧
     * @return 最新关键帧数据指针，若无关键帧返回nullptr
     */
    virtual std::shared_ptr<KeyFrameData> getLatestKeyFrame() const = 0;
    
    /**
     * 获取所有关键帧
     * @return 关键帧数据向量
     */
    virtual std::vector<std::shared_ptr<KeyFrameData>> getAllKeyFrames() const = 0;
    
    /**
     * 获取新增的关键帧（自上次调用以来）
     * @return 新增关键帧数据向量
     */
    virtual std::vector<std::shared_ptr<KeyFrameData>> getNewKeyFrames() = 0;
    
    // ========== 系统状态查询 ==========
    
    /**
     * 检查SLAM系统是否已初始化
     */
    virtual bool isInitialized() const = 0;
    
    /**
     * 检查SLAM系统是否已结束
     */
    virtual bool isFinished() const = 0;
    
    /**
     * 检查是否有新的关键帧到达
     */
    virtual bool hasNewKeyFrame() const = 0;
    
    // ========== 相机内参访问 ==========
    
    /**
     * 获取相机内参
     * @param fx, fy, cx, cy 内参输出
     * @return 是否成功获取
     */
    virtual bool getCameraIntrinsics(float& fx, float& fy, float& cx, float& cy) const = 0;
    
    /**
     * 获取图像分辨率
     * @param width, height 分辨率输出
     * @return 是否成功获取
     */
    virtual bool getImageSize(int& width, int& height) const = 0;
};

} // namespace gaussian_mapping

#endif // GAUSSIAN_DATA_PROVIDER_H
