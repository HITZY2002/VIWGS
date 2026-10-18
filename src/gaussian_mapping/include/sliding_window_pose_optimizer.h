#pragma once
/*
 * SlidingWindowPoseOptimizer
 * ──────────────────────────
 * 在 3DGS 侧对最近 N 帧的位姿做光度一致性优化（主动位姿优化）。
 *
 * 实现原理
 * ─────────
 * CUDA 光栅化器的 backward 不计算 viewmatrix 的梯度，因此位姿优化
 * 通过以下可微路径实现：
 *
 *   delta_xi [6]  (requires_grad=true, 初始为 0)
 *     │
 *     ▼  Rodrigues(delta_xi[0:3]) × R_orig  →  R_opt [3,3]
 *        R_delta @ t_orig + delta_xi[3:6]   →  t_opt [3]
 *     │
 *     ▼  means_cam = xyz_world @ R_opt.T + t_opt  (可微 matmul)
 *     │
 *     ▼  GaussianRenderer::renderWithDiffPose(...)  [3,H,W]
 *     │
 *     ▼  L1 光度损失 w.r.t. gt_image
 *     │
 *     ▼  backward → dL/d_delta_xi
 *     │
 *     ▼  Adam step → 更新 delta_xi
 *
 * 每隔 call_interval 个训练迭代调用一次 optimize()，
 * 优化完成后将 delta 吸收回关键帧 Eigen 位姿，并重置 delta 为 0。
 */

#include <deque>
#include <unordered_map>
#include <memory>
#include <vector>
#include <unordered_set>

#include <torch/torch.h>
#include <Eigen/Geometry>

#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_parameters.h"

// ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowPoseOptimizer
{
public:
    // ─── 配置参数 ────────────────────────────────────────────────────────────
    struct Config {
        int   window_size    = 50;     ///< 触发优化的窗口帧数（满 N 帧触发）
        int   slide_size     = 25;     ///< 每次优化后丢弃最前面的帧数
        int   anchor_size    = 5;      ///< 窗口前 anchor_size 帧为死锚点（不优化）
        float batch_ratio    = 0.20f;  ///< 每步随机采样参与优化的关键帧比例（相对可优化帧）
        int   steps_per_call = 10;     ///< 每次 optimize() 执行的 Adam 步数
        float lr             = 1e-3f;  ///< delta_xi 的 Adam 学习率
        int   call_interval  = 100;    ///< 每隔多少训练迭代调用一次 optimize()
        int   min_iteration  = 1000;   ///< 训练达到该迭代后才开始位姿优化
        float lambda_ssim    = 0.10f;  ///< 位姿优化中的 SSIM 权重
        float rot_reg_weight = 1e-3f;  ///< 旋转增量 L2 正则权重
        float trans_reg_weight = 5e-3f;///< 平移增量 L2 正则权重
        float max_grad_norm  = 0.05f;  ///< delta_xi 梯度范数裁剪阈值（<=0 关闭）
        float absorb_rot_min_deg = 0.02f;   ///< absorb 写回最小旋转变化阈值（度）
        float absorb_trans_min = 1e-5f;     ///< absorb 写回最小平移变化阈值（米）
        int   max_optimized_fids_per_absorb = 8; ///< absorb 每轮最多写回关键帧数量（<=0 不限）
        int   cache_downsample = 2;              ///< PoseOpt 缓存降采样倍数（1/2/4）
        bool  use_lowres_cache = true;           ///< PoseOpt 默认使用低分辨率缓存计算 loss
    };

    struct ChunkInfo {
        std::vector<unsigned long> frame_fids;   ///< 当前 chunk 帧 ID（按窗口顺序）
        std::vector<unsigned long> anchor_fids;  ///< anchor 帧 ID（前 anchor_size）
        int active_count = 0;                    ///< chunk 实际帧数
        int anchor_count = 0;                    ///< anchor 帧数
    };

    // ─── 构造 ────────────────────────────────────────────────────────────────
    explicit SlidingWindowPoseOptimizer(Config cfg, torch::DeviceType device);

    // ─── 接口 ────────────────────────────────────────────────────────────────

    /**
     * @brief 将关键帧加入滑动窗口。
     *
     * 从 pkf 中提取当前 Eigen 位姿作为 R_orig/t_orig，
     * 初始化对应的 delta_xi = 0，并重建 Adam 优化器。
     * 当窗口超出 window_size 时，弹出最旧帧（delta 已在上次 optimize() 中吸收）。
     *
     * @param pkf  刚处理完的关键帧（pose 已更新）
     */
    void push(std::shared_ptr<GaussianKeyframe> pkf);

    /**
     * @brief 对窗口内所有关键帧执行 steps_per_call 步位姿优化，
     *        然后将 delta 吸收回 Eigen 位姿，重置 delta = 0，重建优化器。
     *
     * @param gaussians  3D Gaussian 模型（内部参数只读 detach）
     * @param pipe       渲染流水线参数
     * @param bg_color   背景颜色张量
     */
    void optimize(
        std::shared_ptr<GaussianModel>  gaussians,
        GaussianPipelineParams&         pipe,
        torch::Tensor&                  bg_color,
        int                             steps_override = -1,
        float                           batch_ratio_override = -1.0f);

    /** 当前是否满足“满窗口触发优化”条件 */
    bool isChunkReady() const;

    /** 当前可处理 chunk 的快照信息（用于上层 task 排队） */
    ChunkInfo getCurrentChunkInfo() const;

    /** 一次优化后按配置滑窗（移除最前 slide_size 帧） */
    void slideWindowAfterOptimize();

    /** 返回窗口中的关键帧数量 */
    int windowSize() const { return static_cast<int>(window_.size()); }

    /** 是否启用（窗口非空） */
    bool isReady() const { return !window_.empty(); }

    /** 最近一次 optimize() 中实际吸收位姿更新的关键帧 ID 列表 */
    const std::vector<unsigned long>& getLastOptimizedFids() const { return last_optimized_fids_; }
    const std::vector<float>& getLastOptimizedScores() const { return last_optimized_scores_; }

private:
    struct FrameCache {
        torch::Tensor gt_image_gpu;     ///< [3,H,W] 预上传 GPU 图像
        torch::Tensor gt_mask_gpu;      ///< [1,H,W] 预上传 GPU mask
        torch::Tensor gt_image_lowres_gpu; ///< [3,h,w] 低分辨率缓存图像
        torch::Tensor gt_mask_lowres_gpu;  ///< [1,h,w] 低分辨率缓存 mask
    };

    // ─── 单帧状态 ─────────────────────────────────────────────────────────────
    struct KFState {
        unsigned long fid;
        torch::Tensor R_orig;       ///< [3,3] float, detached baseline 旋转（world-to-cam）
        torch::Tensor t_orig;       ///< [3]   float, detached baseline 平移
        torch::Tensor delta_xi;     ///< [6]   float, requires_grad=true（前3:轴角，后3:平移增量）
        torch::Tensor gt_image;     ///< [3,H,W] float, detached 真实图像
        torch::Tensor gt_mask;      ///< [1,H,W] float, 有效像素掩码（去黑边）
        int height, width;
        std::weak_ptr<GaussianKeyframe> pkf_weak;
    };

    // ─── 内部方法 ─────────────────────────────────────────────────────────────
    void rebuildOptimizer();
    void ensureFrameCache(const KFState& s);
    void releaseFrameCache(unsigned long fid);

    /// 将 delta_xi 应用到 R_orig/t_orig，返回 (R_opt [3,3], t_opt [3])（可微）
    std::pair<torch::Tensor, torch::Tensor>
    applyDelta(const KFState& s) const;

    /// 将已收敛的 delta 吸收回关键帧 Eigen 位姿，并重置 delta=0
    void absorbAndReset(int active_count, const std::vector<bool>& participated_flags);

    int activeOptimizeCount() const;

    // ─── 成员 ─────────────────────────────────────────────────────────────────
    Config              cfg_;
    torch::DeviceType   device_;

    std::deque<KFState> window_;                               ///< 滑动窗口（先进先出）
    std::unordered_map<unsigned long, FrameCache> frame_cache_;///< 滑窗帧 GPU 缓存
    std::shared_ptr<torch::optim::Adam> optimizer_;            ///< 位姿 Adam 优化器
    std::vector<unsigned long> last_optimized_fids_;           ///< 最近一次 optimize 吸收位姿的关键帧 ID
    std::vector<float> last_optimized_scores_;                 ///< 最近一次 optimize 吸收帧的变化分数（降序）
};
