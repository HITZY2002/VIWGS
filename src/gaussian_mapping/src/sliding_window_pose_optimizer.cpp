/*
 * SlidingWindowPoseOptimizer — 实现
 */

#include "include/sliding_window_pose_optimizer.h"
#include "include/gaussian_renderer.h"
#include "include/loss_utils.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace F = torch::nn::functional;

namespace {
inline float radToDeg(float rad) {
    return rad * 57.29577951308232f;
}
}

// ─────────────────────────────────────────────────────────────────────────────
//  构造
// ─────────────────────────────────────────────────────────────────────────────
SlidingWindowPoseOptimizer::SlidingWindowPoseOptimizer(Config cfg, torch::DeviceType device)
    : cfg_(cfg), device_(device)
{}

// ─────────────────────────────────────────────────────────────────────────────
//  push：将关键帧加入窗口
// ─────────────────────────────────────────────────────────────────────────────
void SlidingWindowPoseOptimizer::push(std::shared_ptr<GaussianKeyframe> pkf)
{
    // ── 从 Eigen 位姿提取旋转矩阵和平移向量 ──────────────────────────────────
    Eigen::Matrix3d R_eigen = pkf->R_quaternion_.normalized().toRotationMatrix();
    Eigen::Vector3d t_eigen = pkf->t_;

    // ── 转为 float Tensor（CUDA）────────────────────────────────────────────
    auto opts_f32 = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(device_);

    auto R_orig = torch::zeros({3, 3}, opts_f32);
    auto t_orig = torch::zeros({3},   opts_f32);
    {
        // 转 float 并复制到设备
        auto R_f = R_eigen.cast<float>();
        auto t_f = t_eigen.cast<float>();
        for (int i = 0; i < 3; i++) {
            t_orig[i] = t_f(i);
            for (int j = 0; j < 3; j++)
                R_orig[i][j] = R_f(i, j);
        }
    }

    // ── 初始化 delta_xi = [0,0,0, 0,0,0]（requires_grad）────────────────────
    auto delta_xi = torch::zeros({6}, opts_f32.requires_grad(true));

    // ── 获取真实图像（detach 到 CPU 缓存，渲染时再移到 GPU）─────────────────
    auto gt_img = pkf->original_image_.detach().cpu();  // [3,H,W]
    auto gt_mask = (gt_img.sum(0, /*keepdim=*/true) > 1e-6f).to(torch::kFloat32); // [1,H,W]

    // ── 组装状态 ─────────────────────────────────────────────────────────────
    KFState s;
    s.fid      = pkf->fid_;
    s.R_orig   = R_orig;
    s.t_orig   = t_orig;
    s.delta_xi = delta_xi;
    s.gt_image = gt_img;
    s.gt_mask  = gt_mask;
    s.height   = pkf->image_height_;
    s.width    = pkf->image_width_;
    s.pkf_weak = pkf;

    window_.push_back(std::move(s));

    const int min_keep = std::max(1, cfg_.anchor_size + 1);
    const int window_cap = std::max(min_keep, cfg_.window_size);
    while (static_cast<int>(window_.size()) > window_cap) {
        releaseFrameCache(window_.front().fid);
        window_.pop_front();
    }

    rebuildOptimizer();
}

bool SlidingWindowPoseOptimizer::isChunkReady() const
{
    return static_cast<int>(window_.size()) >= cfg_.window_size
           && cfg_.window_size > cfg_.anchor_size;
}

SlidingWindowPoseOptimizer::ChunkInfo SlidingWindowPoseOptimizer::getCurrentChunkInfo() const
{
    ChunkInfo info;
    info.active_count = activeOptimizeCount();
    info.anchor_count = std::max(0, std::min(cfg_.anchor_size, info.active_count));
    info.frame_fids.reserve(info.active_count);
    info.anchor_fids.reserve(info.anchor_count);
    for (int i = 0; i < info.active_count; ++i) {
        info.frame_fids.push_back(window_[i].fid);
        if (i < info.anchor_count) {
            info.anchor_fids.push_back(window_[i].fid);
        }
    }
    return info;
}

void SlidingWindowPoseOptimizer::slideWindowAfterOptimize()
{
    int to_drop = std::min(cfg_.slide_size, static_cast<int>(window_.size()));
    for (int i = 0; i < to_drop; ++i) {
        releaseFrameCache(window_.front().fid);
        window_.pop_front();
    }
    rebuildOptimizer();
}

int SlidingWindowPoseOptimizer::activeOptimizeCount() const
{
    return std::min(static_cast<int>(window_.size()), cfg_.window_size);
}

// ─────────────────────────────────────────────────────────────────────────────
//  rebuildOptimizer：用当前所有 delta_xi 重建 Adam 优化器
// ─────────────────────────────────────────────────────────────────────────────
void SlidingWindowPoseOptimizer::rebuildOptimizer()
{
    const int active_count = activeOptimizeCount();
    const int anchor_count = std::max(0, std::min(cfg_.anchor_size, active_count));

    std::vector<torch::Tensor> params;
    params.reserve(std::max(0, active_count - anchor_count));
    for (int i = anchor_count; i < active_count; ++i)
        params.push_back(window_[i].delta_xi);

    if (params.empty()) {
        optimizer_.reset();
        return;
    }

    optimizer_ = std::make_shared<torch::optim::Adam>(
        params,
        torch::optim::AdamOptions(cfg_.lr));
}

void SlidingWindowPoseOptimizer::ensureFrameCache(const KFState& s)
{
    if (frame_cache_.find(s.fid) != frame_cache_.end()) {
        return;
    }

    FrameCache cache;
    auto gt_image_gpu = s.gt_image.to(device_, /*non_blocking=*/true);
    auto gt_mask_gpu = s.gt_mask.to(device_, /*non_blocking=*/true);

    int ds = cfg_.cache_downsample;
    if (ds != 2 && ds != 4) ds = 1;
    if (ds > 1) {
        const int low_h = std::max(1, s.height / ds);
        const int low_w = std::max(1, s.width / ds);
        cache.gt_image_lowres_gpu = F::interpolate(
            gt_image_gpu.unsqueeze(0),
            F::InterpolateFuncOptions().size(std::vector<int64_t>{low_h, low_w}).mode(torch::kBilinear).align_corners(false)
        ).squeeze(0);
        cache.gt_mask_lowres_gpu = F::interpolate(
            gt_mask_gpu.unsqueeze(0),
            F::InterpolateFuncOptions().size(std::vector<int64_t>{low_h, low_w}).mode(torch::kNearest)
        ).squeeze(0);
    }

    if (cfg_.use_lowres_cache && cache.gt_image_lowres_gpu.defined()) {
        cache.gt_image_gpu = torch::Tensor();
        cache.gt_mask_gpu = torch::Tensor();
    } else {
        cache.gt_image_gpu = std::move(gt_image_gpu);
        cache.gt_mask_gpu = std::move(gt_mask_gpu);
    }
    frame_cache_.emplace(s.fid, std::move(cache));
}

void SlidingWindowPoseOptimizer::releaseFrameCache(unsigned long fid)
{
    frame_cache_.erase(fid);
}

// ─────────────────────────────────────────────────────────────────────────────
//  applyDelta：将 delta_xi 应用到 R_orig/t_orig，返回 (R_opt, t_opt)（可微）
//
//    delta_xi[0:3] = delta_r（轴角向量）
//    delta_xi[3:6] = delta_t（平移增量）
//
//    R_delta = Rodrigues(delta_r)
//    R_opt   = R_delta @ R_orig
//    t_opt   = R_delta @ t_orig + delta_t
// ─────────────────────────────────────────────────────────────────────────────
std::pair<torch::Tensor, torch::Tensor>
SlidingWindowPoseOptimizer::applyDelta(const KFState& s) const
{
    auto delta_r = s.delta_xi.narrow(0, 0, 3);  // [3]
    auto delta_t = s.delta_xi.narrow(0, 3, 3);  // [3]

    auto R_delta = GaussianRenderer::rodrigues(delta_r);                 // [3,3]
    auto R_opt   = torch::mm(R_delta, s.R_orig.to(R_delta.device()));   // [3,3]
    auto t_opt   = torch::mv(R_delta, s.t_orig.to(R_delta.device()))
                   + delta_t;                                             // [3]
    return {R_opt, t_opt};
}

// ─────────────────────────────────────────────────────────────────────────────
//  optimize：主优化循环
// ─────────────────────────────────────────────────────────────────────────────
void SlidingWindowPoseOptimizer::optimize(
    std::shared_ptr<GaussianModel>  gaussians,
    GaussianPipelineParams&         pipe,
    torch::Tensor&                  bg_color,
    int                             steps_override,
    float                           batch_ratio_override)
{
    last_optimized_fids_.clear();
    last_optimized_scores_.clear();

    if (!isChunkReady()) return;
    if (!optimizer_) rebuildOptimizer();
    if (!optimizer_) return;

    torch::AutoGradMode enable_grad(true);

    auto dev_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(device_);

    const int optimize_steps = (steps_override > 0) ? steps_override : cfg_.steps_per_call;
    const int active_count = activeOptimizeCount();
    const int anchor_count = std::max(0, std::min(cfg_.anchor_size, active_count));
    const int candidate_count = std::max(0, active_count - anchor_count);
    const float batch_ratio_base = (batch_ratio_override > 0.0f) ? batch_ratio_override : cfg_.batch_ratio;
    const float batch_ratio = std::max(0.0f, std::min(1.0f, batch_ratio_base));
    int batch_count = static_cast<int>(std::ceil(batch_ratio * static_cast<float>(candidate_count)));
    if (candidate_count > 0) {
        batch_count = std::max(1, batch_count);
        batch_count = std::min(batch_count, candidate_count);
    }

    thread_local std::mt19937 rng(std::random_device{}());

    // ── steps_per_call 步 Adam 优化 ──────────────────────────────────────────
    float last_photo_loss = 0.0f;
    float last_reg_loss = 0.0f;
    int last_valid_kf = 0;
    std::vector<bool> participated_flags(active_count, false);
    for (int step = 0; step < optimize_steps; ++step) {
        auto total_loss = torch::zeros({1}, dev_opts);
        int valid_kf = 0;
        auto photo_loss_acc = torch::zeros({1}, dev_opts);
        auto reg_loss_acc = torch::zeros({1}, dev_opts);

        std::vector<int> candidates(candidate_count);
        std::iota(candidates.begin(), candidates.end(), anchor_count);
        std::shuffle(candidates.begin(), candidates.end(), rng);

        for (int k = 0; k < batch_count; ++k) {
            int i = candidates[k];
            auto& s = window_[i];
            auto pkf = s.pkf_weak.lock();
            if (!pkf) continue;

            // 计算当前优化位姿
            auto [R_opt, t_opt] = applyDelta(s);

            // 可微渲染
            auto rendered = GaussianRenderer::renderWithDiffPose(
                pkf, R_opt, t_opt, gaussians, pipe, bg_color);

            // 掩码光度损失（去除 undistort 黑边）
            ensureFrameCache(s);
            auto cache_it = frame_cache_.find(s.fid);
            if (cache_it == frame_cache_.end()) {
                continue;
            }
            auto gt = cache_it->second.gt_image_gpu;  // [3,H,W]
            auto mask = cache_it->second.gt_mask_gpu; // [1,H,W]
            if (cfg_.use_lowres_cache && cache_it->second.gt_image_lowres_gpu.defined()) {
                gt = cache_it->second.gt_image_lowres_gpu;
                mask = cache_it->second.gt_mask_lowres_gpu;
                rendered = F::interpolate(
                    rendered.unsqueeze(0),
                    F::InterpolateFuncOptions()
                        .size(std::vector<int64_t>{gt.size(1), gt.size(2)})
                        .mode(torch::kBilinear)
                        .align_corners(false)
                ).squeeze(0);
            } else if (!gt.defined() || !mask.defined()) {
                continue;
            }
            auto masked_render = rendered * mask;
            auto masked_gt = gt * mask;

            auto l1 = torch::mean(torch::abs(masked_render - masked_gt));
            auto ssim_loss = 1.0f - loss_utils::ssim(masked_render, masked_gt, device_);
            auto photo_loss = (1.0f - cfg_.lambda_ssim) * l1 + cfg_.lambda_ssim * ssim_loss;

            // 小扰动正则：限制每次窗口优化偏离基准位姿太多
            auto delta_r = s.delta_xi.narrow(0, 0, 3);
            auto delta_t = s.delta_xi.narrow(0, 3, 3);
            auto pose_reg = cfg_.rot_reg_weight * torch::mean(delta_r * delta_r)
                          + cfg_.trans_reg_weight * torch::mean(delta_t * delta_t);

            auto loss = photo_loss + pose_reg;
            total_loss = total_loss + loss;
            photo_loss_acc = photo_loss_acc + photo_loss;
            reg_loss_acc = reg_loss_acc + pose_reg;
            participated_flags[i] = true;
            ++valid_kf;
        }

        if (valid_kf == 0) {
            optimizer_->zero_grad(true);
            continue;
        }

        total_loss.backward();

        // delta_xi 梯度裁剪，抑制位姿更新过冲
        if (cfg_.max_grad_norm > 0.0f) {
            for (auto& s : window_) {
                if (!s.delta_xi.grad().defined()) continue;
                auto g = s.delta_xi.grad();
                float gn = g.norm().item<float>();
                if (gn > cfg_.max_grad_norm) {
                    s.delta_xi.grad().mul_(cfg_.max_grad_norm / (gn + 1e-8f));
                }
            }
        }

        optimizer_->step();
        optimizer_->zero_grad(true);

        last_valid_kf = valid_kf;
        last_photo_loss = (photo_loss_acc / static_cast<float>(valid_kf)).item<float>();
        last_reg_loss = (reg_loss_acc / static_cast<float>(valid_kf)).item<float>();
    }

    if (last_valid_kf > 0) {
        std::cout << "[PoseOpt] window=" << window_.size()
                  << " active=" << active_count
                  << " anchors=" << anchor_count
                  << " batch=" << batch_count
                  << " steps=" << optimize_steps
                  << " photo=" << last_photo_loss
                  << " reg=" << last_reg_loss
                  << std::endl;
    }

    // ── 将优化结果吸收回关键帧 Eigen 位姿 ───────────────────────────────────
    absorbAndReset(active_count, participated_flags);
}

// ─────────────────────────────────────────────────────────────────────────────
//  absorbAndReset：吸收 delta 到 Eigen 位姿，重置 delta=0，重建优化器
// ─────────────────────────────────────────────────────────────────────────────
void SlidingWindowPoseOptimizer::absorbAndReset(int active_count, const std::vector<bool>& participated_flags)
{
    torch::NoGradGuard no_grad;

    active_count = std::max(0, std::min(active_count, static_cast<int>(window_.size())));
    const int anchor_count = std::max(0, std::min(cfg_.anchor_size, active_count));
    const float rot_thr_deg = std::max(0.0f, cfg_.absorb_rot_min_deg);
    const float trans_thr = std::max(0.0f, cfg_.absorb_trans_min);

    struct PendingWriteback {
        int idx = -1;
        unsigned long fid = 0;
        std::shared_ptr<GaussianKeyframe> pkf;
        torch::Tensor R_opt;
        torch::Tensor t_opt;
        float rot_deg = 0.0f;
        float trans_norm = 0.0f;
        float score = 0.0f;
        bool participated = false;
        bool anchor = false;
        bool changed_enough = false;
    };
    std::vector<PendingWriteback> pending;
    pending.reserve(active_count);

    for (int i = 0; i < active_count; ++i) {
        auto& s = window_[i];
        auto pkf = s.pkf_weak.lock();
        if (!pkf) continue;

        const bool is_anchor = (i < anchor_count);
        const bool participated = (i < static_cast<int>(participated_flags.size())) ? participated_flags[i] : false;

        auto delta_cpu = s.delta_xi.detach().cpu();
        const float rot_deg = radToDeg(delta_cpu.narrow(0, 0, 3).norm().item<float>());
        const float trans_norm = delta_cpu.narrow(0, 3, 3).norm().item<float>();
        const bool changed_enough = (rot_deg >= rot_thr_deg) || (trans_norm >= trans_thr);

        // 计算吸收后的 R_opt、t_opt（detach）
        auto [R_opt, t_opt] = applyDelta(s);

        PendingWriteback wb;
        wb.idx = i;
        wb.fid = s.fid;
        wb.pkf = pkf;
        wb.R_opt = R_opt;
        wb.t_opt = t_opt;
        wb.rot_deg = rot_deg;
        wb.trans_norm = trans_norm;
        wb.score = rot_deg + 100.0f * trans_norm;
        wb.participated = participated;
        wb.anchor = is_anchor;
        wb.changed_enough = changed_enough;
        pending.push_back(std::move(wb));

        // 新基准 = 吸收后的位姿，delta 归零
        s.R_orig = R_opt.detach().clone();
        s.t_orig = t_opt.detach().clone();
        s.delta_xi.fill_(0.0f);

        std::cout << "[PoseOpt] kf " << s.fid
                  << " participated=" << (participated ? 1 : 0)
                  << " anchor=" << (is_anchor ? 1 : 0)
                  << " rot_deg=" << rot_deg
                  << " trans=" << trans_norm
                  << " writeback=" << ((!is_anchor && participated && changed_enough) ? 1 : 0)
                  << " delta reset.\n";
    }

    std::vector<PendingWriteback*> selected;
    selected.reserve(pending.size());
    for (auto& wb : pending) {
        if (!wb.anchor && wb.participated && wb.changed_enough) {
            selected.push_back(&wb);
        }
    }
    std::sort(selected.begin(), selected.end(),
              [](const PendingWriteback* a, const PendingWriteback* b) {
                  return a->score > b->score;
              });

    int top_k = cfg_.max_optimized_fids_per_absorb;
    if (top_k <= 0) top_k = static_cast<int>(selected.size());
    top_k = std::min(top_k, static_cast<int>(selected.size()));

    for (int si = 0; si < top_k; ++si) {
        auto* wb = selected[si];
        if (!wb || !wb->pkf) continue;

        auto R_cpu = wb->R_opt.detach().cpu().to(torch::kDouble);
        auto t_cpu = wb->t_opt.detach().cpu().to(torch::kDouble);

        Eigen::Matrix3d R_mat;
        for (int ri = 0; ri < 3; ri++)
            for (int cj = 0; cj < 3; cj++)
                R_mat(ri, cj) = R_cpu[ri][cj].item<double>();

        Eigen::Vector3d t_vec;
        for (int ti = 0; ti < 3; ti++)
            t_vec(ti) = t_cpu[ti].item<double>();

        wb->pkf->setPose(Eigen::Quaterniond(R_mat).normalized(), t_vec);
        wb->pkf->computeTransformTensors();
        last_optimized_fids_.push_back(wb->fid);
        last_optimized_scores_.push_back(wb->score);
    }

    // 重建优化器（重置 Adam 动量，从新基准重新开始）
    rebuildOptimizer();
}
