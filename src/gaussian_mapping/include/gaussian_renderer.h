/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <tuple>

#include <torch/torch.h>

#include "sh_utils.h"

#include "gaussian_parameters.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_rasterizer.h"

class GaussianRenderer
{
public:
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> render(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);

    /**
     * @brief 可微位姿渲染：将 3D 高斯球预变换到相机坐标系，支持对 R_cw / t_cw 求梯度。
     *
     * 由于底层 CUDA 光栅化器的 backward 不计算 viewmatrix 的梯度，本函数采用以下策略：
     *   1. 用 torch::mm 将 means3D 变换到相机坐标系（可微）；
     *   2. 传入单位矩阵作为 viewmatrix，使光栅化器认为点已在相机空间；
     *   3. 颜色由 SH 预计算（detach），避免错误的 SH 方向干扰；
     *   4. dL/d_means3D_cam 通过链式法则自动流回 R_cw 和 t_cw。
     *
     * @param viewpoint_camera  关键帧（提供 FoV、projection_matrix_ 等相机参数）
     * @param R_cw  [3,3] 世界->相机旋转矩阵（float，可带 requires_grad）
     * @param t_cw  [3] 世界->相机平移向量（float，可带 requires_grad）
     * @param gaussians  Gaussian 模型（内部参数 detach，不影响 Gaussian 优化器）
     * @param pipe  流水线参数
     * @param bg_color  背景颜色张量
     * @return  渲染图像 [3, H, W]，关于 R_cw/t_cw 可微
     */
    static torch::Tensor renderWithDiffPose(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        torch::Tensor R_cw,
        torch::Tensor t_cw,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color);

    /** 构造 [3x3] 反对称矩阵（differentiable），v: [3] */
    static torch::Tensor skewMatrix(const torch::Tensor& v);

    /** Rodrigues 公式：轴角向量 v [3] -> 旋转矩阵 R [3x3]（differentiable）*/
    static torch::Tensor rodrigues(const torch::Tensor& v);
};
