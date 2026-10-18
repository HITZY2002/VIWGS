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

#include "include/gaussian_renderer.h"

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros_like(pc->getXYZ(),
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->getScalingActivation();
        rotations = pc->getRotationActivation();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else {
            shs = pc->getFeatures();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen). 
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii               /*radii*/
    );
}

// =============================================================================
//  辅助：反对称矩阵（differentiable）
// =============================================================================
torch::Tensor GaussianRenderer::skewMatrix(const torch::Tensor& v)
{
    // v: [3]，构造反对称矩阵 K，使得 K @ x = v × x
    auto v0 = v[0].unsqueeze(0);   // [1]
    auto v1 = v[1].unsqueeze(0);   // [1]
    auto v2 = v[2].unsqueeze(0);   // [1]
    auto z  = torch::zeros({1}, v.options());

    auto row0 = torch::cat({z,   -v2,  v1});   // [3]
    auto row1 = torch::cat({v2,   z,  -v0});   // [3]
    auto row2 = torch::cat({-v1,  v0,  z  });  // [3]
    return torch::stack({row0, row1, row2});    // [3, 3]
}

// =============================================================================
//  辅助：Rodrigues 旋转公式（differentiable）
//    v ∈ R³ 为轴角向量（||v|| = 旋转角度，方向 = 旋转轴）
//    当 ||v|| → 0 时退化为 I（分子 sin(θ)→0, (1-cos(θ))→0）
// =============================================================================
torch::Tensor GaussianRenderer::rodrigues(const torch::Tensor& v)
{
    auto theta = torch::norm(v);                   // scalar
    auto axis  = v / (theta + 1e-8f);             // [3] unit axis
    auto K  = skewMatrix(axis);                    // [3,3]
    auto K2 = torch::mm(K, K);                     // [3,3]

    auto R = torch::eye(3, v.options())
           + torch::sin(theta) * K
           + (1.0f - torch::cos(theta)) * K2;
    return R;  // [3,3]
}

// =============================================================================
//  renderWithDiffPose：可微位姿渲染
//    R_cw [3,3]、t_cw [3] 可带 requires_grad=true
//    梯度路径：rendered_image -> dL/d_means_cam -> dL/dR_cw, dL/dt_cw
// =============================================================================
torch::Tensor GaussianRenderer::renderWithDiffPose(
    std::shared_ptr<GaussianKeyframe> cam,
    torch::Tensor R_cw,           // [3,3] world-to-cam 旋转，float
    torch::Tensor t_cw,           // [3]   world-to-cam 平移，float
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& /*pipe*/,
    torch::Tensor& bg_color)
{
    auto dev = R_cw.device();
    int H = cam->image_height_;
    int W = cam->image_width_;

    // ------------------------------------------------------------------
    // 1. 将 3D 高斯球均值变换到相机坐标系（可微）
    //    means_cam = xyz_world @ R_cw.T + t_cw     [N,3]
    // ------------------------------------------------------------------
    auto xyz_world = pc->getXYZ().detach();
    int64_t N_all = xyz_world.size(0);
    auto means_cam_all = xyz_world.mm(R_cw.t())
                       + t_cw.unsqueeze(0).expand({N_all, 3});

    // Pose optimization is sensitive to runtime for large maps. Pre-filter points with a
    // coarse camera-space frustum test before SH/color prep and rasterization.
    auto means_cam_detached = means_cam_all.detach();
    auto z = means_cam_detached.index({torch::indexing::Slice(), 2});
    auto x = means_cam_detached.index({torch::indexing::Slice(), 0});
    auto y = means_cam_detached.index({torch::indexing::Slice(), 1});

    const float near_z = std::max(1e-3f, cam->znear_);
    const float far_z = std::max(near_z + 1e-3f, cam->zfar_);
    const float fov_margin = 1.10f;
    float tanfovx = std::tan(cam->FoVx_ * 0.5f);
    float tanfovy = std::tan(cam->FoVy_ * 0.5f);

    auto z_ok = torch::logical_and(z > near_z, z < far_z);
    auto x_ok = torch::abs(x) < (z * tanfovx * fov_margin);
    auto y_ok = torch::abs(y) < (z * tanfovy * fov_margin);
    auto visible_mask = torch::logical_and(z_ok, torch::logical_and(x_ok, y_ok));
    auto visible_idx = torch::nonzero(visible_mask).squeeze(1);

    if (!visible_idx.defined() || visible_idx.numel() == 0) {
        return bg_color.view({3, 1, 1}).expand({3, H, W}).clone();
    }

    auto means_cam = means_cam_all.index({visible_idx});
    int64_t N = means_cam.size(0);

    // ------------------------------------------------------------------
    // 2. 预计算 SH 颜色（使用 detach 后的位姿，颜色不参与梯度）
    //    相机中心（世界系）：c_w = -R_cw.T @ t_cw
    // ------------------------------------------------------------------
    torch::Tensor colors_precomp;
    {
        auto cam_center = torch::mv(R_cw.detach().t(), -t_cw.detach());
        int max_sh_deg  = pc->max_sh_degree_ + 1;
        int sh_dim      = max_sh_deg * max_sh_deg;

        auto xyz_world_vis = xyz_world.index({visible_idx});
        auto feats = pc->getFeatures().detach().index({visible_idx});
        auto shs_view = feats.transpose(1, 2).view({N, 3, sh_dim});

        auto dir = xyz_world_vis - cam_center.unsqueeze(0).expand({N, 3});
        auto dir_len = torch::frobenius_norm(dir, /*dim=*/{1}, /*keepdim=*/true) + 1e-8f;
        auto dir_norm = dir / dir_len;

        auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_norm);
        colors_precomp = torch::clamp_min(sh2rgb + 0.5f, 0.0f).detach();
    }

    // ------------------------------------------------------------------
    // 3. 构造光栅化设置
    //    viewmatrix = I₄ （点已在相机坐标系，无需再变换）
    //    projmatrix = projection_matrix_ （纯透视投影）
    //    campos     = [0,0,0] （相机在相机坐标系中位于原点）
    // ------------------------------------------------------------------
    auto I4 = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto proj_only = cam->projection_matrix_.to(dev);  // [4,4] 纯透视投影（已转置）

    auto campos = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    GaussianRasterizationSettings raster_settings(
        H, W,
        tanfovx, tanfovy,
        bg_color,
        1.0f,       // scaling_modifier
        I4,         // viewmatrix = 单位矩阵
        proj_only,  // projmatrix = 纯透视
        pc->active_sh_degree_,
        campos,
        false       // prefiltered
    );

    GaussianRasterizer rasterizer(raster_settings);

    // ------------------------------------------------------------------
    // 4. screenspace_points（光栅化器要求，此处不用于密度化）
    // ------------------------------------------------------------------
    auto ss_pts = torch::zeros({N, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(dev).requires_grad(true));
    try { ss_pts.retain_grad(); } catch (...) {}

    // ------------------------------------------------------------------
    // 5. 光栅化（Gaussian 旋转/缩放使用世界坐标系近似，适用于小扰动）
    //    注意：严格做法需将四元数旋转到相机系：q_cam = q(R_cw) * q_gauss
    //    对于小角度扰动，该近似引入的误差为二阶小量，不影响梯度方向。
    // ------------------------------------------------------------------
    auto opacities = pc->getOpacityActivation().detach().index({visible_idx});
    auto scales    = pc->getScalingActivation().detach().index({visible_idx});
    auto rotations = pc->getRotationActivation().detach().index({visible_idx});
    torch::Tensor cov3D_empty;

    auto rasterizer_result = rasterizer.forward(
        means_cam,          // [N,3] 相机空间，梯度可流回 R_cw/t_cw
        ss_pts,
        opacities,
        /*has_shs*/          false,
        /*has_colors_precomp*/true,
        /*has_scales*/       true,
        /*has_rotations*/    true,
        /*has_cov3D_precomp*/false,
        torch::Tensor(),    // shs（空）
        colors_precomp,
        scales,
        rotations,
        cov3D_empty
    );

    return std::get<0>(rasterizer_result);  // [3, H, W]
}