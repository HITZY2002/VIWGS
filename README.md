# VIWGS

A minimal visual–inertial–wheel odometry and Gaussian mapping research prototype.

VIWGS 是基于 [VIW-Fusion](https://github.com/TouchDeeper/VIW-Fusion) 和 [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) 整理的早期系统原型：前端提供位姿、关键帧图像和稀疏点，独立 ROS 节点进行 3D Gaussian Splatting 建图。仓库用于学习、复现和实验，仅为多系统组件适配后集成，未提供完整系统的精度、实时性或长期运行保证。

此版本存在如下已知问题：

1.受前端影响较大，稍有波动会导致 3DGS 地图瞬间发散，出现巨大的不透明椭球灰色大点云；

2.性能优化不足，10000 次迭代以内会快速达到 20G 及以上显存；

3.关键帧传输、灰度彩色图片处理等等实现并不稳定成熟；

**本仓库为复现学习用途，正式新实现为[GaussVIO](https://github.com/HITZY2002/GaussVIO)**

**许可范围：** 仓库同时包含 GPL 代码与受 Inria Gaussian Splatting 非商业研究/评估条款约束的代码，不能将整体标为 MIT、Apache 或纯 GPL 项目。原始许可已保留，组合分发的兼容性仍需确认；详见 [LICENSE](LICENSE) 和 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 范围

保留的主流程：

```text
图像 + IMU + 可选轮速
        │
        ▼
VIW-Fusion 前端（vins_node / viwo_node）
        │ ROS：关键帧相机位姿、图像、稀疏点
        ▼
3DGS 建图与训练 ──► Gaussian PLY、相机参数与训练记录
        │
        └── 可选 ImGui Viewer / Depth Anything V2 补点
```

保留关键帧筛选、滑动窗口光度位姿优化、Gaussian 年龄相关处理和各向同性正则等实验代码。3DGS 侧优化的是建图关键帧位姿，不向前端反馈校正。未包含回环/GPS 模块、离线 COLMAP 输入工具、数据集、模型权重和预编译库。

默认示例使用 **EuRoC 双目 + IMU**，关闭 Viewer 与深度补点。EuRoC 为灰度图像，默认结果也是灰度外观；彩色建图需要匹配标定与时间戳的彩色输入。

```text
src/
├── camera_models/       # 前端相机模型
├── vins_estimator/      # VIW-Fusion 前端与示例标定
└── gaussian_mapping/    # 数据接入、3DGS、CUDA 光栅化与可选 Viewer
licenses/                # 原始许可文本
docker/                  # ROS Noetic + CUDA 开发环境
```

## 环境与构建

目标依赖为 Ubuntu 20.04、ROS Noetic、CUDA 11.8、CMake 3.20+、OpenCV 4.8（含 contrib 与 CUDA）、LibTorch 2.1.2 CUDA 11.8 的 C++11 ABI 版本、Ceres 1.14，以及固定提交的 Sophus。需要 NVIDIA GPU；没有 CPU 建图路径。

快速使用，建议在目标 GPU 机器构建并运行 Docker 环境，本 Docker 在实验室 4090 服务器上正常构建运行，但不保证其他环境稳定可用，如有问题请提交。宿主机需已有 Docker、NVIDIA 驱动和 NVIDIA Container Toolkit。

```bash
git clone https://github.com/HITZY2002/viwgs.git
cd viwgs
bash docker/build.sh
bash docker/run.sh -v /absolute/path/to/datasets:/data:ro
```

镜像只安装依赖，源码挂载到 `/workspace/viwgs`。镜像中的 `cv_bridge` 与工程使用相同的 OpenCV。第一次构建会下载 CUDA/LibTorch 并编译 OpenCV，耗时和磁盘占用较大。

在容器中编译：

```bash
catkin_make -j1 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DVIWGS_BUILD_VIEWER=OFF -DVIWGS_WITH_ONNX=OFF
source devel/setup.bash
```

`75` 对应 Turing。按目标 GPU 修改 CUDA 架构；如果需要重建 OpenCV 的 GPU 架构，可执行：

```bash
bash docker/build.sh --build-arg CUDA_ARCH_BIN='7.5;8.6' --build-arg BUILD_JOBS=2
```

本机编译时需自行安装上述依赖，并先加载 ROS 和使用相同 OpenCV 构建的 `cv_bridge`。通过 `Torch_DIR`、`OpenCV_DIR`、`Sophus_DIR` 指定 CMake 包位置，无需修改源码中的个人路径。默认构建不依赖 ONNX Runtime。

## 验证范围

已在上述 Docker 环境、`sm_75` 架构下通过依赖镜像构建、Release 全量编译，以及无数据条件下的 CUDA 无界面启动和 `SIGINT` 退出检查。Viewer 可选构建也已编译通过；ONNX 适配器通过了开关两种配置的语法编译检查。

本次验证不覆盖真实数据回放、GUI 显示和深度模型推理，未提供轨迹或渲染质量基准。

## 运行 EuRoC 示例

从 [EuRoC 官方页面](https://projects.asl.ethz.ch/datasets/euroc-mav/) 的下载入口获取 ROS bag，放到宿主机数据目录，例如 `MH_01_easy.bag`。本仓库不重新分发数据。

容器终端一：

```bash
source devel/setup.bash
roslaunch gaussian_mapping vins_with_3dgs.launch
```

终端二（进入同一容器，自动加载 ROS 与工作区环境）：

```bash
docker exec -it viwgs-dev /ros_entrypoint.sh bash
rosbag play --clock /data/MH_01_easy.bag
```

等待前端完成初始化、建图端收集足够的关键帧。数据播放完毕后，建图端仍可继续训练；在终端一按 `Ctrl+C` 结束并保存结果。默认输出在容器的 `/home/viwgs/.ros/viwgs`，由 `docker/run.sh` 持久化到宿主机仓库的 `output/`：

- `vins/`：轨迹与前端记录。
- `3dgs/`：训练记录与保存目录；Gaussian 模型位于 `<iteration>_shutdown/ply/point_cloud/iteration_<iteration>/point_cloud.ply`。

可覆盖输出目录：

```bash
roslaunch gaussian_mapping vins_with_3dgs.launch output_dir:=/workspace/viwgs/output/run1
```

容器重启后，在已有源码构建目录可用的情况下，入口脚本会自动加载工作区环境。

## 接入其他相机或轮速

默认配置对应 EuRoC，其他设备必须一起修改以下文件：

| 文件 | 内容 |
| --- | --- |
| `src/vins_estimator/config/` 下的前端配置与标定 | 图像/IMU/轮速话题、相机模型、外参、时间偏移、噪声与轮速开关 |
| `gaussian_mapping/config/gaussian_mapping.yaml` 的副本 | 建图相机内参、尺寸、同步容差与筛选阈值 |
| `gaussian_mapping/config/gaussian_all.yaml` 的副本 | 同一相机的畸变系数与训练参数 |

```bash
roslaunch gaussian_mapping vins_with_3dgs.launch \
  frontend_node:=viwo_node \
  config_path:=/absolute/path/to/frontend.yaml \
  mapping_config:=/absolute/path/to/mapping.yaml \
  gaussian_config:=/absolute/path/to/algorithm.yaml \
  use_sim_time:=false
```

`viwo_node` 支持轮速输入，但是否使用轮速由前端 YAML 的 `wheel` 控制。保留的 Ridgeback 标定是上游示例，不能直接视作新设备标定。实时设备使用 `use_sim_time:=false`；bag 回放使用 `true` 并添加 `rosbag play --clock`。

建图接口默认是：

| 话题 | 类型 | 约定 |
| --- | --- | --- |
| `/vins_estimator/keyframe_pose` | `nav_msgs/Odometry` | 相机到世界的位姿 `T_wc` |
| `/vins_estimator/keyframe_image` | `sensor_msgs/Image` | 与关键帧位姿时间戳一致的图像 |
| `/vins_estimator/keyframe_point` | `sensor_msgs/PointCloud` | 世界系稀疏点；沿用 VINS 的逐点 channel 格式 |

可只运行建图端，接收已有兼容数据源：

```bash
roslaunch gaussian_mapping gaussian_mapping.launch
```

## 可选功能

### Viewer

```bash
catkin_make -j1 -DVIWGS_BUILD_VIEWER=ON
source devel/setup.bash
roslaunch gaussian_mapping vins_with_3dgs.launch enable_viewer:=true
```

需要可用的 X11/OpenGL。`docker/run.sh` 在宿主机有 `DISPLAY` 时挂载 X11 socket，并在存在时传递 Xauthority；无显示环境保持默认关闭。
在无头服务器上此功能存在未知BUG，无法进行远程显示。
### Depth Anything V2

默认关闭。自行下载官方 ONNX Runtime **1.16.3 GPU 完整发行包**，保留其 `include/`、`lib/` 及所有 provider 库，并准备兼容的 ONNX 模型：

```bash
export ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime-linux-x64-gpu-1.16.3
export LD_LIBRARY_PATH="$ONNXRUNTIME_ROOT/lib:${LD_LIBRARY_PATH:-}"
catkin_make -j1 -DVIWGS_WITH_ONNX=ON -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
source devel/setup.bash
roslaunch gaussian_mapping vins_with_3dgs.launch \
  depth_fusion_enable:=true depth_model:=/absolute/path/to/depth_anything_v2_vits.onnx
```

模型输入需匹配当前实现：RGB、ImageNet 归一化、NCHW、默认 `518×518`；输出类型默认解释为相对逆深度，再通过稀疏几何估计尺度。模型导出、尺度方向及预处理差异会影响结果，不能直接替换为任意同名模型。模型路径为空、加载失败或未编译 ONNX 支持时，程序会记录日志并关闭补点。

ONNX Runtime 的官方发行包见 [v1.16.3](https://github.com/microsoft/onnxruntime/releases/tag/v1.16.3)，模型来源与许可见 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)。本仓库不提供权重或模型导出工具。

## 已知边界

- 当前实现为单进程前端与独立建图节点的实验集成；不包含完整回环地图校正，不向前端反馈光度优化结果。
- 默认参数是 EuRoC，未给出系统级性能基准或跨数据集调参结论。
- 单目初始化、时间同步、相机标定及深度尺度误差会影响建图；切换设备需重新匹配配置。
- ROS Noetic、CUDA、LibTorch 和 OpenCV 版本耦合较强。旧构建缓存与其他工作区的 `cv_bridge` 可能造成链接冲突。

## 来源与致谢

核心实现来自以下项目，原作者版权声明保留在源码中。本仓库的整理和集成不替代上游贡献：

- [VIW-Fusion](https://github.com/TouchDeeper/VIW-Fusion)：Tingda Zhuang，视觉–惯性–轮速前端；其基础为 [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)。
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)：Huajian Huang、Longwei Li、Hui Cheng、Sai-Kit Yeung，3DGS 建图实现。
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)、[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)、[simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn)：Gaussian 模型、CUDA 光栅化和近邻计算。
- [Dear ImGui](https://github.com/ocornut/imgui)、[tinyply](https://github.com/ddiakopoulos/tinyply)：界面与 PLY I/O。

学术使用时请引用实际采用的上游方法；原始引文和各部分许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。
