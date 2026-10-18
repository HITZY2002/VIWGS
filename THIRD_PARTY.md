# Third-party sources and license scope

VIWGS is a derivative integration prototype. Original copyright and license
notices in source files are retained. This inventory records the visible source
lineage; it does not assert that unknown historical import commits have been
identified or that upstream owners have licensed the combined work differently.

| Component | Local scope | Upstream and license evidence |
| --- | --- | --- |
| VIW-Fusion / VINS-Fusion | `src/vins_estimator`, `src/camera_models` | [VIW-Fusion](https://github.com/TouchDeeper/VIW-Fusion), based on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion); original repository GPLv3 text is preserved as `licenses/GPL-3.0.txt`; existing file notices remain in place. |
| Photo-SLAM | `src/gaussian_mapping` | [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM); GPL-marked files specify v3 or later; Gaussian Splatting-derived files retain their separate Inria notices. |
| Gaussian Splatting / rasterizer / simple-knn | Inria-marked Gaussian mapping files, `cuda_rasterizer`, `ThirdParty/simple-knn` | [Photo-SLAM's retained Gaussian-Splatting license](https://github.com/HuajianUP/Photo-SLAM/blob/f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac/LICENSE.md); exact text copied to `licenses/Gaussian-Splatting.md` and the source locations that refer to `LICENSE.md`. Research/evaluation and non-commercial restrictions apply as specified by that text. |
| Dear ImGui | `src/gaussian_mapping/src/viewer/imgui` | [Dear ImGui](https://github.com/ocornut/imgui); original MIT text is retained at `src/gaussian_mapping/src/viewer/imgui/LICENSE.txt`. |
| tinyply | `src/gaussian_mapping/ThirdParty/tinyply` | [tinyply](https://github.com/ddiakopoulos/tinyply); its file headers contain a public-domain dedication and fallback permission. |

Repository-original integration, build and documentation changes use
GPL-3.0-or-later as stated in `LICENSE`, without changing inherited terms.

The `LicenseRef-Gaussian-Splatting` and `LicenseRef-tinyply` package metadata
identifiers refer to the corresponding entries above, not blanket licenses
over the entire mapping package.

The original import commits for Photo-SLAM and VIW-Fusion are not recorded in the
available local history. For comparison during source cleanup, the inspected
upstream tips were Photo-SLAM `f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac` and
VIW-Fusion `37075eb8beeaf93acf15bbc86109deee3777faca`.
These are reference snapshots, **not claimed import versions**. The preserved
VIW-Fusion license and `simple-knn/simple_knn.cu` were byte-identical to the
inspected upstream copies.

## Combined-distribution status

The tree includes both GPL-covered code and files carrying Gaussian Splatting's
non-commercial terms. Copying both license texts does not establish that their
combination can be distributed consistently with all applicable terms. That
compatibility remains unresolved here. No blanket MIT/Apache/GPL license or
commercial-use permission is granted by this repository. Before public release
of the combined system, resolve this with the relevant rights holders or use an
appropriately licensed replacement. The current cleanup prepares a reviewable
source tree; it does not claim license clearance.

## External dependencies

OpenCV, ROS/catkin/vision_opencv, LibTorch, CUDA/cuDNN, Ceres, Sophus, Eigen, Boost,
GLM, GLFW and jsoncpp are installed separately. Their own licenses apply.
The Dockerfile pins the Sophus and vision_opencv source revisions used by the
build. No dependency binaries are vendored in this source snapshot.

Optional ONNX Runtime and Depth Anything V2 models must also be obtained
separately with their own notices. Consult the [Depth Anything V2 license
section](https://github.com/DepthAnything/Depth-Anything-V2#license) for the
specific model size; model licenses differ. Dataset and pretrained-model
redistribution rights are not granted by this repository.

## Cleanup changes

The source cleanup on 2026-10-17 removes unrelated modules, datasets, PDFs,
prebuilt binaries and obsolete integrations; consolidates ROS entrypoints;
makes Viewer/ONNX support optional; replaces machine-specific paths; and aligns
the default example with EuRoC pinhole calibration. Original method code and
copyright notices are retained. Modified source is identifiable in the Git diff;
this document records the modification date without replacing original notices.

## Upstream research references

- Huajian Huang, Longwei Li, Hui Cheng and Sai-Kit Yeung.
  **Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping
  for Monocular, Stereo, and RGB-D Cameras.** CVPR 2024.
  [Official project](https://huajianup.github.io/research/Photo-SLAM/).
- Tingda Zhuang. **VIW-Fusion: visual-inertial-wheel fusion odometry.** 2021.
  [Repository and citation](https://github.com/TouchDeeper/VIW-Fusion#4-citation).
- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler and George Drettakis.
  **3D Gaussian Splatting for Real-Time Radiance Field Rendering.**
  ACM Transactions on Graphics, 2023.
  [Official project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).
