FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=2
ARG CUDA_ARCH_BIN="7.5;8.6"
ARG USER_UID=1000
ARG USER_GID=1000
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git gnupg2 build-essential ninja-build cmake unzip wget \
    python3-pip python3-dev python3-setuptools python3-numpy \
    libeigen3-dev libboost-all-dev libceres-dev libfmt-dev libyaml-cpp-dev \
    libjsoncpp-dev libglm-dev libglfw3-dev libgl1-mesa-dev libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libjpeg-dev libpng-dev libtiff-dev \
    && python3 -m pip install --no-cache-dir cmake==3.27.9 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/ros.gpg] http://packages.ros.org/ros/ubuntu focal main' \
      > /etc/apt/sources.list.d/ros1.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-ros-base ros-noetic-image-transport ros-noetic-tf ros-noetic-tf2-ros \
      ros-noetic-geometry-msgs ros-noetic-nav-msgs ros-noetic-sensor-msgs ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

# Keep the install prefix outside the disposable OpenCV source tree.
RUN git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv.git /opt/opencv-src \
    && git clone --depth 1 --branch 4.8.0 https://github.com/opencv/opencv_contrib.git /opt/opencv-contrib \
    && cmake -S /opt/opencv-src -B /opt/opencv-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencv \
      -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv-contrib/modules \
      -DBUILD_LIST=cudev,core,imgproc,imgcodecs,calib3d,features2d,video,videoio,highgui,cudaarithm,cudaimgproc,cudawarping,cudastereo \
      -DWITH_CUDA=ON -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="" \
      -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DWITH_OPENGL=ON \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF \
    && cmake --build /opt/opencv-build --parallel ${BUILD_JOBS} \
    && cmake --install /opt/opencv-build \
    && rm -rf /opt/opencv-src /opt/opencv-contrib /opt/opencv-build
ENV OpenCV_DIR=/opt/opencv/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/opt/opencv/lib:/opt/libtorch/lib:${LD_LIBRARY_PATH}

RUN wget -q 'https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu118.zip' -O /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV LIBTORCH_ROOT=/opt/libtorch
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch
RUN git init /tmp/Sophus && cd /tmp/Sophus \
    && git remote add origin https://github.com/strasdat/Sophus.git \
    && git fetch --depth 1 origin a0fe89a323e20c42d3cecb590937eb7a06b8343a && git checkout FETCH_HEAD \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    && cmake --build build --parallel ${BUILD_JOBS} && cmake --install build \
    && rm -rf /tmp/Sophus

# Build cv_bridge against the same OpenCV as the frontend and mapper.
RUN git init /tmp/bridge_ws/src/vision_opencv && cd /tmp/bridge_ws/src/vision_opencv \
    && git remote add origin https://github.com/ros-perception/vision_opencv.git \
    && git fetch --depth 1 origin cfabf72fb02970a661b5e68fbee503c5d9f94729 && git checkout FETCH_HEAD \
    && source /opt/ros/noetic/setup.bash && cd /tmp/bridge_ws \
    && catkin_make install -j${BUILD_JOBS} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/viwgs_ros \
      -DOpenCV_DIR=${OpenCV_DIR} -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && test -f /opt/viwgs_ros/setup.bash \
    && rm -rf /tmp/bridge_ws

RUN groupadd --gid ${USER_GID} viwgs \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m viwgs \
    && mkdir -p /workspace/viwgs && chown viwgs:viwgs /workspace/viwgs
COPY docker/entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
USER viwgs
WORKDIR /workspace/viwgs
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
