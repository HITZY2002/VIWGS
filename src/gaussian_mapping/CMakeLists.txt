cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING "CUDA target architectures")
endif()
project(gaussian_mapping LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
option(VIWGS_BUILD_VIEWER "Build the GLFW/ImGui viewer" OFF)
option(VIWGS_WITH_ONNX "Enable Depth Anything V2 through ONNX Runtime" OFF)

find_package(catkin REQUIRED COMPONENTS
    roscpp std_msgs geometry_msgs nav_msgs sensor_msgs cv_bridge image_transport tf tf2 tf2_ros)
find_package(Torch REQUIRED HINTS "$ENV{LIBTORCH_ROOT}/share/cmake/Torch")
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4.8 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(glm REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_compile_definitions(PTHREAD_STACK_MIN=16384)

catkin_package(INCLUDE_DIRS include
    LIBRARIES gaussian_mapping_lib gaussian_mapping_cuda_lib simple_knn
    CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs sensor_msgs cv_bridge image_transport tf tf2 tf2_ros
    DEPENDS OpenCV EIGEN3)

include_directories(include src ${PROJECT_SOURCE_DIR}
    ThirdParty/simple-knn ThirdParty/tinyply cuda_rasterizer
    ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

add_library(simple_knn SHARED
    ThirdParty/simple-knn/simple_knn.cu ThirdParty/simple-knn/spatial.cu)
set_target_properties(simple_knn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(simple_knn ${TORCH_LIBRARIES} CUDA::cudart)

add_library(gaussian_mapping_cuda_lib SHARED
    src/operate_points.cu src/rasterize_points.cu src/stereo_vision.cu
    cuda_rasterizer/backward.cu cuda_rasterizer/forward.cu cuda_rasterizer/rasterizer_impl.cu)
set_target_properties(gaussian_mapping_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(gaussian_mapping_cuda_lib glm::glm Eigen3::Eigen ${TORCH_LIBRARIES} CUDA::cudart)

add_library(gaussian_mapping_lib SHARED
    ThirdParty/tinyply/tinyply.cpp
    src/gaussian_keyframe.cpp src/gaussian_model.cpp src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp src/gaussian_renderer.cpp src/gaussian_scene.cpp
    src/gaussian_trainer.cpp src/gaussian_mapper.cpp src/sliding_window_pose_optimizer.cpp
    src/depth_anything_v2_estimator.cpp src/ros_data_provider.cpp)
target_link_libraries(gaussian_mapping_lib
    gaussian_mapping_cuda_lib simple_knn ${catkin_LIBRARIES} ${OpenCV_LIBS}
    ${TORCH_LIBRARIES} Eigen3::Eigen ${Sophus_LIBRARIES} jsoncpp_lib)

if(VIWGS_WITH_ONNX)
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime installation")
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
    target_include_directories(gaussian_mapping_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(gaussian_mapping_lib PUBLIC VIWGS_WITH_ONNX)
    # Execution providers are loaded by ONNX Runtime, not linked directly here.
    target_link_libraries(gaussian_mapping_lib ${ONNXRUNTIME_LIBRARY})
endif()

add_executable(gaussian_mapping_node src/ros/gaussian_mapping_node.cpp)
target_link_libraries(gaussian_mapping_node gaussian_mapping_lib)
set(VIWGS_INSTALL_TARGETS simple_knn gaussian_mapping_cuda_lib gaussian_mapping_lib gaussian_mapping_node)

if(VIWGS_BUILD_VIEWER)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 REQUIRED)
    add_library(gaussian_viewer SHARED
        src/viewer/imgui_viewer.cpp
        src/viewer/imgui/imgui.cpp src/viewer/imgui/imgui_draw.cpp
        src/viewer/imgui/imgui_tables.cpp src/viewer/imgui/imgui_widgets.cpp
        src/viewer/imgui/imgui_demo.cpp src/viewer/imgui/imgui_impl_glfw.cpp
        src/viewer/imgui/imgui_impl_opengl3.cpp)
    target_include_directories(gaussian_viewer PRIVATE src/viewer/imgui)
    target_link_libraries(gaussian_viewer gaussian_mapping_lib glm::glm glfw OpenGL::GL)
    target_compile_definitions(gaussian_mapping_node PRIVATE VIWGS_BUILD_VIEWER)
    target_link_libraries(gaussian_mapping_node gaussian_viewer)
    list(APPEND VIWGS_INSTALL_TARGETS gaussian_viewer)
endif()

install(TARGETS ${VIWGS_INSTALL_TARGETS}
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/ DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(FILES ThirdParty/simple-knn/simple_knn.h ThirdParty/simple-knn/spatial.h
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/ThirdParty/simple-knn)
install(FILES ThirdParty/tinyply/tinyply.h
    DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}/ThirdParty/tinyply)
