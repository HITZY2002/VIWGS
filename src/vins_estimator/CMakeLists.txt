cmake_minimum_required(VERSION 3.20)
project(vins)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 与 LibTorch ABI 保持一致
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
add_compile_definitions(PTHREAD_STACK_MIN=16384)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf
    cv_bridge
    camera_models
    image_transport
    rosbag)
# OpenCV 4 (Ubuntu 20.04 ROS Noetic 需要)
find_package(OpenCV 4 REQUIRED)

message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
if(OpenCV_VERSION VERSION_LESS "4.0")
    message(FATAL_ERROR "OpenCV 4.x is required for ROS Noetic on Ubuntu 20.04")
endif()

find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)


catkin_package(
  INCLUDE_DIRS src
  LIBRARIES vins_lib
  CATKIN_DEPENDS roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf
    cv_bridge
    camera_models
    image_transport
        rosbag
)

add_library(vins_lib
        src/estimator/parameters.cpp
        src/estimator/estimator.cpp
        src/estimator/feature_manager.cpp
        src/factor/pose_local_parameterization.cpp
        src/factor/pose_subset_parameterization.cpp
        src/factor/orientation_local_parameterization.cpp
        src/factor/orientation_subset_parameterization.cpp
        src/factor/projectionTwoFrameOneCamFactor.cpp
        src/factor/projectionTwoFrameTwoCamFactor.cpp
        src/factor/projectionOneFrameTwoCamFactor.cpp
        src/factor/marginalization_factor.cpp
        src/utility/utility.cpp
        src/utility/visualization.cpp
        src/utility/CameraPoseVisualization.cpp
        src/initial/solve_5pts.cpp
        src/initial/initial_aligment.cpp
        src/initial/initial_sfm.cpp
        src/initial/initial_ex_rotation.cpp
        src/featureTracker/feature_tracker.cpp)
target_link_libraries(vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})


add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib)

add_executable(viwo_node src/rosNodeTestWheel.cpp)
target_link_libraries(viwo_node vins_lib)

install(TARGETS vins_lib vins_node viwo_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
