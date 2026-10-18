cmake_minimum_required(VERSION 3.20)
project(camera_models)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )

find_package(Boost REQUIRED COMPONENTS filesystem system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_models
    CATKIN_DEPENDS roscpp std_msgs
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

include_directories("include")

add_library(camera_models
        src/camera_models/Camera.cc
        src/camera_models/CameraFactory.cc
        src/camera_models/CostFunctionFactory.cc
        src/camera_models/PinholeCamera.cc
        src/camera_models/PinholeFullCamera.cc
        src/camera_models/CataCamera.cc
        src/camera_models/EquidistantCamera.cc
        src/camera_models/ScaramuzzaCamera.cc
        src/gpl/gpl.cc
        src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(camera_models ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

install(TARGETS camera_models
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY include/ DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
