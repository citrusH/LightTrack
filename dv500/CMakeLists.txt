cmake_minimum_required(VERSION 3.10)


if(NOT DEFINED PLATFORM)
    set(PLATFORM "aarch64-v01c01-linux-gnu")
endif()


set(CMAKE_C_COMPILER    ${PLATFORM}-gcc)
set(CMAKE_CXX_COMPILER  ${PLATFORM}-g++)
set(CMAKE_AR            ${PLATFORM}-ar)
set(CMAKE_RANLIB        ${PLATFORM}-ranlib)

project(aiDetect LANGUAGES C CXX)# 项目名

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

# 全局编译参数，对应 COMPILE_FLAGS
add_compile_options(
    -g
    -fPIC
)

# INC_DIRS
include_directories(
    ./fxInc/ascend/include 
    ./fxInc/opencv_4.3/include/opencv4 
    ./fxInc/mpp/out/include 
    ./fxInc/mpp/sample/common 
    ./fxInc/mpp/sample/svp/common
)

# LIB_DIRS
link_directories(
    ./fxEnv/ascend/devlib 
    ./fxEnv/opencv-4.3/lib 
    ./fxEnv/mpp/out/lib 
)


set(SRC_ALL
    sample_common_svp_npu_model.c
    Detector.cpp
    FaceRecognition.cpp
    FaceRecognitionSystem.cpp
    Facekps.cpp
    KalmanBoxTracker.cpp
    KalmanFilter.cpp
    LightTracker.cpp
    PersonReID.cpp
    PoseEstimator.cpp
    SvpAclRuntime.cpp
    Track.cpp
    fx_wrapper_track.cpp
    utils.cpp
)

add_library(aiDetect SHARED ${SRC_ALL})

target_link_libraries(aiDetect PRIVATE
    -Wl,--as-needed
    -Wl,--copy-dt-needed-entries
    ascendcl
    opencv_core
    opencv_imgproc
    opencv_dnn
    opencv_imgcodecs
    securec
    m
    pthread
    opencv_video
    -Wl,-Bstatic
    svp_acl
    protobuf-c
    ss_mpi_km
    ss_mpi_sysmem
    -Wl,-Bdynamic
)
