cmake_minimum_required(VERSION 3.10)

set(PLATFORM "aarch64-mix210-linux" CACHE STRING "Cross compile platform prefix")
set(CMAKE_C_COMPILER   ${PLATFORM}-gcc)
set(CMAKE_CXX_COMPILER ${PLATFORM}-g++)
set(CMAKE_AR           ${PLATFORM}-ar)
set(CMAKE_RANLIB       ${PLATFORM}-ranlib)

project(aiDetect LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(
    -O2
    -g
    -fPIC
)
add_definitions(-DNDEBUG)

include_directories(
    ${CMAKE_SOURCE_DIR}/fxInc/ascend/include
    ${CMAKE_SOURCE_DIR}/fxInc/opencv_4.3/include/opencv4
    ${CMAKE_SOURCE_DIR}/fxInc/mpp/out/include
    ${CMAKE_SOURCE_DIR}/fxInc/mpp/sample/common
    ${CMAKE_SOURCE_DIR}/fxInc/mpp/sample/svp/common
)

link_directories(
    ${CMAKE_SOURCE_DIR}/fxEnv/ascend/devlib
    ${CMAKE_SOURCE_DIR}/fxEnv/opencv-4.3/lib
    ${CMAKE_SOURCE_DIR}/fxEnv/mpp/out/lib
)

file(GLOB CPP_SOURCES "${CMAKE_SOURCE_DIR}/*.cpp")

add_library(aiDetect SHARED
    ${CPP_SOURCES}
)

target_link_libraries(aiDetect PRIVATE
    -Wl,--as-needed
    ascendcl
    opencv_core
    opencv_imgproc
    opencv_dnn
    opencv_imgcodecs
    opencv_video
    opencv_calib3d
    ss_dnvqe
    ss_upvqe
    ss_voice_engine
    ss_ive
    ss_mpi
    securec
    m
    pthread
)

set_target_properties(aiDetect PROPERTIES
    OUTPUT_NAME "aiDetect"
)