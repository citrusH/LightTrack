cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED FX_CHIP)
    set(FX_CHIP 1)
endif()

if(${FX_CHIP} EQUAL 1)
    set(PLATFORM_PREFIX "aarch64-mix210-linux")
elseif(${FX_CHIP} EQUAL 2)
    set(PLATFORM_PREFIX "aarch64-v01c01-linux-gnu")
else()
    set(PLATFORM_PREFIX "aarch64-mix210-linux")
endif()

# 交叉编译器
set(CMAKE_C_COMPILER    ${PLATFORM_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER  ${PLATFORM_PREFIX}-g++)
set(CMAKE_AR            ${PLATFORM_PREFIX}-ar)
set(CMAKE_RANLIB        ${PLATFORM_PREFIX}-ranlib)
set(CMAKE_STRIP         ${PLATFORM_PREFIX}-strip)

project(aitrack LANGUAGES C CXX)
set(TARGET_NAME aitrack)

# 头文件
include_directories(
    ${CMAKE_SOURCE_DIR}/code_fx
    ${CMAKE_SOURCE_DIR}/code_source
)

file(GLOB SRC_C
    ${CMAKE_SOURCE_DIR}/*.c
    ${CMAKE_SOURCE_DIR}/code_fx/*.c
    ${CMAKE_SOURCE_DIR}/code_source/*.c
)

# 编译标志
add_compile_options(
    -fPIC
    -ffunction-sections
    -fdata-sections
    -DFX_CHIP=${FX_CHIP}
)

# 构建动态库
add_library(${TARGET_NAME} SHARED ${SRC_C})


target_link_libraries(${TARGET_NAME}
    PRIVATE
    -Wl,--gc-sections
    stdc++
)

# 编译后自动strip
# add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
#     COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${TARGET_NAME}>
#     COMMENT "strip libaitrack.so"
# )