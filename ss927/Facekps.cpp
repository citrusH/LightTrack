/**
 * @file FaceKeypoint106.cpp
 *
 * 人脸关键点检测模型推理实现（昇腾平台版本）
 */

#include "Facekps.h"
#include "ModelProfiler.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include "acl/acl.h"

using namespace cv;
using namespace std;
using namespace VISION_ENGINE;

//const float M_PI = 3.14159265358979323846;

CFaceKeypoint106::CFaceKeypoint106()
    : initialized_(false)
    , picDevBuffer_(nullptr)
    , devBufferSize_(0) {
}

CFaceKeypoint106::~CFaceKeypoint106() {
    release();
}

int CFaceKeypoint106::init(const FaceKeypointConfig& config) {
    if (initialized_) return 0;
    release();
    config_ = config;
    inputSize_ = config.inputSize;

    // 从内存加载模型
    Result ret = modelProcess_.LoadModelFromMem(config.bufferModel, config.sizeModel);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromMem failed");
        release();
        return -1;
    }

    // 创建模型描述
    ret = modelProcess_.CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateModelDesc failed");
        release();
        return -1;
    }

    // 获取输入维度
    ret = modelProcess_.GetInputDimsByIndex(0, inputDims_);
    if (ret != SUCCESS) {
        ERROR_LOG("execute GetInputDimsByIndex failed");
        release();
        return -1;
    }

    if (inputDims_.size() != 4) {
        ERROR_LOG("invalid input dimensions, expected 4, got %zu", inputDims_.size());
        release();
        return -1;
    }

    INFO_LOG("FaceKeypoint model input size: %dx%d", inputSize_.width, inputSize_.height);

    // 获取输出维度
    ret = modelProcess_.GetOutputDimsByIndex(0, outputDims_);
    if (ret != SUCCESS) {
        ERROR_LOG("execute GetOutputDimsByIndex failed");
        release();
        return -1;
    }

    // 获取输入大小
    ret = modelProcess_.GetInputSizeByIndex(0, devBufferSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("execute GetInputSizeByIndex failed");
        release();
        return -1;
    }

    // 分配设备端输入缓冲区
    aclError aclRet = aclrtMalloc(&picDevBuffer_, devBufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu, errorCode is %d",
            devBufferSize_, static_cast<int32_t>(aclRet));
        picDevBuffer_ = nullptr;
        release();
        return -1;
    }

    // 创建模型输入
    ret = modelProcess_.CreateInput(picDevBuffer_, devBufferSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        release();
        return -1;
    }

    // 创建模型输出
    ret = modelProcess_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        release();
        return -1;
    }

    initialized_ = true;
    INFO_LOG("[FaceKps] init");

    return 0;
}

FaceKeypointResult CFaceKeypoint106::run(const cv::Mat& src_img, const cv::Rect& face_bbox) {
    // int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();
    FaceKeypointResult result;
    result.face_rect = face_bbox;

    if (!initialized_) {
        ERROR_LOG("FaceKeypoint106 module not initialized!");
        return result;
    }

    if (src_img.empty()) {
        ERROR_LOG("input image is empty!");
        return result;
    }

    fxprof::add_call(fxprof::Model::FaceKps);

    // 预处理
    cv::Mat processed_img;
    cv::Mat M;
    int ret;
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Pre);
        ret = preProcess(src_img, face_bbox, processed_img, M);
    }
    if (ret < 0) {
        ERROR_LOG("pre-process failed! ret=%d", ret);
        return result;
    }

    // 推理
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Infer);
        ret = inference(processed_img);
    }
    if (ret < 0) {
        ERROR_LOG("inference failed! ret=%d", ret);
        return result;
    }

    // 后处理
    std::vector<FaceKeypoint> keypoints;
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Post);
        ret = postProcess(M, face_bbox, keypoints);
    }
    if (ret < 0) {
        ERROR_LOG("post-process failed! ret=%d", ret);
        return result;
    }
    
    result.points = keypoints;

    // int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    // std::chrono::steady_clock::now().time_since_epoch()).count();

    // std::cout << "FaceKeypoint106 run time: " << (now_ms - ts_ms) << " ms" << std::endl;

    return result;
}

cv::Mat CFaceKeypoint106::GetAffineTransform(const cv::Point2f& center,
    int output_size,
    float scale,
    float rotation) {
    cv::Mat M(2, 3, CV_64F);

    double rot_rad = rotation * M_PI / 180.0;
    double sn = sin(rot_rad);
    double cs = cos(rot_rad);

    M.at<double>(0, 0) = scale * cs;
    M.at<double>(0, 1) = scale * -sn;
    M.at<double>(1, 0) = scale * sn;
    M.at<double>(1, 1) = scale * cs;

    M.at<double>(0, 2) = output_size * 0.5 - scale * cs * center.x + scale * sn * center.y;
    M.at<double>(1, 2) = output_size * 0.5 - scale * sn * center.x - scale * cs * center.y;

    return M;
}

std::vector<cv::Point2f> CFaceKeypoint106::TransformPoints(const std::vector<cv::Point2f>& points,
    const cv::Mat& M) {
    if (points.empty()) return {};

    std::vector<cv::Point2f> transformed_points(points.size());

    for (size_t i = 0; i < points.size(); i++) {
        const cv::Point2f& pt = points[i];
        double new_x = M.at<double>(0, 0) * pt.x + M.at<double>(0, 1) * pt.y + M.at<double>(0, 2);
        double new_y = M.at<double>(1, 0) * pt.x + M.at<double>(1, 1) * pt.y + M.at<double>(1, 2);
        transformed_points[i] = cv::Point2f(static_cast<float>(new_x), static_cast<float>(new_y));
    }

    return transformed_points;
}


int CFaceKeypoint106::preProcess(const cv::Mat& src, const cv::Rect& bbox,
    cv::Mat& dst, cv::Mat& M) {
    if (src.empty()) {
        return -1;
    }

    float w = bbox.width - bbox.x;
    float h = bbox.height - bbox.y;
    cv::Point2f center(bbox.x + w / 2.0f, bbox.y + h / 2.0f);
    float rotate = 0.0f;
    float scale = inputSize_.width / (std::max(w, h) * 1.5f);

    M = GetAffineTransform(center, inputSize_.width, scale, rotate);

    // 对齐人脸
    cv::Mat aligned_face;
    cv::warpAffine(src, aligned_face, M, inputSize_,
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    if (config_.bgr2rgb) {
        cv::cvtColor(aligned_face, aligned_face, cv::COLOR_BGR2RGB);
    }

    dst = aligned_face;

    int channel_pixels = inputSize_.width * inputSize_.height;
    size_t channel_size = channel_pixels * sizeof(uint8_t);
    size_t requiredSize = channel_size * 3;

    //INFO_LOG("Input size: %dx%d, Buffer size: %zu, Required size: %zu",
    //    inputSize_.width, inputSize_.height, devBufferSize_, requiredSize);

    // 检查缓冲区大小
    if (devBufferSize_ < requiredSize) {
        ERROR_LOG("Device buffer size too small: %zu < %zu", devBufferSize_, requiredSize);
        return -1;
    }


    size_t data_len = dst.rows * dst.cols * dst.channels();
    Result ret = Utils::MemcpyHostToDevice(dst.data, picDevBuffer_, data_len);
    if (ret != SUCCESS) {
        ERROR_LOG("memcpy device buffer failed");
        return -1;
    }

    return 0;
}

int CFaceKeypoint106::inference(const cv::Mat& image) {
    // 执行推理
    Result ret = modelProcess_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("execute inference failed");
        return -1;
    }

    return 0;
}

int CFaceKeypoint106::postProcess(const cv::Mat& M, const cv::Rect& face_bbox,
    std::vector<FaceKeypoint>& result) {
    result.clear();

    // 获取模型输出数据
    std::vector<std::vector<float>> modelOutput;
    if (modelProcess_.OutputModelResult(modelOutput) != SUCCESS) {
        ERROR_LOG("read FaceKps model output failed");
        return -1;
    }

    if (modelOutput.empty() || modelOutput[0].empty()) {
        ERROR_LOG("model output is empty");
        return -1;
    }

    // 解析关键点
    const float* output_ptr = modelOutput[0].data();

    int num_points = config_.numPoints;
    int lmk_dim = 2;  // 每个关键点有x,y两个坐标

    std::vector<cv::Point2f> points_2d;
    //std::cout << "before postProcess : " << std::endl;
    if (modelOutput[0].size() >= num_points * lmk_dim) {
        for (int i = 0; i < num_points; i++) {
            float x = output_ptr[i * lmk_dim];
            float y = output_ptr[i * lmk_dim + 1];
            //std::cout << " x : " << x << " y " << y;
            // 从[-1,1]映射到[0, inputSize_.width]
            x = (x + 1.0f) * (int)(inputSize_.width / 2.0f);
            y = (y + 1.0f) * (int)(inputSize_.width / 2.0f);

            points_2d.emplace_back(x, y);
        }
    }

    // 计算逆变换矩阵
    cv::Mat IM;
    cv::invertAffineTransform(M, IM);

    // 变换回原图坐标
    auto transformed_points = TransformPoints(points_2d, IM);

    // 转换为输出格式
    for (size_t i = 0; i < transformed_points.size(); i++) {
        FaceKeypoint kp;
        kp.x = transformed_points[i].x;
        kp.y = transformed_points[i].y;
        result.push_back(kp);
    }

    return 0;
}

void CFaceKeypoint106::release() {
    const bool had_resources = initialized_ || picDevBuffer_ != nullptr;
    initialized_ = false;
    modelProcess_.Release();
    if (picDevBuffer_ != nullptr) {
        aclError ret = aclrtFree(picDevBuffer_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("[FaceKps] free input buffer failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        picDevBuffer_ = nullptr;
    }
    devBufferSize_ = 0;
    inputDims_.clear();
    outputDims_.clear();
    if (had_resources) INFO_LOG("[FaceKps] release");
}
