#include "FaceRecognition.h"
#include "ModelProfiler.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "opencv2/opencv.hpp"
#include "acl/acl.h"

using namespace cv;
using namespace std;
using namespace VISION_ENGINE;

CFaceRecognition::CFaceRecognition()
    : bInit_(false)
    , picDevBuffer_(nullptr)
    , devBufferSize_(0)
    , input_mean_(127.5f)
    , input_std_(127.5f) {
}

CFaceRecognition::~CFaceRecognition() {
    release();
}

int CFaceRecognition::init(const FaceRecogConfig& config) {
    if (bInit_) return 0;
    release();
    config_ = config;

    // 设置预处理参数
    if (config.find_sub && config.find_mul) {
        // mxnet arcface模型
        input_mean_ = 0.0f;
        input_std_ = 1.0f;
        means_ = { 0.0f, 0.0f, 0.0f };
        stds_ = { 1.0f, 1.0f, 1.0f };
    }
    else {
        // 其他模型
        input_mean_ = 127.5f;
        input_std_ = 127.5f;
        means_ = { 127.5f, 127.5f, 127.5f };
        stds_ = { 1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f };
    }

    // 如果配置中有指定means和stds，使用配置的值
    if (!config.means.empty()) {
        means_ = config.means;
    }
    if (!config.stds.empty()) {
        stds_ = config.stds;
    }

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

    netSize_ = Size(inputDims_[1], inputDims_[2]); // 假设NHWC
    //INFO_LOG("FaceRecognition model input size: %dx%d", netSize_.width, netSize_.height);

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

    bInit_ = true;
    INFO_LOG("[FaceReco] init");

    return 0;
}

int CFaceRecognition::get(const cv::Mat& img, FaceInfo& face) {
    // int64_t ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();

    if (!bInit_) {
        ERROR_LOG("FaceRecognition module not initialized!");
        return -1;
    }

    if (img.empty()) {
        ERROR_LOG("input image is empty!");
        return -1;
    }

    if (face.kps.size() != 10) {
        ERROR_LOG("face must have 5 keypoints (10 values)");
        return -1;
    }

    // 人脸对齐
    cv::Mat aligned_face = norm_crop(img, face.kps, config_.imageSize);
    if (aligned_face.empty()) {
        ERROR_LOG("Face alignment failed!");
        return -1;
    }

    // 提取特征
    face.embedding = get_feat(aligned_face);

    // int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // std::cout << "CFaceRecognition get time: " << (now_ms - ts_ms) << " ms" << std::endl;

    return 0;
}

std::vector<float> CFaceRecognition::get_feat(const cv::Mat& aligned_face) {
    std::vector<float> feature;

    if (!bInit_) {
        ERROR_LOG("FaceRecognition module not initialized!");
        return feature;
    }

    fxprof::add_call(fxprof::Model::FaceReco);

    // 预处理
    cv::Mat processed;
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Pre);
        if (preProcess(aligned_face, processed) < 0) {
            ERROR_LOG("Preprocess failed!");
            return feature;
        }
    }

    // 推理（inference 内部分别计时 Infer / Post）
    if (inference(processed, feature) < 0) {
        ERROR_LOG("Inference failed!");
        return feature;
    }

    // t = ((double)getTickCount() - t) / getTickFrequency();
    // cout << "face recognition infrence time:" << t << "s" << endl;

    return feature;
}

cv::Mat CFaceRecognition::norm_crop(const cv::Mat& img, const std::vector<float>& landmark, int image_size) {
    if (landmark.size() != 10) {
        ERROR_LOG("landmark size must be 10");
        return cv::Mat();
    }

    // ArcFace的标准参考点（112x112）
    std::vector<cv::Point2f> srcPoints(5);
    std::vector<cv::Point2f> dstPoints(5);

    // 设置源关键点
    for (int i = 0; i < 5; i++) {
        srcPoints[i] = cv::Point2f(landmark[i * 2], landmark[i * 2 + 1]);
    }

    // 计算缩放比例和偏移
    float ratio;
    float diff_x = 0.0f;

    if (image_size % 112 == 0) {
        ratio = static_cast<float>(image_size) / 112.0f;
    }
    else if (image_size % 128 == 0) {
        ratio = static_cast<float>(image_size) / 128.0f;
        diff_x = 8.0f * ratio;
    }
    else {
        // 默认使用112的缩放
        ratio = static_cast<float>(image_size) / 112.0f;
    }

    // 标准ArcFace 112x112模板点
    cv::Point2f base_dst[5] = {
        cv::Point2f(38.2946, 51.6963),
        cv::Point2f(73.5318, 51.5014),
        cv::Point2f(56.0252, 71.7366),
        cv::Point2f(41.5493, 92.3655),
        cv::Point2f(70.7299, 92.2041)
    };

    // 应用缩放和偏移
    for (int i = 0; i < 5; i++) {
        dstPoints[i] = cv::Point2f(base_dst[i].x * ratio + diff_x,
            base_dst[i].y * ratio);
    }

    // 相似变换（4 DoF：旋转 + 均匀缩放 + 平移）
    // 利用全部 5 个关键点最小二乘拟合，禁止剪切和非均匀缩放
    cv::Mat M = cv::estimateAffinePartial2D(srcPoints, dstPoints,
        cv::noArray(), cv::LMEDS);
    if (M.empty()) {
        // 极端退化情况（所有关键点重合），回退到 3 点仿射
        std::vector<cv::Point2f> srcTri = { srcPoints[0], srcPoints[1], srcPoints[2] };
        std::vector<cv::Point2f> dstTri = { dstPoints[0], dstPoints[1], dstPoints[2] };
        M = cv::getAffineTransform(srcTri, dstTri);
    }

    // 应用变换
    cv::Mat aligned_face;
    cv::warpAffine(img, aligned_face, M, cv::Size(image_size, image_size),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    return aligned_face;
}

int CFaceRecognition::preProcess(const cv::Mat& src, cv::Mat& dst) {
    // 调整尺寸
    cv::Mat resized;
    if (src.size() != netSize_) {
        cv::resize(src, resized, netSize_, 0, 0, cv::INTER_LINEAR);
    }
    else {
        resized = src.clone();
    }

    // 颜色空间转换
    if (config_.bgr2rgb) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }
    dst = resized;
    size_t data_len = dst.rows * dst.cols * dst.channels();
    Result ret = Utils::MemcpyHostToDevice(dst.data, picDevBuffer_, data_len);
    if (ret != SUCCESS) {
        ERROR_LOG("memcpy device buffer failed");
        return -1;
    }

    return 0;
}

int CFaceRecognition::inference(const cv::Mat& img, std::vector<float>& feature) {
    // 执行推理
    Result ret;
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Infer);
        ret = modelProcess_.Execute();
    }
    if (ret != SUCCESS) {
        ERROR_LOG("execute inference failed");
        return -1;
    }

    fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Post);
    return postProcess(feature);
}

int CFaceRecognition::postProcess(std::vector<float>& feature) {
    // 获取模型输出数据
    std::vector<std::vector<float>> modelOutput;
    if (modelProcess_.OutputModelResult(modelOutput) != SUCCESS) {
        ERROR_LOG("read FaceReco model output failed");
        return -1;
    }

    if (modelOutput.empty() || modelOutput[0].empty()) {
        ERROR_LOG("model output is empty");
        return -1;
    }

    // 提取特征向量
    feature = modelOutput[0];

    // L2归一化
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float& val : feature) {
            val /= norm;
        }
    }

    return 0;
}

float CFaceRecognition::compute_sim(const std::vector<float>& feat1, const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size() || feat1.empty()) {
        ERROR_LOG("feature dimension mismatch or empty");
        return 0.0f;
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < feat1.size(); ++i) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 < 1e-12 || norm2 < 1e-12) {
        return 0.0f;
    }

    return dot_product / (norm1 * norm2);
}

void CFaceRecognition::release() {
    const bool had_resources = bInit_ || picDevBuffer_ != nullptr;
    bInit_ = false;
    modelProcess_.Release();
    if (picDevBuffer_ != nullptr) {
        aclError ret = aclrtFree(picDevBuffer_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("[FaceReco] free input buffer failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        picDevBuffer_ = nullptr;
    }
    devBufferSize_ = 0;
    inputDims_.clear();
    outputDims_.clear();
    if (had_resources) INFO_LOG("[FaceReco] release");
}
