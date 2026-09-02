
#include "PersonReID.h"
#include "ModelProfiler.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include "acl/acl.h"

using namespace cv;
using namespace std;
using namespace VISION_ENGINE;

CPersonReID::CPersonReID()
    : bInit_(false)
    , picDevBuffer_(nullptr)
    , devBufferSize_(0) {
}

CPersonReID::~CPersonReID() {
    release();
}

int CPersonReID::init(const ReIDConfig& config) {
    if (bInit_) return 0;
    release();
    config_ = config;

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

    // 解析网络输入尺寸（兼容 NCHW 和 NHWC）
    // NCHW: [1, 3, 256, 128]  →  netSize_ = (128, 256)
    // NHWC: [1, 256, 128, 3]  →  netSize_ = (128, 256)
    if (inputDims_[1] > 0 && inputDims_[1] <= 4) {
        // NCHW: [N, C, H, W]
        netSize_ = Size(inputDims_[3], inputDims_[2]);
    } else {
        // NHWC: [N, H, W, C]
        netSize_ = Size(inputDims_[2], inputDims_[1]);
    }
    INFO_LOG("ReID model input size: %dx%d", netSize_.width, netSize_.height);

    // 获取输入缓冲区大小
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
    INFO_LOG("[ReID] init");
    return 0;
}

int CPersonReID::run(const cv::Mat& image, const cv::Rect& box,
    cv::Mat& feature, bool needNorm) {
    if (!bInit_) {
        ERROR_LOG("PersonReID module not initialized!");
        return -1;
    }
    if (image.empty()) {
        ERROR_LOG("input image is empty!");
        return -1;
    }

    fxprof::add_call(fxprof::Model::ReID);

    // 预处理（裁剪+缩放+颜色转换+拷贝到设备）
    Mat processedImg;
    int ret;
    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Pre);
        ret = preProcess(image, box, processedImg);
    }
    if (ret < 0) return -1;

    // 推理
    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Infer);
        ret = inference(processedImg);
    }
    if (ret < 0) return -1;

    // 后处理
    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Post);
        ret = postProcess(feature, needNorm);
    }
    if (ret < 0) return -1;

    return 0;
}


int CPersonReID::preProcess(const cv::Mat& src, const cv::Rect& box, cv::Mat& dst) {
    // 注意：本项目中 cv::Rect 语义为 (x1, y1, x2, y2)，非标准 (x, y, w, h)
    int x1 = max(box.x, 0);
    int y1 = max(box.y, 0);
    int x2 = min(box.width, src.cols - 1);
    int y2 = min(box.height, src.rows - 1);

    if (x1 >= x2 || y1 >= y2) {
        ERROR_LOG("invalid box coordinates: (%d,%d)-(%d,%d)", x1, y1, x2, y2);
        return -1;
    }

    // ① 不再 .clone()：直接以 ROI 视图作为 resize 输入（零拷贝 header）。
    //    resize 支持非连续/带 ROI 的源，省掉每候选一整框的像素拷贝。
    Mat roi = src(Rect(x1, y1, x2 - x1, y2 - y1));

    // ② 复用暂存缓冲：thread_local，避免每候选都重新分配目标 Mat。
    //    目标尺寸恒为 netSize_，resize 内部 create() 命中后原地复用、不再申请内存。
    //    保持 INTER_LINEAR：ReID 嵌入依赖纹理细节，NEAREST 会降质→更易 ID 跳变，
    //    而 resize 本身并非耗时大头（耗时在 clone/分配/上传，已在 ①②④ 处理）。
    thread_local Mat stage;
    resize(roi, stage, netSize_, 0, 0, INTER_LINEAR);

    // ③ 颜色：仅当模型未用 AIPP 在芯片上做 BGR→RGB 时才转；小图原地转换，开销可忽略。
    if (config_.bgr2rgb)
        cvtColor(stage, stage, COLOR_BGR2RGB);

    // ④ 上传设备：picDevBuffer_ 为 aclrtMalloc 的设备内存，OpenCV 无法直接写入，
    //    必须保留一次 memcpy（HWC/uint8 打包布局，归一化与 HWC→CHW 由模型 AIPP 完成）。
    size_t imgBytes = stage.total() * stage.elemSize();
    if (imgBytes > devBufferSize_) {
        ERROR_LOG("ReID input too large: %zu > %zu", imgBytes, devBufferSize_);
        return -1;
    }
    Result copy_ret = Utils::MemcpyHostToDevice(stage.data, picDevBuffer_, imgBytes);
    if (copy_ret != SUCCESS) {
        ERROR_LOG("memcpy device buffer failed");
        return -1;
    }

    dst = stage;   // 出参仅作占位（inference 用 picDevBuffer_）：header 浅拷贝，无数据复制
    return 0;
}

int CPersonReID::inference(const cv::Mat& image) {
    // 执行推理
    Result ret = modelProcess_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("execute inference failed");
        return -1;
    }

    return 0;
}

int CPersonReID::postProcess(cv::Mat& feature, bool needNorm) {
    // 获取模型输出数据
    vector<vector<float>> modelOutput;
    if (modelProcess_.OutputModelResult(modelOutput) != SUCCESS) {
        ERROR_LOG("read ReID model output failed");
        return -1;
    }

    if (modelOutput.empty() || modelOutput[0].empty()) {
        ERROR_LOG("model output is empty");
        return -1;
    }

    int out_len = static_cast<int>(modelOutput[0].size());
    Mat feature1(1, out_len, CV_32FC1, &modelOutput[0][0]);

    cv::Mat nanMask;
    cv::compare(feature1, feature1, nanMask, cv::CMP_NE); // NaN != NaN 为真
    feature1.setTo(0, nanMask);

    if (needNorm) {
        float fnorm = norm(feature1, NORM_L2);
        if (fnorm > 1e-6f) feature1 /= fnorm;
    }
    feature = feature1.clone();

    return 0;
}


float CPersonReID::computeSimilarity(const std::vector<float>& feat1,
    const std::vector<float>& feat2) {
    if (feat1.size() != feat2.size() || feat1.empty()) {
        ERROR_LOG("feature dimension mismatch or empty");
        return -1.0f;
    }

    float dotProduct = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < feat1.size(); ++i) {
        dotProduct += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }

    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);

    if (norm1 == 0.0f || norm2 == 0.0f) {
        return 0.0f;
    }

    return dotProduct / (norm1 * norm2);
}

void CPersonReID::normalizeFeature(std::vector<float>& feature) {
    if (feature.empty()) return;

    // 计算L2范数
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    // 归一化（避免除零）
    if (norm > 1e-6f) {
        for (float& val : feature) {
            val /= norm;
        }
    }
    else {
        // 范数接近零的处理：可以置零或保持原样
        std::fill(feature.begin(), feature.end(), 0.0f);
    }
}

void CPersonReID::release() {
    const bool had_resources = bInit_ || picDevBuffer_ != nullptr;
    bInit_ = false;
    modelProcess_.Release();
    if (picDevBuffer_ != nullptr) {
        aclError ret = aclrtFree(picDevBuffer_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("[ReID] free input buffer failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        picDevBuffer_ = nullptr;
    }
    devBufferSize_ = 0;
    inputDims_.clear();
    if (had_resources) INFO_LOG("[ReID] release");
}
