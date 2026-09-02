#include "Detector.h"
#include "ModelProfiler.h"

#include "acl/acl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace cv;
using namespace std;
using namespace VISION_ENGINE;

#ifndef FX_TRACKER_LIFECYCLE_TEST_HOOKS
#define FX_TRACKER_LIFECYCLE_TEST_HOOKS 0
#endif

namespace {

bool fail_detector_yolox_stage_for_test(const char* stage)
{
#if FX_TRACKER_LIFECYCLE_TEST_HOOKS
    const char* requested = std::getenv("FX_TRACKER_TEST_FAIL_DETECTOR");
    return requested != nullptr && stage != nullptr
        && std::strcmp(requested, stage) == 0;
#else
    (void)stage;
    return false;
#endif
}

} // namespace

Detector_yolox::Detector_yolox() = default;

Detector_yolox::~Detector_yolox()
{
    release();
}

int Detector_yolox::init(Config& config)
{
    if (initialized_) return SUCCESS;
    release();
    config_ = config;

    Result ret = personDetector_yolox.LoadModelFromMem(
        config.bufferModel, config.sizeModel);
    if (ret != SUCCESS) {
        ERROR_LOG("YOLOX LoadModelFromMem failed");
        release();
        return FAILED;
    }

    ret = personDetector_yolox.CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("YOLOX CreateModelDesc failed");
        release();
        return FAILED;
    }

    ret = personDetector_yolox.GetInputDimsByIndex(0, input_dim_);
    if (ret != SUCCESS || input_dim_.size() != 4) {
        ERROR_LOG("YOLOX invalid input dimensions: %zu", input_dim_.size());
        release();
        return FAILED;
    }

    INFO_LOG("[YOLOX] input_dim=[%d,%d,%d,%d]",
             input_dim_[0], input_dim_[1], input_dim_[2], input_dim_[3]);

    if (input_dim_[1] == 3) {
        net_size_ = cv::Size(input_dim_[3], input_dim_[2]);
        INFO_LOG("[YOLOX] model input layout=NCHW, size=%dx%d",
                 net_size_.width, net_size_.height);
    } else if (input_dim_[3] == 3) {
        net_size_ = cv::Size(input_dim_[2], input_dim_[1]);
        INFO_LOG("[YOLOX] model input layout=NHWC, size=%dx%d",
                 net_size_.width, net_size_.height);
    } else {
        ERROR_LOG("YOLOX unknown input layout: [%d,%d,%d,%d]",
                  input_dim_[0], input_dim_[1], input_dim_[2], input_dim_[3]);
        release();
        return FAILED;
    }

    if (net_size_.width != 416 || net_size_.height != 416) {
        ERROR_LOG("YOLOX unsupported input size: %dx%d, expected 416x416",
                  net_size_.width, net_size_.height);
        release();
        return FAILED;
    }

    ret = personDetector_yolox.GetOutputDimsByIndex(0, output_dim_);
    if (ret != SUCCESS || output_dim_.size() < 2
        || output_dim_[output_dim_.size() - 2] != 3549
        || output_dim_[output_dim_.size() - 1] != 8) {
        const int actual_anchors = output_dim_.size() >= 2
            ? output_dim_[output_dim_.size() - 2] : -1;
        const int actual_dim = !output_dim_.empty() ? output_dim_.back() : -1;
        ERROR_LOG("YOLOX output layout mismatch: rank=%zu tail=[%d,%d], "
                  "expected [...,3549,8]",
                  output_dim_.size(), actual_anchors, actual_dim);
        release();
        return FAILED;
    }
    size_t output_elements = 1;
    for (int dim : output_dim_) {
        if (dim <= 0) {
            ERROR_LOG("YOLOX invalid output dimension: %d", dim);
            release();
            return FAILED;
        }
        output_elements *= static_cast<size_t>(dim);
    }
    if (output_elements != 3549U * 8U) {
        ERROR_LOG("YOLOX output element mismatch: actual=%zu expected=%u",
                  output_elements, 3549U * 8U);
        release();
        return FAILED;
    }
    INFO_LOG("[YOLOX] output anchors=3549 dim=8 classes=3");

    ret = personDetector_yolox.GetInputSizeByIndex(0, devBufferSize_);
    if (ret != SUCCESS || devBufferSize_ == 0) {
        ERROR_LOG("YOLOX invalid input buffer size: %zu", devBufferSize_);
        release();
        return FAILED;
    }

    const size_t expected_input_size =
        static_cast<size_t>(net_size_.width) * net_size_.height * 3;
    if (devBufferSize_ != expected_input_size) {
        ERROR_LOG("YOLOX input contract mismatch: model=%zu expected HWC U8x3=%zu",
                  devBufferSize_, expected_input_size);
        release();
        return FAILED;
    }

    aclError acl_ret = aclrtMalloc(&picDevBuffer_, devBufferSize_,
                                    ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
        ERROR_LOG("YOLOX malloc input failed, size=%zu errorCode=%d",
                  devBufferSize_, static_cast<int32_t>(acl_ret));
        picDevBuffer_ = nullptr;
        release();
        return FAILED;
    }

    ret = personDetector_yolox.CreateInput(picDevBuffer_, devBufferSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("YOLOX CreateInput failed");
        release();
        return FAILED;
    }

    ret = personDetector_yolox.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("YOLOX CreateOutput failed");
        release();
        return FAILED;
    }

    thresh_iou_ = config.threshold_iou;
    thresh_conf_ = 0.0f;
    initialized_ = true;
    INFO_LOG("[Detector_yolox] init success, input=%dx%d buffer=%zu",
             net_size_.width, net_size_.height, devBufferSize_);
    return SUCCESS;
}

void Detector_yolox::release()
{
    const bool had_resources = initialized_ || picDevBuffer_ != nullptr
        || !input_dim_.empty();
    initialized_ = false;

    personDetector_yolox.Release();
    if (picDevBuffer_ != nullptr) {
        const aclError ret = aclrtFree(picDevBuffer_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("[Detector_yolox] free input failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        picDevBuffer_ = nullptr;
    }

    preprocess_buffer_.release();
    devBufferSize_ = 0;
    input_dim_.clear();
    output_dim_.clear();
    net_size_ = cv::Size();
    thresh_conf_ = 0.0f;
    thresh_iou_ = 0.0f;
    if (had_resources) INFO_LOG("[Detector_yolox] release");
}

int Detector_yolox::run(const cv::Mat& image, float thresh_score,
                        std::vector<ObjDetInfo>& result)
{
    result.clear();
    if (!initialized_) {
        ERROR_LOG("Detector_yolox is not initialized");
        return FAILED;
    }
    if (image.empty()) {
        ERROR_LOG("Detector_yolox input image is empty");
        return FAILED;
    }
    if (!std::isfinite(thresh_score) || thresh_score <= 0.0f
        || thresh_score > 1.0f) {
        ERROR_LOG("Detector_yolox invalid confidence threshold: %f", thresh_score);
        return FAILED;
    }

    thresh_conf_ = thresh_score;
    fxprof::add_call(fxprof::Model::Detector);
    float scale_w = 1.0f;
    float scale_h = 1.0f;
    int ret = SUCCESS;

    {
        fxprof::ScopedPhase phase(fxprof::Model::Detector, fxprof::Phase::Pre);
        ret = preProcess(image, scale_w, scale_h);
    }
    if (ret != SUCCESS) return FAILED;

    {
        fxprof::ScopedPhase phase(fxprof::Model::Detector, fxprof::Phase::Infer);
        ret = inference();
    }
    if (ret != SUCCESS) return FAILED;

    {
        fxprof::ScopedPhase phase(fxprof::Model::Detector, fxprof::Phase::Post);
        ret = postProcess(image.size(), scale_w, scale_h, result);
    }
    return ret == SUCCESS ? SUCCESS : FAILED;
}

int Detector_yolox::preProcess(const cv::Mat& image,
                               float& scale_w, float& scale_h)
{
    if (fail_detector_yolox_stage_for_test("memcpy")) {
        ERROR_LOG("Detector_yolox memcpy failure injected for lifecycle test");
        return FAILED;
    }
    if (image.empty() || image.type() != CV_8UC3) {
        ERROR_LOG("YOLOX expects non-empty CV_8UC3 input, type=%d", image.type());
        return FAILED;
    }

    const int net_w = net_size_.width;
    const int net_h = net_size_.height;
    const float resize_ratio = std::min(
        static_cast<float>(net_h) / image.rows,
        static_cast<float>(net_w) / image.cols);
    if (!std::isfinite(resize_ratio) || resize_ratio <= 0.0f) return FAILED;

    const int new_w = static_cast<int>(image.cols * resize_ratio);
    const int new_h = static_cast<int>(image.rows * resize_ratio);
    if (new_w <= 0 || new_h <= 0 || new_w > net_w || new_h > net_h) {
        ERROR_LOG("YOLOX invalid resized size: %dx%d", new_w, new_h);
        return FAILED;
    }
    scale_w = 1.0f / resize_ratio;
    scale_h = 1.0f / resize_ratio;

    preprocess_buffer_.create(net_h, net_w, CV_8UC3);
    if (new_w < net_w) {
        preprocess_buffer_(cv::Rect(new_w, 0, net_w - new_w, new_h))
            .setTo(cv::Scalar(114, 114, 114));
    }
    if (new_h < net_h) {
        preprocess_buffer_(cv::Rect(0, new_h, net_w, net_h - new_h))
            .setTo(cv::Scalar(114, 114, 114));
    }

    cv::Mat resize_roi = preprocess_buffer_(cv::Rect(0, 0, new_w, new_h));
    cv::resize(image, resize_roi, cv::Size(new_w, new_h), 0.0, 0.0,
               cv::INTER_NEAREST);

    // 当前 OM 的 AIPP 接收 OpenCV BGR；不要在 CPU 重复交换通道。
    if (config_.bgr2rgb) {
        cv::cvtColor(preprocess_buffer_, preprocess_buffer_, cv::COLOR_BGR2RGB);
    }
    if (!preprocess_buffer_.isContinuous()) {
        ERROR_LOG("YOLOX preprocess buffer is not continuous");
        return FAILED;
    }

    const size_t data_len = preprocess_buffer_.total()
        * preprocess_buffer_.elemSize();
    if (data_len != devBufferSize_) {
        ERROR_LOG("YOLOX input length mismatch: data=%zu buffer=%zu",
                  data_len, devBufferSize_);
        return FAILED;
    }
    if (Utils::MemcpyHostToDevice(preprocess_buffer_.data, picDevBuffer_,
                                  data_len) != SUCCESS) {
        ERROR_LOG("YOLOX copy input to device failed");
        return FAILED;
    }
    return SUCCESS;
}

int Detector_yolox::inference()
{
    if (fail_detector_yolox_stage_for_test("execute")) {
        ERROR_LOG("Detector_yolox execute failure injected for lifecycle test");
        return FAILED;
    }
    if (personDetector_yolox.Execute() != SUCCESS) {
        ERROR_LOG("YOLOX execute inference failed");
        return FAILED;
    }
    return SUCCESS;
}

int Detector_yolox::postProcess(cv::Size src_img_size,
                                float scale_w, float scale_h,
                                std::vector<ObjDetInfo>& result)
{
    result.clear();
    if (src_img_size.width <= 0 || src_img_size.height <= 0
        || !std::isfinite(scale_w) || !std::isfinite(scale_h)
        || scale_w <= 0.0f || scale_h <= 0.0f) {
        ERROR_LOG("YOLOX invalid postprocess geometry");
        return FAILED;
    }

    std::vector<ObjDetInfo> bboxes;
    if (personDetector_yolox.OutputModelResultDetX(
            thresh_conf_, thresh_iou_, bboxes) != SUCCESS) {
        ERROR_LOG("YOLOX output parsing failed");
        return FAILED;
    }

    result.reserve(bboxes.size());
    for (const ObjDetInfo& src : bboxes) {
        // ModelProcess 的 YOLOX 中间框是普通 cv::Rect(x,y,w,h)。
        float x1 = src.box.x * scale_w;
        float y1 = src.box.y * scale_h;
        float x2 = (src.box.x + src.box.width) * scale_w;
        float y2 = (src.box.y + src.box.height) * scale_h;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(src_img_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(src_img_size.height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(src_img_size.width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(src_img_size.height)));
        if (x2 <= x1 || y2 <= y1) continue;

        int left = std::max(0, std::min(static_cast<int>(std::floor(x1)),
                                        src_img_size.width - 1));
        int top = std::max(0, std::min(static_cast<int>(std::floor(y1)),
                                       src_img_size.height - 1));
        int right = std::max(left + 1,
            std::min(static_cast<int>(std::ceil(x2)), src_img_size.width));
        int bottom = std::max(top + 1,
            std::min(static_cast<int>(std::ceil(y2)), src_img_size.height));

        ObjDetInfo dst;
        dst.label = src.label;
        dst.score = src.score;
        // 项目全局约定：cv::Rect 的 width/height 字段承载 x2/y2。
        dst.box = cv::Rect(left, top, right, bottom);
        result.push_back(dst);
    }
    return SUCCESS;
}

void Detector_yolox::yuv420spToRGB(unsigned char* yuv420sp,
                                   unsigned char* rgb,
                                   int width, int height)
{
    if (yuv420sp == nullptr || rgb == nullptr || width <= 0 || height <= 0) return;
    cv::Mat yuv(height + height / 2, width, CV_8UC1, yuv420sp);
    cv::Mat bgr(height, width, CV_8UC3, rgb);
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);
}
