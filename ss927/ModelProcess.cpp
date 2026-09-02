/**
* @file model_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "ModelProcess.h"
#include <iostream>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <algorithm>
#include <functional>
#include "utils.h"
#include "opencv2/opencv.hpp"

using namespace VISION_ENGINE;
using namespace std;

namespace {
constexpr int kYoloxInputW = 416;
constexpr int kYoloxInputH = 416;
constexpr int kYoloxNumClasses = 3;
constexpr int kYoloxOutputDim = 5 + kYoloxNumClasses;
constexpr size_t kYoloxPreNmsTopK = 200;
constexpr size_t kYoloxMaxDetPerClass = 100;

struct GridAndStride {
    int grid_x;
    int grid_y;
    int stride;
};

struct YoloxProposal {
    float x1;
    float y1;
    float x2;
    float y2;
    float area;
    float score;
    int label;
};

std::vector<GridAndStride> generate_yolox_grids()
{
    static const int strides[] = {8, 16, 32};
    std::vector<GridAndStride> grids;
    grids.reserve(3549);
    for (int stride : strides) {
        const int grid_w = kYoloxInputW / stride;
        const int grid_h = kYoloxInputH / stride;
        for (int gy = 0; gy < grid_h; ++gy) {
            for (int gx = 0; gx < grid_w; ++gx) {
                grids.push_back({gx, gy, stride});
            }
        }
    }
    return grids;
}

void sort_and_limit_yolox_proposals(std::vector<YoloxProposal>& proposals)
{
    const auto score_compare = [](const YoloxProposal& a,
                                  const YoloxProposal& b) {
        return a.score > b.score;
    };
    if (proposals.size() > kYoloxPreNmsTopK) {
        auto nth = proposals.begin()
            + static_cast<std::ptrdiff_t>(kYoloxPreNmsTopK);
        std::nth_element(proposals.begin(), nth, proposals.end(), score_compare);
        proposals.resize(kYoloxPreNmsTopK);
    }
    std::sort(proposals.begin(), proposals.end(), score_compare);
}

void nms_yolox_single_class(const std::vector<YoloxProposal>& proposals,
                            float thresh_iou,
                            std::vector<YoloxProposal>& keep)
{
    keep.clear();
    if (proposals.empty()) return;

    std::vector<uint8_t> removed(proposals.size(), 0);
    keep.reserve(std::min(proposals.size(), kYoloxMaxDetPerClass));
    for (size_t i = 0; i < proposals.size(); ++i) {
        if (removed[i]) continue;
        const YoloxProposal& a = proposals[i];
        keep.push_back(a);
        if (keep.size() >= kYoloxMaxDetPerClass) break;

        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (removed[j]) continue;
            const YoloxProposal& b = proposals[j];
            if (b.x1 >= a.x2 || b.x2 <= a.x1
                || b.y1 >= a.y2 || b.y2 <= a.y1) {
                continue;
            }
            const float inter_w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
            const float inter_h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
            if (inter_w <= 0.0f || inter_h <= 0.0f) continue;
            const float inter_area = inter_w * inter_h;
            const float union_area = a.area + b.area - inter_area;
            if (union_area > 0.0f && inter_area / union_area >= thresh_iou) {
                removed[j] = 1;
            }
        }
    }
}

bool convert_yolox_proposal(const YoloxProposal& proposal,
                            VISION_ENGINE::ObjDetInfo& obj)
{
    const int left = static_cast<int>(std::floor(proposal.x1));
    const int top = static_cast<int>(std::floor(proposal.y1));
    const int right = static_cast<int>(std::ceil(proposal.x2));
    const int bottom = static_cast<int>(std::ceil(proposal.y2));
    if (right <= left || bottom <= top) return false;

    obj.label = proposal.label;
    obj.score = proposal.score;
    // ModelProcess 内部临时结果使用标准 cv::Rect(x,y,w,h)。
    obj.box = cv::Rect(left, top, right - left, bottom - top);
    return true;
}

void log_cleanup_error(const char* operation, aclError ret)
{
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("%s failed, errorCode=%d", operation,
                  static_cast<int32_t>(ret));
    }
}
} // namespace

ModelProcess::ModelProcess() :modelId_(0), modelWorkSize_(0), modelWeightSize_(0), modelWorkPtr_(nullptr),
modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr)
{
}

ModelProcess::~ModelProcess()
{
    Release();
}

void ModelProcess::Release()
{
    DestroyInput();
    DestroyOutput();
    UnloadModel();
    DestroyModelDesc();
}

Result ModelProcess::LoadModel(const char* modelPath)
{
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }
    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
            modelWorkSize_, static_cast<int32_t>(ret));
        return FAILED;
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
            modelWeightSize_, static_cast<int32_t>(ret));
        UnloadModel();
        return FAILED;
    }

    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
        modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
            modelPath, static_cast<int32_t>(ret));
        UnloadModel();
        return FAILED;
    }

    loadFlag_ = true;
    // INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::LoadModelFromMem(const uchar* model_buf, size_t model_size)
{
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }
    aclError ret = aclmdlQuerySizeFromMem(model_buf, model_size, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("query model failed, errorCode is %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
            modelWorkSize_, static_cast<int32_t>(ret));
        return FAILED;
    }

    // using ACL_MEM_MALLOC_HUGE_FIRST to malloc memory, huge memory is preferred to use
    // and huge memory can improve performance.
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
            modelWeightSize_, static_cast<int32_t>(ret));
        UnloadModel();
        return FAILED;
    }

    ret = aclmdlLoadFromMemWithMem(model_buf, model_size, &modelId_, modelWorkPtr_,
        modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, errorCode is %d", static_cast<int32_t>(ret));
        UnloadModel();
        return FAILED;
    }

    loadFlag_ = true;
    // INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}

Result ModelProcess::CreateModelDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        DestroyModelDesc();
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyModelDesc()
{
    if (modelDesc_ != nullptr) {
        aclError ret = aclmdlDestroyDesc(modelDesc_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy model description failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        } else {
            INFO_LOG("destroy model description success");
        }
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::GetInputDimsByIndex(const size_t index, vector<int>& dims)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    aclmdlIODims re_dims;
    aclError ret = aclmdlGetInputDims(modelDesc_, index, &re_dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get input dims failed, index=%zu, errorCode=%d",
                  index, static_cast<int32_t>(ret));
        dims.clear();
        return FAILED;
    }

    dims.resize(re_dims.dimCount);
    for (int i = 0; i < re_dims.dimCount; ++i)
        dims[i] = re_dims.dims[i];

    return SUCCESS;
}

Result ModelProcess::GetOutputDimsByIndex(const size_t index, vector<int>& dims)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    aclmdlIODims re_dims;
    aclError ret = aclmdlGetOutputDims(modelDesc_, index, &re_dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get output dims failed, index=%zu, errorCode=%d",
                  index, static_cast<int32_t>(ret));
        dims.clear();
        return FAILED;
    }

    dims.resize(re_dims.dimCount);
    for (int i = 0; i < re_dims.dimCount; ++i)
        dims[i] = re_dims.dims[i];

    return SUCCESS;
}

Result ModelProcess::GetInputSizeByIndex(const size_t index, size_t& inputSize)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    inputSize = aclmdlGetInputSizeByIndex(modelDesc_, index);
    return SUCCESS;
}

Result ModelProcess::CreateInput(void* inputDataBuffer, size_t bufferSize)
{
    // om used in this sample has only one input
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, 0);
    if (bufferSize != modelInputSize) {
        ERROR_LOG("input image size[%zu] is not equal to model input size[%zu]", bufferSize, modelInputSize);
        return FAILED;
    }

    if (input_ != nullptr) {
        ERROR_LOG("model input already exists");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }

    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        log_cleanup_error("rollback input dataset destroy",
                          aclmdlDestroyDataset(input_));
        input_ = nullptr;
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        log_cleanup_error("rollback input data buffer destroy",
                          aclDestroyDataBuffer(inputData));
        log_cleanup_error("rollback input dataset destroy",
                          aclmdlDestroyDataset(input_));
        input_ = nullptr;
        return FAILED;
    }
    INFO_LOG("create model input success");

    return SUCCESS;
}

void ModelProcess::DestroyInput()
{
    if (input_ != nullptr) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(input_, i);
            if (dataBuffer != nullptr) {
                aclError ret = aclDestroyDataBuffer(dataBuffer);
                if (ret != ACL_SUCCESS) {
                    ERROR_LOG("destroy model input data buffer %zu failed, errorCode=%d",
                              i, static_cast<int32_t>(ret));
                }
            }
        }
        aclError ret = aclmdlDestroyDataset(input_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy model input dataset failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        input_ = nullptr;
        INFO_LOG("destroy model input success");
    }

    // 释放动态 batch 创建时分配的额外设备缓冲区
    for (void* buf : dynAllocatedBuffers_) {
        if (buf) log_cleanup_error("free dynamic input buffer", aclrtFree(buf));
    }
    dynAllocatedBuffers_.clear();

}

Result ModelProcess::CreateOutput()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    if (output_ != nullptr) {
        ERROR_LOG("model output already exists");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);

        void* outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            DestroyOutput();
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            log_cleanup_error("rollback output buffer free", aclrtFree(outputBuffer));
            DestroyOutput();
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
                static_cast<int32_t>(ret));
            log_cleanup_error("rollback output data buffer destroy",
                              aclDestroyDataBuffer(outputData));
            log_cleanup_error("rollback output buffer free", aclrtFree(outputBuffer));
            DestroyOutput();
            return FAILED;
        }
    }

    INFO_LOG("create model output success");

    return SUCCESS;
}

void ModelProcess::DumpModelOutputResult()
{
    if (output_ == nullptr) {
        ERROR_LOG("no model output to dump");
        return;
    }

    stringstream ss;
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);
    static int executeNum = 0;
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << ++executeNum << "_" << i << ".bin";
        string outputFileName = ss.str();
        FILE* outputFile = fopen(outputFileName.c_str(), "wb");
        if (outputFile != nullptr) {
            // get model output data
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(dataBuffer);
            uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

            if (data == nullptr || len == 0) {
                ERROR_LOG("model output buffer is invalid");
                fclose(outputFile);
                return;
            }

            void* outHostData = nullptr;
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS || outHostData == nullptr) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len, static_cast<int32_t>(ret));
                fclose(outputFile);
                return;
            }

            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("device-to-host memcpy failed, errorCode[%d]",
                          static_cast<int32_t>(ret));
                log_cleanup_error("rollback host output buffer free",
                                  aclrtFreeHost(outHostData));
                fclose(outputFile);
                return;
            }

            fwrite(outHostData, 1, len, outputFile);
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]",
                          static_cast<int32_t>(ret));
                fclose(outputFile);
                return;
            }
            fclose(outputFile);
        }
        else {
            ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
            return;
        }
    }

    INFO_LOG("dump data success");
    return;
}

Result ModelProcess::OutputModelResult(vector<vector<float>>& result)
{
    result.clear();
    if (output_ == nullptr) {
        ERROR_LOG("model output dataset is null");
        return FAILED;
    }

    result.resize(aclmdlGetDatasetNumBuffers(output_));

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        if (dataBuffer == nullptr) {
            ERROR_LOG("model output data buffer %zu is null", i);
            result.clear();
            return FAILED;
        }
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        if (data == nullptr || len == 0 || len % sizeof(float) != 0) {
            ERROR_LOG("model output buffer %zu invalid, len=%u", i, len);
            result.clear();
            return FAILED;
        }
        result[i].resize(len / sizeof(float));
        aclError ret = aclrtMemcpy(result[i].data(), len, data, len,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("copy model output %zu to host failed, errorCode=%d",
                      i, static_cast<int32_t>(ret));
            result.clear();
            return FAILED;
        }
    }

    return SUCCESS;
}

Result ModelProcess::OutputModelResultDet(float thresh_conf, float thresh_iou,
                                          vector<VISION_ENGINE::ObjDetInfo>& result)
{
    result.clear();
    vector<vector<float>> modelOutput;
    if (OutputModelResult(modelOutput) != SUCCESS) return FAILED;

    for (size_t output_idx = 0; output_idx < modelOutput.size(); ++output_idx) {
        uint32_t len = static_cast<uint32_t>(modelOutput[output_idx].size()
                                             * sizeof(float));
        float* outData = modelOutput[output_idx].data();

        //--------------------------------
        //			  decode
        //--------------------------------
        // Resizing factor.
        int dim_anchor = 7;
        int num_anchor = 10647;
        int num_cls = dim_anchor - 5;
        // A11 崩溃防线：旧实现完全忽略 len，按硬编码 10647×7 裸走输出缓冲。
        // 一旦部署的 .om 输出更小（FP16 输出 / 不同输入分辨率 / 图内融合 NMS），
        // 每帧越界读 >100KB → 首帧 SIGSEGV → 进程死 → 看门狗循环重启整机。
        // 以真实缓冲大小为上界（模型匹配硬编码时行为完全不变）。
        if (outData == nullptr || len < (uint32_t)(dim_anchor * sizeof(float))) {
            ERROR_LOG("det output buffer invalid: len=%u", (unsigned)len);
            return FAILED;
        }
        {
            int max_anchor = (int)(len / (dim_anchor * sizeof(float)));
            if (max_anchor < num_anchor) {
                ERROR_LOG("det output smaller than expected: len=%u -> clamp anchors %d -> %d"
                          " (check .om output shape/dtype vs hardcoded %dx%d)",
                          (unsigned)len, num_anchor, max_anchor, num_anchor, dim_anchor);
                num_anchor = max_anchor;
            }
        }
        for (int i = 0; i < num_anchor; ++i) {
            float confidence = outData[4];
            if (confidence >= thresh_conf)
            {
                 float* classesScores = outData + 5;

                 int max_label = -1;
                 float max_score = 0;
                //int max_label = 0;
                //float max_score = confidence;
                 for(int j = 0; j < num_cls; ++j){
                     if (max_score <= classesScores[j]){
                         max_score = classesScores[j];
                         max_label = j;
                     }
                 }
         	

		 max_score = max_score * confidence;

                // Discard bad ModelProcesss and continue.
                if (max_score >= thresh_conf)
                {
                VISION_ENGINE::ObjDetInfo obj;
                obj.label = max_label;

                // Center.
                float cx = outData[0];
                float cy = outData[1];
                // Box dimension.
                float w = outData[2];
                float h = outData[3];
                // Bounding box coordinates.
                obj.box = cv::Rect(
                    cv::Point(cx - 0.5 * w, cy - 0.5 * h),
                    cv::Point(cx + 0.5 * w, cy + 0.5 * h));
                // score
                obj.score = max_score;

                result.push_back(obj);
                }
            }

            outData += dim_anchor;
        }



        //--------------------------------
        //			   nms
        //--------------------------------
        std::sort(result.begin(), result.end(),
            [](const VISION_ENGINE::ObjDetInfo& a, const VISION_ENGINE::ObjDetInfo& b)
            {
                return a.score > b.score;
            });

        vector<float>vArea(result.size());
        for (int i = 0; i < result.size(); ++i)
            vArea[i] = result[i].box.width * result[i].box.height;

        for (int i = 0; i < int(result.size()); ++i) {
            for (int j = i + 1; j < int(result.size());) {
                if (result[i].label != result[j].label) {
                    j++;
                    continue;
                }
                float xx1 = MAX(result[i].box.x, result[j].box.x);
                float yy1 = MAX(result[i].box.y, result[j].box.y);
                float xx2 = MIN(result[i].box.br().x, result[j].box.br().x);
                float yy2 = MIN(result[i].box.br().y, result[j].box.br().y);
                float w = MAX(float(0), xx2 - xx1 + 1);
                float h = MAX(float(0), yy2 - yy1 + 1);
                float inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= thresh_iou) {
                    result.erase(result.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else
                    j++;
            }
        }
    }

    return SUCCESS;
}

Result ModelProcess::OutputModelResultDetX(
    float thresh_conf, float thresh_iou,
    vector<VISION_ENGINE::ObjDetInfo>& result)
{
    result.clear();
    if (!std::isfinite(thresh_conf) || !std::isfinite(thresh_iou)
        || thresh_conf <= 0.0f || thresh_conf > 1.0f
        || thresh_iou <= 0.0f || thresh_iou > 1.0f) {
        ERROR_LOG("YOLOX invalid threshold: conf=%f iou=%f",
                  thresh_conf, thresh_iou);
        return FAILED;
    }

    static const std::vector<GridAndStride> grid_strides =
        generate_yolox_grids();
    const int expected_anchors = static_cast<int>(grid_strides.size());
    if (expected_anchors != 3549) {
        ERROR_LOG("YOLOX grid count invalid: %d", expected_anchors);
        return FAILED;
    }

    // SS927 现有 ModelProcess 契约统一把 ACL output 显式 D2H 后再解析。
    vector<vector<float>> model_output;
    if (OutputModelResult(model_output) != SUCCESS) return FAILED;
    if (model_output.size() != 1) {
        ERROR_LOG("YOLOX output count invalid: %zu, expected 1",
                  model_output.size());
        return FAILED;
    }

    const size_t expected_floats = static_cast<size_t>(expected_anchors)
        * static_cast<size_t>(kYoloxOutputDim);
    if (model_output[0].size() < expected_floats) {
        ERROR_LOG("YOLOX output too small: actual=%zu expected=%zu floats",
                  model_output[0].size(), expected_floats);
        return FAILED;
    }
    if (model_output[0].size() != expected_floats) {
        INFO_LOG("YOLOX output has trailing data: actual=%zu expected=%zu floats",
                 model_output[0].size(), expected_floats);
    }

    const float* output_data = model_output[0].data();
    std::array<std::vector<YoloxProposal>, kYoloxNumClasses>
        class_proposals;
    for (auto& proposals : class_proposals) proposals.reserve(128);

#ifdef YOLOX_POSTPROCESS_PROFILE
    const auto time_begin = std::chrono::steady_clock::now();
#endif

    for (int anchor_idx = 0; anchor_idx < expected_anchors; ++anchor_idx) {
        const float* pred = output_data
            + static_cast<size_t>(anchor_idx) * kYoloxOutputDim;
        const float objectness = pred[4];
        if (!std::isfinite(objectness) || objectness < thresh_conf) continue;

        int max_label = -1;
        float max_class_score = -1.0f;
        for (int cls = 0; cls < kYoloxNumClasses; ++cls) {
            const float cls_score = pred[5 + cls];
            if (std::isfinite(cls_score) && cls_score > max_class_score) {
                max_class_score = cls_score;
                max_label = cls;
            }
        }
        if (max_label < 0) continue;
        const float score = objectness * max_class_score;
        if (!std::isfinite(score) || score < thresh_conf) continue;

        const float raw_x = pred[0];
        const float raw_y = pred[1];
        const float raw_w = pred[2];
        const float raw_h = pred[3];
        if (!std::isfinite(raw_x) || !std::isfinite(raw_y)
            || !std::isfinite(raw_w) || !std::isfinite(raw_h)) {
            continue;
        }

        const GridAndStride& gs = grid_strides[anchor_idx];
        const float stride = static_cast<float>(gs.stride);
        const float cx = (raw_x + gs.grid_x) * stride;
        const float cy = (raw_y + gs.grid_y) * stride;
        const float box_w = std::exp(raw_w) * stride;
        const float box_h = std::exp(raw_h) * stride;
        if (!std::isfinite(cx) || !std::isfinite(cy)
            || !std::isfinite(box_w) || !std::isfinite(box_h)
            || box_w <= 0.0f || box_h <= 0.0f) {
            continue;
        }

        const float x1 = std::max(0.0f, std::min(
            cx - box_w * 0.5f, static_cast<float>(kYoloxInputW)));
        const float y1 = std::max(0.0f, std::min(
            cy - box_h * 0.5f, static_cast<float>(kYoloxInputH)));
        const float x2 = std::max(0.0f, std::min(
            cx + box_w * 0.5f, static_cast<float>(kYoloxInputW)));
        const float y2 = std::max(0.0f, std::min(
            cy + box_h * 0.5f, static_cast<float>(kYoloxInputH)));
        if (x2 <= x1 || y2 <= y1) continue;

        class_proposals[max_label].push_back({
            x1, y1, x2, y2, (x2 - x1) * (y2 - y1), score, max_label});
    }

#ifdef YOLOX_POSTPROCESS_PROFILE
    const auto time_decode = std::chrono::steady_clock::now();
#endif

    size_t total_proposals = 0;
    for (auto& proposals : class_proposals) {
        total_proposals += proposals.size();
        sort_and_limit_yolox_proposals(proposals);
    }
    result.reserve(std::min<size_t>(
        total_proposals, kYoloxMaxDetPerClass * kYoloxNumClasses));

#ifdef YOLOX_POSTPROCESS_PROFILE
    const auto time_sort = std::chrono::steady_clock::now();
#endif

    std::vector<YoloxProposal> keep;
    keep.reserve(kYoloxPreNmsTopK);
    for (int cls = 0; cls < kYoloxNumClasses; ++cls) {
        nms_yolox_single_class(class_proposals[cls], thresh_iou, keep);
        for (const YoloxProposal& proposal : keep) {
            VISION_ENGINE::ObjDetInfo obj;
            if (convert_yolox_proposal(proposal, obj)) result.push_back(obj);
        }
    }

    std::sort(result.begin(), result.end(),
        [](const VISION_ENGINE::ObjDetInfo& a,
           const VISION_ENGINE::ObjDetInfo& b) {
            return a.score > b.score;
        });

#ifdef YOLOX_POSTPROCESS_PROFILE
    const auto time_end = std::chrono::steady_clock::now();
    INFO_LOG("[YOLOX_POST] decode=%lld us sort=%lld us nms=%lld us total=%lld us "
             "proposal=%zu face=%zu body=%zu head=%zu result=%zu",
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_decode - time_begin).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_sort - time_decode).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - time_sort).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - time_begin).count()),
        total_proposals, class_proposals[0].size(),
        class_proposals[1].size(), class_proposals[2].size(), result.size());
#endif
    return SUCCESS;
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        if (dataBuffer != nullptr) {
            void* data = aclGetDataBufferAddr(dataBuffer);
            aclError ret = aclDestroyDataBuffer(dataBuffer);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("destroy model output data buffer %zu failed, errorCode=%d",
                          i, static_cast<int32_t>(ret));
            }
            if (data != nullptr) {
                ret = aclrtFree(data);
                if (ret != ACL_SUCCESS) {
                    ERROR_LOG("free model output %zu failed, errorCode=%d",
                              i, static_cast<int32_t>(ret));
                }
            }
        }
    }

    aclError ret = aclmdlDestroyDataset(output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("destroy model output dataset failed, errorCode=%d",
                  static_cast<int32_t>(ret));
    }
    output_ = nullptr;
    INFO_LOG("destroy model output success");
}

Result ModelProcess::Execute()
{
    if (!loadFlag_ || input_ == nullptr || output_ == nullptr) {
        ERROR_LOG("model execute rejected: loaded=%d input=%p output=%p",
                  loadFlag_ ? 1 : 0, static_cast<void*>(input_),
                  static_cast<void*>(output_));
        return FAILED;
    }
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    // INFO_LOG("model execute success");
    return SUCCESS;
}

void ModelProcess::UnloadModel()
{
    const uint32_t old_model_id = modelId_;
    if (loadFlag_) {
        aclError ret = aclmdlUnload(modelId_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
                modelId_, static_cast<int32_t>(ret));
        } else {
            INFO_LOG("unload model success, modelId is %u", modelId_);
        }
    }

    DestroyModelDesc();

    if (modelWorkPtr_ != nullptr) {
        aclError ret = aclrtFree(modelWorkPtr_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("free model work memory failed, modelId=%u, errorCode=%d",
                      old_model_id, static_cast<int32_t>(ret));
        }
        modelWorkPtr_ = nullptr;
        modelWorkSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        aclError ret = aclrtFree(modelWeightPtr_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("free model weight memory failed, modelId=%u, errorCode=%d",
                      old_model_id, static_cast<int32_t>(ret));
        }
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    modelId_ = 0;
    modelWorkSize_ = 0;
    modelWeightSize_ = 0;
}

// ════════════════════════════════════════════════════════════
// 动态 Batch 支持
// ════════════════════════════════════════════════════════════

Result ModelProcess::CreateInputDynamicBatch(void* inputDataBuffer, size_t bufferSize)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }

    if (input_ != nullptr) {
        ERROR_LOG("model input already exists");
        return FAILED;
    }

    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }

    // ── 添加主数据缓冲区 (index 0) ──
    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        DestroyInput();
        return FAILED;
    }
    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", (int)ret);
        log_cleanup_error("rollback dynamic input data buffer destroy",
                          aclDestroyDataBuffer(inputData));
        DestroyInput();
        return FAILED;
    }

    // ── 添加动态 batch 参数缓冲区 ──
    size_t dynIdx = 0;
    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &dynIdx);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("model does not support dynamic batch, errorCode is %d", (int)ret);
        DestroyInput();
        return FAILED;
    }

    size_t dynSize = aclmdlGetInputSizeByIndex(modelDesc_, dynIdx);
    void* dynBuf = nullptr;
    ret = aclrtMalloc(&dynBuf, dynSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc dynamic batch buffer failed, size=%zu, errorCode=%d",
                  dynSize, (int)ret);
        DestroyInput();
        return FAILED;
    }

    aclDataBuffer* dynData = aclCreateDataBuffer(dynBuf, dynSize);
    if (dynData == nullptr) {
        ERROR_LOG("create dynamic batch data buffer failed");
        log_cleanup_error("rollback dynamic input buffer free", aclrtFree(dynBuf));
        DestroyInput();
        return FAILED;
    }

    ret = aclmdlAddDatasetBuffer(input_, dynData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add dynamic batch buffer failed, errorCode=%d", (int)ret);
        log_cleanup_error("rollback dynamic data buffer destroy",
                          aclDestroyDataBuffer(dynData));
        log_cleanup_error("rollback dynamic input buffer free", aclrtFree(dynBuf));
        DestroyInput();
        return FAILED;
    }

    dynAllocatedBuffers_.push_back(dynBuf);
    INFO_LOG("create model input with dynamic batch success (dynIdx=%zu)", dynIdx);
    return SUCCESS;
}

Result ModelProcess::SetDynamicBatchSize(size_t dynIdx, uint32_t batchSize)
{
    if (input_ == nullptr || !loadFlag_) {
        ERROR_LOG("model not ready, cannot set dynamic batch size");
        return FAILED;
    }

    aclError ret = aclmdlSetDynamicBatchSize(modelId_, input_, dynIdx, batchSize);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("set dynamic batch size failed, batch=%u, errorCode=%d",
                  batchSize, (int)ret);
        return FAILED;
    }
    return SUCCESS;
}
