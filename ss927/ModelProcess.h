/**
* @file model_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>
#include "types.h"
#include "acl/acl.h"

using namespace VISION_ENGINE;

class ModelProcess {
public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    virtual ~ModelProcess();

    // 释放本对象拥有的全部模型、描述、dataset 和输出/工作区设备内存。
    // 不释放 CreateInput() 传入的主输入 buffer；该 buffer 归模型 wrapper。
    void Release();

    /**
    * @brief load model
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModel(const char* modelPath);

    /**
    * @brief unload model
    */
    void UnloadModel();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateModelDesc();

    /**
    * @brief destroy desc
    */
    void DestroyModelDesc();

    /**
    * @get input size by index
    * @param [in] index: input index
    * @param [out] inputSize: input size of index
    * @return result
    */
    Result GetInputSizeByIndex(const size_t index, size_t& inputSize);

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result CreateInput(void* inputDataBuffer, size_t bufferSize);

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief dump model output result to file
    */
    void DumpModelOutputResult();

    /**
    * @brief print model output result
    */
    Result OutputModelResult(std::vector<std::vector<float>>& result);
    Result OutputModelResultDet(float thresh_conf, float thresh_iou, std::vector<VISION_ENGINE::ObjDetInfo>& result);
    Result OutputModelResultDetX(float thresh_conf, float thresh_iou,
                                 std::vector<VISION_ENGINE::ObjDetInfo>& result);

    Result LoadModelFromMem(const uchar* model_buf, size_t model_size);
    Result GetInputDimsByIndex(const size_t index, std::vector<int>& dims);
    Result GetOutputDimsByIndex(const size_t index, std::vector<int>& dims);

    // ── 动态 Batch 支持 ──────────────────────────────────
    // 创建输入（同时添加动态batch参数缓冲区）
    Result CreateInputDynamicBatch(void* inputDataBuffer, size_t bufferSize);

    // 设置本次推理的实际 batch size（每次 Execute 前调用）
    Result SetDynamicBatchSize(size_t dynIdx, uint32_t batchSize);

    // 访问器
    uint32_t GetModelId() const { return modelId_; }
    aclmdlDesc* GetModelDesc() const { return modelDesc_; }

private:
    uint32_t modelId_;
    size_t modelWorkSize_; // model work memory buffer size
    size_t modelWeightSize_; // model weight memory buffer size
    void* modelWorkPtr_; // model work memory buffer
    void* modelWeightPtr_; // model weight memory buffer
    bool loadFlag_;  // model load flag
    aclmdlDesc* modelDesc_;
    aclmdlDataset* input_;
    aclmdlDataset* output_;
    std::vector<void*> dynAllocatedBuffers_;  // 动态batch创建时分配的额外设备缓冲区
};

