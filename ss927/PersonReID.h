
#ifndef PERSON_REID_H
#define PERSON_REID_H

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "ModelProcess.h"
#include "utils.h"

namespace VISION_ENGINE {

/**
    * @brief ReID特征向量
    */
struct ReIDFeature {
    std::vector<float> feature;  // 特征向量
    float norm;                  // 向量模长
};

/**
    * @brief PersonReID配置结构体
    */
struct ReIDConfig {
    size_t sizeModel;            // 模型大小
    uint8_t* bufferModel;        // 模型缓冲区
    cv::Size inputSize;          // 输入尺寸 (宽, 高)
    std::vector<float> means;    // 均值 [R, G, B]
    std::vector<float> stds;     // 标准差 [R, G, B]
    bool bgr2rgb;               // 是否BGR转RGB
    int deviceId;               // 设备ID
};

/**
    * @brief PersonReID推理类
    *
    * 基于昇腾平台实现行人重识别特征提取（单 batch）
    */
class CPersonReID {
public:
    CPersonReID();
    ~CPersonReID();

    /**
        * @brief 初始化模型
        * @param config 配置参数
        * @return int 0成功，其他失败
        */
    int init(const ReIDConfig& config);

    /**
        * @brief 执行ReID推理（单张）
        * @param image 输入图像
        * @param box 行人检测框（cv::Rect 语义为 x1,y1,x2,y2）
        * @param feature 输出特征向量
        * @param needNorm 是否需要L2归一化
        * @return int 0成功，其他失败
        */
    int run(const cv::Mat& image, const cv::Rect& box,
        cv::Mat& feature, bool needNorm = true);

    /**
        * @brief 释放资源
        */
    void release();

    /**
        * @brief 计算两个特征向量的相似度
        * @param feat1 特征向量1
        * @param feat2 特征向量2
        * @return float 余弦相似度
        */
    static float computeSimilarity(const std::vector<float>& feat1,
        const std::vector<float>& feat2);

    /**
        * @brief 特征向量L2归一化
        * @param feature 输入特征向量
        * @param normFeature 输出归一化特征向量
        */
    static void normalizeFeature(std::vector<float>& feature);

private:
    // 预处理：裁剪+缩放+颜色转换 + 拷贝到设备
    int preProcess(const cv::Mat& src, const cv::Rect& box, cv::Mat& dst);

    // 模型推理
    int inference(const cv::Mat& image);

    // 后处理：从模型输出提取 L2 归一化特征
    int postProcess(cv::Mat& feature, bool needNorm);

private:
    bool bInit_;                 // 是否已初始化
    ReIDConfig config_;          // 配置参数
    cv::Size netSize_;           // 网络输入尺寸

    ModelProcess modelProcess_;  // 模型处理对象
    void* picDevBuffer_;         // 设备端图像缓冲区
    size_t devBufferSize_;       // 设备缓冲区大小

    std::vector<int> inputDims_; // 输入维度

    };

}

#endif // PERSON_REID_H
