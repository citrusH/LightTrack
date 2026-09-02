/**
 * @file FaceRecognition.h
 *
 * 人脸识别模型推理类（昇腾平台版本）
 */

#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
#include "ModelProcess.h"
#include "acl/acl.h"
#include "acl/acl_base.h"



namespace VISION_ENGINE {

    /**
     * @brief 人脸信息结构体
     */
    struct FaceInfo {
        std::vector<float> kps;           // 5个关键点，格式: [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5]
        std::vector<float> embedding;     // 人脸特征向量
    };

    /**
     * @brief 人脸识别配置结构体
     */
    struct FaceRecogConfig {
        size_t sizeModel;                 // 模型大小
        uint8_t* bufferModel;             // 模型缓冲区
        cv::Size inputSize;               // 输入尺寸
        std::vector<float> means;         // 均值
        std::vector<float> stds;          // 标准差
        bool bgr2rgb;                     // 是否BGR转RGB
        int deviceId;                     // 设备ID
        int imageSize;                    // 人脸对齐尺寸
        bool find_sub;                    // 是否需要减均值
        bool find_mul;                    // 是否需要乘系数
    };

    /**
     * @brief 人脸识别推理类
     */
    class CFaceRecognition {
    public:
        CFaceRecognition();
        ~CFaceRecognition();

        /**
         * @brief 初始化模型
         * @param config 配置参数
         * @return int 0成功，其他失败
         */
        int init(const FaceRecogConfig& config);

        /**
         * @brief 根据人脸关键点对齐并提取特征
         * @param img 输入图像
         * @param face 人脸信息（包含关键点，返回特征）
         * @return int 0成功，其他失败
         */
        int get(const cv::Mat& img, FaceInfo& face);

        /**
         * @brief 从对齐后的人脸图像提取特征
         * @param aligned_face 对齐后的人脸图像
         * @return std::vector<float> 特征向量
         */
        std::vector<float> get_feat(const cv::Mat& aligned_face);

        /**
         * @brief 计算两个特征向量的相似度
         * @param feat1 特征向量1
         * @param feat2 特征向量2
         * @return float 余弦相似度
         */
        float compute_sim(const std::vector<float>& feat1, const std::vector<float>& feat2);

        /**
         * @brief 人脸对齐
         * @param img 输入图像
         * @param landmark 关键点
         * @param image_size 输出尺寸
         * @return cv::Mat 对齐后的人脸图像
         */
        cv::Mat norm_crop(const cv::Mat& img, const std::vector<float>& landmark, int image_size);

        /**
         * @brief 释放资源
         */
        void release();

    private:
        /**
         * @brief 预处理
         * @param src 源图像
         * @param dst 处理后的图像
         * @return int 0成功，其他失败
         */
        int preProcess(const cv::Mat& src, cv::Mat& dst);

        /**
         * @brief 模型推理
         * @param img 输入图像
         * @param feature 输出特征向量
         * @return int 0成功，其他失败
         */
        int inference(const cv::Mat& img, std::vector<float>& feature);

        /**
         * @brief 后处理
         * @param feature 输出特征向量
         * @return int 0成功，其他失败
         */
        int postProcess(std::vector<float>& feature);

    private:
        bool bInit_;                       // 是否已初始化
        FaceRecogConfig config_;           // 配置参数
        cv::Size netSize_;                 // 网络输入尺寸

        ModelProcess modelProcess_;        // 模型处理对象
        void* picDevBuffer_;               // 设备端图像缓冲区
        size_t devBufferSize_;             // 设备缓冲区大小

        std::vector<int> inputDims_;       // 输入维度
        std::vector<int> outputDims_;      // 输出维度

        float input_mean_;                 // 输入均值
        float input_std_;                  // 输入标准差
        std::vector<float> means_;         // 各通道均值
        std::vector<float> stds_;          // 各通道标准差

    };

} // namespace VISION_ENGINE

#endif // FACE_RECOGNITION_H
