/**
 * @file FaceKeypoint106.h
 *
 * 人脸关键点检测模型推理类（昇腾平台版本）
 * 支持106个关键点检测
 */

#ifndef FACE_KEYPOINT_106_H
#define FACE_KEYPOINT_106_H

#include <vector>
#include <string>
#include <memory>
#include "opencv2/opencv.hpp"
#include "ModelProcess.h"
#include "acl/acl.h"
#include "acl/acl_base.h"



namespace VISION_ENGINE {

    /**
     * @brief 人脸关键点结构体
     */
    struct FaceKeypoint {
        float x;    // x坐标
        float y;    // y坐标
    };

    /**
     * @brief 人脸关键点检测结果结构体
     */
    struct FaceKeypointResult {
        std::vector<FaceKeypoint> points;  // 关键点列表
        cv::Rect face_rect;                // 人脸框
    };

    /**
     * @brief 人脸关键点配置结构体
     */
    struct FaceKeypointConfig {
        size_t sizeModel;                  // 模型大小
        uint8_t* bufferModel;              // 模型缓冲区
        cv::Size inputSize;                // 输入尺寸
        float inputMean;                   // 输入均值
        float inputStd;                    // 输入标准差
        bool bgr2rgb;                      // 是否BGR转RGB
        int deviceId;                      // 设备ID
        int numPoints;                     // 关键点数量
    };

    /**
     * @brief 人脸关键点检测推理类
     */
    class CFaceKeypoint106 {
    public:
        CFaceKeypoint106();
        ~CFaceKeypoint106();

        /**
         * @brief 初始化模型
         * @param config 配置参数
         * @return int 0成功，其他失败
         */
        int init(const FaceKeypointConfig& config);

        /**
         * @brief 执行关键点检测
         * @param src_img 输入图像
         * @param face_bbox 人脸框
         * @return FaceKeypointResult 关键点检测结果
         */
        FaceKeypointResult run(const cv::Mat& src_img, const cv::Rect& face_bbox);

        /**
         * @brief 获取仿射变换矩阵
         * @param center 中心点
         * @param output_size 输出尺寸
         * @param scale 缩放比例
         * @param rotation 旋转角度
         * @return cv::Mat 仿射变换矩阵
         */
        static cv::Mat GetAffineTransform(const cv::Point2f& center,
            int output_size,
            float scale,
            float rotation);

        /**
         * @brief 变换点集坐标
         * @param points 输入点集
         * @param M 变换矩阵
         * @return std::vector<cv::Point2f> 变换后的点集
         */
        static std::vector<cv::Point2f> TransformPoints(const std::vector<cv::Point2f>& points,
            const cv::Mat& M);

        /**
         * @brief 释放资源
         */
        void release();

        /**
         * @brief 检查是否已初始化
         * @return bool 是否已初始化
         */
        bool IsInitialized() const { return initialized_; }

    private:
        /**
         * @brief 预处理
         * @param src 源图像
         * @param bbox 人脸框
         * @param dst 处理后的图像
         * @param M 输出变换矩阵
         * @return int 0成功，其他失败
         */
        int preProcess(const cv::Mat& src, const cv::Rect& bbox,
            cv::Mat& dst, cv::Mat& M);

        /**
         * @brief 模型推理
         * @param image 输入图像
         * @return int 0成功，其他失败
         */
        int inference(const cv::Mat& image);

        /**
         * @brief 后处理
         * @param M 变换矩阵
         * @param face_bbox 人脸框
         * @param result 输出关键点结果
         * @return int 0成功，其他失败
         */
        int postProcess(const cv::Mat& M, const cv::Rect& face_bbox,
            std::vector<FaceKeypoint>& result);

    private:
        bool initialized_;                 // 是否已初始化
        FaceKeypointConfig config_;        // 配置参数
        cv::Size inputSize_;               // 输入尺寸
        cv::Mat current_M_;                // 当前变换矩阵

        ModelProcess modelProcess_;        // 模型处理对象
        void* picDevBuffer_;               // 设备端图像缓冲区
        size_t devBufferSize_;             // 设备缓冲区大小

        std::vector<int> inputDims_;       // 输入维度
        std::vector<int> outputDims_;      // 输出维度

    };

} // namespace VISION_ENGINE

#endif // FACE_KEYPOINT_106_H
