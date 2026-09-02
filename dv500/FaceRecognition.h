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
#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "extern_c_wrapper.h"



namespace VISION_ENGINE {


    struct FaceInfo {
        std::vector<float> kps;           // 5个关键点，格式: [x1,y1, x2,y2, x3,y3, x4,y4, x5,y5]
        std::vector<float> embedding;     // 人脸特征向量
    };


    enum class InputFormatDV500 {
        RGB_PACKAGE,
        RGB_PLANAR
    };

    struct FaceRecogConfigDV500 {
        std::string model_path;
        int device_id = 0;
        cv::Size inputSize = cv::Size(112, 112);
        bool bgr2rgb = true;                // 是否 BGR→RGB
        InputFormatDV500 input_format = InputFormatDV500::RGB_PACKAGE;
        bool output_fp16 = false;           // 模型输出是否为 fp16
    };

    class CFaceRecognitionDV500 {
        public:
            CFaceRecognitionDV500();
            ~CFaceRecognitionDV500();

            int init(const FaceRecogConfigDV500& config);
            std::vector<float> get(const cv::Mat& img, const std::vector<float>& landmarks);
            std::vector<float> get_feat(const cv::Mat& aligned_face);
            void destroy();
            static float compute_sim(const std::vector<float>& feat1,
                                    const std::vector<float>& feat2);

        private:
            cv::Mat norm_crop(const cv::Mat& img, const std::vector<float>& landmark, int image_size);

            int preProcess(const cv::Mat& src);
            int inference();
            int postProcess(std::vector<float>& feature);

            // 模型/task 资源；SVP ACL runtime 由 LightTracker 持有
            int load_model_();
            void unload_model_();
            int init_task_();
            void deinit_task_();

        private:
            bool inited_;
            bool model_loaded_ = false;
            bool task_initialized_ = false;
            FaceRecogConfigDV500 cfg_;
            int net_h_, net_w_;
            int embedding_dim_;
            size_t out_stride_elem_;
            int out_total_elem_;

            // NPU task handle
            sample_svp_npu_task_info task_info_;
            static constexpr int MODEL_IDX = 2;
        };
} // namespace VISION_ENGINE

#endif // FACE_RECOGNITION_H
