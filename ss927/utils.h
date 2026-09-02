#pragma once

#ifndef AI_UTILS_H_
#define AI_UTILS_H_
#include "types.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <deque>
#include <cmath>
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...)fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)


using namespace VISION_ENGINE;

struct EmbQualityMetrics {
    float visibility_ratio;  // 身体可见比例估计 [0,1]
    float size_score;        // 相对画面大小得分
    float aspect_score;      // 宽高比合理性得分
    float final_score;       // 综合得分
};

class Utils {
public:

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] inputBuff: input data buffer
    * @param [out] fileSize: size of file
     * @return result；成功后的 inputBuff 由调用方使用 std::free() 释放
    */
    static Result ReadBinFile(const std::string& fileName, void*& inputBuff, uint32_t& fileSize);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] picDevBuffer: input data device buffer which need to be memcpy
    * @param [out] inputBuffSize: size of inputBuff
    * @return result
    */
    // cv::Mat/std::vector 等 CPU 内存上传到 init 阶段长期分配的设备输入 buffer。
    static Result MemcpyHostToDevice(const void* host_data, void* device_buffer,
                                     size_t size);

    static float iou_single(const cv::Mat& bbox1, const cv::Mat& bbox2);

    static cv::Mat iou_batch(const cv::Mat& bboxes1, const cv::Mat& bboxes2);


    static cv::Mat convert_bbox_to_z(const cv::Mat& bbox);

    static std::pair<cv::Vec2f, float> speed_direction(const cv::Mat& bbox1, const cv::Mat& bbox2);

    static cv::Rect2f convert_z_to_bbox(const cv::Mat& z);

    static cv::Mat k_previous_obs(const std::map<int, cv::Mat>& observations, int cur_age, int k);

    static float cosine_distance(const std::vector<float>& emb1, const std::vector<float>& emb2);

    static uint8_t* loadModel(const char* path_file, size_t& size_buffer);

    static float GetIOU(const cv::Rect box1, const cv::Rect box2);

    static EmbQualityMetrics estimate_emb_quality(const cv::Mat& bbox,  const cv::Size& frame_size, float det_confidence = 1.0f);
    // BBOX : XYWH

    static void expand2square(const cv::Mat& img, cv::Scalar background_color, cv::Mat& result, int& x_offset, int& y_offset);

    /**
    * @brief Check whether the path is a file.
    * @param [in] fileName: fold to check
    * @return result
    */
    static Result CheckPathIsFile(const std::string& fileName);
};



class GaussianSmoothBox {
private:
	int windowSize;							
	std::vector<cv::Rect> historyBoxes;		
	std::vector<double> gaussianKernel;		

public:
	GaussianSmoothBox(int winSize = 7, double sigma = 1.4) {
		windowSize = winSize;
		gaussianKernel = generateGaussianKernel(windowSize, sigma);
	}

	std::vector<double> generateGaussianKernel(int size, double sigma);

	cv::Rect smooth(const cv::Rect& currBox);
};

#endif

