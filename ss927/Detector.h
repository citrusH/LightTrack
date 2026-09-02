#ifndef MODULE_PERSON_DETECT_X_H
#define MODULE_PERSON_DETECT_X_H

#include <opencv2/opencv.hpp>

#include "ModelProcess.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "types.h"
#include "utils.h"

namespace VISION_ENGINE {

class Detector_yolox {
public:
    struct Config {
        int srcImgFormat = 0;
        int dstImgFormat = 0;
        size_t sizeModel = 0;
        uint8_t* bufferModel = nullptr;
        std::string pathModel;
        bool bgr2rgb = false;
        std::vector<float> means = {0, 0, 0};
        std::vector<float> norms = {1, 1, 1};
        std::vector<int> input_dim = {3, 416, 416};
        std::vector<std::string> input_names = {"input"};
        std::vector<std::string> output_names = {"output"};
        int numThread = 1;
        int forwardDevice = 0;
        int precisionMode = 0;
        float threshold_iou = 0.5f;
    };

    Detector_yolox();
    ~Detector_yolox();

    int init(Config& config);
    int run(const cv::Mat& image, float thresh_score,
            std::vector<ObjDetInfo>& result);
    void release();

private:
    void yuv420spToRGB(unsigned char* yuv420sp, unsigned char* rgb,
                      int width, int height);
    int inference();
    int preProcess(const cv::Mat& image, float& scale_w, float& scale_h);
    int postProcess(cv::Size src_img_size, float scale_w, float scale_h,
                    std::vector<ObjDetInfo>& result);

    ModelProcess personDetector_yolox;
    Config config_;
    std::vector<int> input_dim_;
    std::vector<int> output_dim_;
    cv::Size net_size_;
    void* picDevBuffer_ = nullptr;
    size_t devBufferSize_ = 0;
    float thresh_conf_ = 0.0f;
    float thresh_iou_ = 0.0f;
    bool initialized_ = false;
    cv::Mat preprocess_buffer_;
};

} // namespace VISION_ENGINE

#endif
