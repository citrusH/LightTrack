#ifndef PERSON_REID_H
#define PERSON_REID_H

#include <vector>
#include <string>
#include <string>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"

#include "extern_c_wrapper.h"
#include "utils.h"

namespace VISION_ENGINE {
    
enum class ReIDInputFormat {
    RGB_PACKAGE,   /* HWC 交错，AIPP: RGB_PACKAGE */
    RGB_PLANAR     /* CHW 平面，AIPP: RGB_PLANAR   */
};

/* ---------------------------------------------------------------
 * 配置（对应原始 PersonReID::Config）
 * --------------------------------------------------------------- */
struct PCBReIDConfig {
    std::string          model_path;

    /* 颜色空间转换（对应原始 src_img_format / dst_img_format） */
    bool                 bgr2rgb       = true;

    /* 归一化参数（对应原始 means_ / norms_）
     * 预处理流程：float32 → (pixel - mean) * norm
     * 默认值：mean=0, norm=1/255（常见 ReID 模型归一化）
     */
    ReIDInputFormat  input_format = ReIDInputFormat::RGB_PACKAGE; /* AIPP 输入格式 */


    /* 输出是否为 fp16（atc 默认 fp32；若指定 --output_type=FP16 则置 true） */
    bool                 output_fp16   = false;

    int                  device_id     = 0;
};

/* ---------------------------------------------------------------
 * PersonReID_PCB DV500 版本
 * --------------------------------------------------------------- */
class PersonReID_PCB {
public:
    PersonReID_PCB ()  = default;
    ~PersonReID_PCB()  { destroy(); }

    PersonReID_PCB(const PersonReID_PCB&)            = delete;
    PersonReID_PCB& operator=(const PersonReID_PCB&) = delete;

    int  init   (const PCBReIDConfig& config);

    /**
     * run
     * @param image   原图（BGR）
     * @param box     行人检测框（x/y/width/height，与原始一致）
     * @param result  输出特征（1×D Mat，CV_32FC1，已 L2 归一化）
     * @return 0=成功，<0=失败
     */
    int  run    (const cv::Mat& image, cv::Rect box, cv::Mat& result);

    void destroy();

private:
    /* ---- 推理三阶段 ---- */
    int preProcess (const cv::Mat& src, cv::Rect box, cv::Mat& dst);
    int inference  (const cv::Mat& img);
    int postProcess(cv::Mat& result);

    /* ---- DV500 模型/task；SVP ACL runtime 由 LightTracker 持有 ---- */
    int  load_model_  ();
    void unload_model_();
    int  init_task_   ();
    void deinit_task_ ();

    /* ---- fp16 转换 ---- */
    static float fp16_to_f32(uint16_t h);

    /* ---- 状态 ---- */
    static constexpr int MODEL_IDX = 3;  /* Detector=0,FaceKps=1,ReID_ACL=2,PCB=3 */

    PCBReIDConfig            cfg_;
    bool                     inited_          = false;
    bool                     model_loaded_    = false;
    bool                     task_initialized_ = false;
    cv::Size                 net_wh_;          /* 模型输入尺寸（W×H），对应原始 net_wh_ */
    int                      feat_dim_        = 0;
    size_t                   out_stride_elem_ = 0;
    sample_svp_npu_task_info task_info_       {};
};

}

#endif // PERSON_REID_H
