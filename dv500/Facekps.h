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
#include "extern_c_wrapper.h"


namespace VISION_ENGINE {

/* ---------------------------------------------------------------
 * 单个关键点（与原始 FaceKeypoint 一致）
 * --------------------------------------------------------------- */
struct FaceKeypoint {
    float x;
    float y;
};

/* ---------------------------------------------------------------
 * 检测结果（与原始 FaceKeypointResult 一致）
 * --------------------------------------------------------------- */
struct FaceKeypointResult {
    cv::Rect              face_rect;   /* 输入的人脸框（原图坐标） */
    std::vector<FaceKeypoint> points;  /* 关键点坐标（原图坐标） */
};

/* ---------------------------------------------------------------
 * 输入格式枚举
 * 对应 AIPP insert_op_conf 里的 input_format
 * --------------------------------------------------------------- */
enum class InputFormat {
    RGB_PACKAGE,   /* HWC 交错，AIPP: RGB_PACKAGE */
    RGB_PLANAR     /* CHW 平面，AIPP: RGB_PLANAR   */
};

/* ---------------------------------------------------------------
 * 配置（对应原始 FaceKeypointConfig）
 * --------------------------------------------------------------- */
struct FaceKeypointConfig {
    std::string  model_path;                          /* .om 文件路径 */
    cv::Size     inputSize    = cv::Size(192, 192);   /* 模型输入尺寸 */
    int          numPoints    = 106;                  /* 关键点数量   */
    bool         bgr2rgb      = true;                 /* OpenCV默认BGR，模型通常需RGB */
    InputFormat  input_format = InputFormat::RGB_PACKAGE; /* AIPP输入格式 */
    bool         output_fp16  = false;                /* 输出是否为fp16 */
    int          device_id    = 0;
};

/* ---------------------------------------------------------------
 * CFaceKeypoint106 DV500 版本
 * --------------------------------------------------------------- */
class CFaceKeypoint106 {
public:
    CFaceKeypoint106()  = default;
    ~CFaceKeypoint106() { destroy(); }

    CFaceKeypoint106(const CFaceKeypoint106&)            = delete;
    CFaceKeypoint106& operator=(const CFaceKeypoint106&) = delete;

    int               init   (const FaceKeypointConfig& config);
    FaceKeypointResult run   (const cv::Mat& src_img,
                               const cv::Rect& face_bbox);
    void              destroy();

private:
    /* ---- 推理三阶段 ---- */
    int preProcess (const cv::Mat& src,
                    const cv::Rect& bbox,
                    cv::Mat& dst_dbg,
                    cv::Mat& M);
    int inference  ();
    int postProcess(const cv::Mat& M,
                    std::vector<FaceKeypoint>& result);

    /* ---- 仿射工具（与原始完全一致） ---- */
    cv::Mat GetAffineTransform(const cv::Point2f& center,
                               int   output_size,
                               float scale,
                               float rotation);

    std::vector<cv::Point2f> TransformPoints(
        const std::vector<cv::Point2f>& pts,
        const cv::Mat& M);

    /* ---- DV500 模型/task；SVP ACL runtime 由 LightTracker 持有 ---- */
    int  load_model_  ();
    void unload_model_();
    int  init_task_   ();
    void deinit_task_ ();

    /* ---- 输出读取辅助 ---- */
    static float fp16_to_f32(uint16_t h);

    /* ---- 状态 ---- */
    static constexpr int MODEL_IDX = 1;   /* 与 Detector 使用 0，此处用 1 避免冲突 */

    FaceKeypointConfig       cfg_;
    bool                     inited_    = false;
    bool                     model_loaded_ = false;
    bool                     task_initialized_ = false;
    int                      net_w_     = 0;
    int                      net_h_     = 0;
    sample_svp_npu_task_info task_info_ {};

    /* 输出 tensor 描述（init 时缓存，避免每帧查询） */
    size_t out_stride_elem_ = 0;    /* 输出行步进（单位：元素个数） */
    int    out_total_elem_  = 0;    /* 输出总元素数（numPoints * 2） */
};

} // namespace VISION_ENGINE

#endif // FACE_KEYPOINT_106_H
