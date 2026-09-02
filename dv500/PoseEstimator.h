#ifndef VISION_ENGINE_POSE_ESTIMATOR_H
#define VISION_ENGINE_POSE_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cstring>

/* DV500 ACL 相关头文件（与 Detector 一致） */
#include "svp_acl.h"
#include "svp_acl_rt.h"
#include "svp_acl_ext.h"
#include "extern_c_wrapper.h"


namespace VISION_ENGINE {



static constexpr int NUM_KEYPOINTS = 17;   /* COCO 17 keypoints */

/* COCO 关键点索引（与 SS927 版本一致，保证调用方代码无需修改） */
enum KeypointIndex {
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16
};

/* ==================================================================
 *  基础数据结构
 * ================================================================== */
struct PoseKeypoint {
    float x = 0.f;
    float y = 0.f;
    float confidence = 0.f;
};

struct PoseResult {
    cv::Rect_<float> box;          /* xywh 形式 */
    float score = 0.f;             /* 人物检测置信度 */
    PoseKeypoint keypoints[NUM_KEYPOINTS];
    int valid_kp_count = 0;        /* conf 高于阈值的关键点数 */
};

/* 骨骼比例描述符（用于 ReID 辅助身份判别） */
struct BodyProportionDescriptor {
    static constexpr int DIM = 14;

    float shoulder_width       = 0.f;
    float hip_width            = 0.f;
    float left_upper_arm       = 0.f;
    float right_upper_arm      = 0.f;
    float left_forearm         = 0.f;
    float right_forearm        = 0.f;
    float left_thigh           = 0.f;
    float right_thigh          = 0.f;
    float left_shin            = 0.f;
    float right_shin           = 0.f;
    float torso_ratio          = 0.f;
    float arm_ratio            = 0.f;   /* 上臂/前臂 */
    float leg_ratio            = 0.f;   /* 大腿/小腿 */
    float shoulder_hip_ratio   = 0.f;   /* 肩宽/髋宽 */

    float data[DIM]{};
    bool  valid = false;
    
    // ── 上半身比例（按肩宽归一，仅需 肩 + 头部点；抗桌面/半身遮挡）──
    // 全身描述符需双髋（valid 才为真）；桌遮挡时髋不可见 → valid=false，但这两维
    // 仍可计算（has_upper）。诚实地说这只是弱 tie-breaker，无法可靠区分同建之人。
    float ub_head_shoulder = 0.f;   // 头宽(耳/眼) / 肩宽（头大小 vs 肩宽）
    float ub_neck_shoulder = 0.f;   // (鼻→肩中点) / 肩宽（颈+头长 vs 肩宽）
    bool  has_upper = false;

    /* 身份强信号（比值的比值）权重 ×3 */
    static constexpr float weights[DIM] = {
        1.f, 1.f,                /* 0,1  shoulder / hip width  */
        1.f, 1.f, 1.f, 1.f,      /* 2-5  arms                  */
        1.f, 1.f, 1.f, 1.f,      /* 6-9  legs                  */
        1.f,                     /* 10   torso_ratio           */
        3.f, 3.f, 3.f            /* 11-13 identity-strong feats */
    };

    void flatten();
};

/* ==================================================================
 *  PoseEstimator 配置
 * ================================================================== */
struct PoseEstimatorConfig {
    std::string model_path;
    int   device_id        = 0;
    bool  bgr2rgb          = true;     /* 与训练时的通道顺序保持一致 */
    float threshold_score  = 0.25f;    /* 人物检测置信度阈值 */
    float kp_conf_thresh   = 0.40f;    /* 关键点可见性阈值 */
};

/* ==================================================================
 *  PoseMatch — 姿态匹配工具（与 SS927 版本完全一致）
 * ================================================================== */
class PoseMatch {
public:
    static float compute_oks(
        const PoseKeypoint kps_gt[NUM_KEYPOINTS],
        const PoseKeypoint kps_dt[NUM_KEYPOINTS],
        float area,
        float kp_conf_thresh = 0.4f);

    static BodyProportionDescriptor extract_body_proportions(
        const PoseKeypoint kps[NUM_KEYPOINTS],
        float kp_conf_thresh = 0.4f);
        
    // 骨骼比例描述符相似度（逐维相对偏差，[0,1]；稳健维不足返回 -1）
    static float body_proportion_similarity(
        const BodyProportionDescriptor& a,
        const BodyProportionDescriptor& b);
    
    // 上半身比例相似度（仅头宽/颈长两维，桌遮挡时全身描述符失效时的弱 tie-breaker）。
    // [0,1]；双方 has_upper 不成立或无可比维 → 返回 -1。
    static float upper_body_similarity(
        const BodyProportionDescriptor& a,
        const BodyProportionDescriptor& b);


    static float compute_pose_score(
        const PoseKeypoint kps_a[NUM_KEYPOINTS],
        const PoseKeypoint kps_b[NUM_KEYPOINTS],
        const BodyProportionDescriptor& bp_a,
        const BodyProportionDescriptor& bp_b,
        float area,
        float kp_conf_thresh = 0.4f);
};

/* ==================================================================
 *  PoseEstimator 主类（DV500 版本）
 * ================================================================== */
class PoseEstimator {
public:
    PoseEstimator() = default;
    ~PoseEstimator() { destroy(); }

    int  init(const PoseEstimatorConfig& cfg);
    int  run(const cv::Mat& bgr_image, const cv::Mat& body_box,
             PoseResult& result);
    void destroy();

    /* 根据 bbox 找匹配的 pose（IoU 最大） */
    static int find_best_match(
        const std::vector<PoseResult>& poses,
        const cv::Mat& bbox,            // [1,4] (x1,y1,x2,y2)
        float iou_thresh = 0.3f);

private:
    /* —— 模型/task 资源；SVP ACL runtime 由 LightTracker 持有 —— */
    int  load_model_();
    void unload_model_();
    int  init_task_();
    void deinit_task_();

    /* —— 推理三段式 —— */
    int  preProcess(const cv::Mat& image, const cv::Rect2f& body_box);
    int  inference();
    int  postProcess(cv::Size src_img_size, const cv::Rect2f& body_box,
                     PoseResult& result);
    bool build_topdown_affine(const cv::Rect2f& body_box);
    cv::Point2f map_point_to_original(float x, float y) const;
    static int argmax(const float* data, int length, float& max_value);

private:
    PoseEstimatorConfig cfg_{};
    bool   inited_      = false;
    bool   model_loaded_ = false;
    bool   task_initialized_ = false;
    int    net_w_       = 0;
    int    net_h_       = 0;
    bool   input_nchw_   = true;
    float  thresh_conf_ = 0.25f;
    static constexpr int POSE_MODEL_IDX = 4;

    /* DV500 NPU task descriptor —— 与 Detector 完全相同的字段 */
    sample_svp_npu_task_info task_info_{};
    cv::Mat affine_mat_;
    cv::Mat inverse_affine_mat_;

    static constexpr int kInputWidth = 192;
    static constexpr int kInputHeight = 256;
    static constexpr int kSimccXLength = 384;
    static constexpr int kSimccYLength = 512;
    static constexpr float kSimccSplitRatio = 2.f;
};

}  // namespace VISION_ENGINE

#endif  /* VISION_ENGINE_POSE_ESTIMATOR_H */
