#pragma once
#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "ModelProcess.h"
#include "acl/acl.h"
#include "acl/acl_base.h"
#include "types.h"

namespace VISION_ENGINE {

// ──────────────────────────────────────────────────────────
// COCO 17 关键点索引
// ──────────────────────────────────────────────────────────
enum CocoKp {
    NOSE = 0,
    LEFT_EYE, RIGHT_EYE,
    LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    NUM_KEYPOINTS = 17
};

// ──────────────────────────────────────────────────────────
// 单个关键点
// ──────────────────────────────────────────────────────────
struct PoseKeypoint {
    float x = 0.f;            // 像素坐标（原图尺度）
    float y = 0.f;
    float confidence = 0.f;   // 关键点置信度
};

// ──────────────────────────────────────────────────────────
// 单个人的 Pose 检测结果
// ──────────────────────────────────────────────────────────
struct PoseResult {
    // ⚠ 真 xywh（标准 cv::Rect 语义；postprocess 以 Rect_<float>(x1,y1,x2-x1,y2-y1)
    //   填充）。与全工程"xyxy 塞 Rect"约定相反！旧注释误标为 (x1,y1,x2,y2)，
    //   曾导致 assess_visibility 调用方把 w/h 当 x2/y2 用（已修复）。
    //   消费方取右下角请用 x+width / y+height。
    cv::Rect_<float> box;
    float score = 0.f;                          // 检测置信度
    PoseKeypoint keypoints[NUM_KEYPOINTS];      // 17 个 COCO 关键点
    int valid_kp_count = 0;                     // 可见关键点数量 (conf > thresh)
};

// ──────────────────────────────────────────────────────────
// 骨骼比例描述符 — 人体结构指纹（不随姿态/尺度变化）
// 19维归一化向量：所有长度除以躯干高度
//
// 设计原则：
//   ① 绝对长度比（11维）：各肢段与躯干的比值
//   ② 肢体内部比例（4维）：上臂/前臂、大腿/小腿 — 基因决定，极其人格化
//   ③ 跨体结构比（4维）：肩髋比、臂腿比、全臂/全腿与躯干比
// 衣服相同也无法伪造这些骨骼比例
// ──────────────────────────────────────────────────────────
struct BodyProportionDescriptor {
    // ── 基础肢段比（10维）：各肢段长度 / 躯干高度 ──
    // 受 2D 投影影响，同一人不同姿态会有波动，但同帧比较时有效
    float shoulder_width;     // 肩宽 / 躯干高度
    float hip_width;          // 髋宽 / 躯干高度
    float left_upper_arm;     // 左上臂 / 躯干高度
    float right_upper_arm;    // 右上臂 / 躯干高度
    float left_forearm;       // 左前臂 / 躯干高度
    float right_forearm;      // 右前臂 / 躯干高度
    float left_thigh;         // 左大腿 / 躯干高度
    float right_thigh;        // 右大腿 / 躯干高度
    float left_shin;          // 左小腿 / 躯干高度
    float right_shin;         // 右小腿 / 躯干高度

    // ── 身份核心特征（4维）：比值的比值，抗 2D 投影畸变 ──
    // 分子分母同方向透视收缩 → 比值基本不随姿态/视角变化
    float torso_ratio;        // 全身高度 / 躯干高度（身高指纹）
    float arm_ratio;          // 上臂/前臂 均值（基因决定）
    float leg_ratio;          // 大腿/小腿 均值（基因决定）
    float shoulder_hip_ratio; // 肩宽/髋宽（V型 vs 梨型体型）

    bool  valid;

    // ── 上半身比例（按肩宽归一，仅需 肩 + 头部点；抗桌面/半身遮挡）──
    // 全身描述符需双髋（valid 才为真）；桌遮挡时髋不可见 → valid=false，但这两维
    // 仍可计算（has_upper）。诚实地说这只是弱 tie-breaker，无法可靠区分同建之人。
    float ub_head_shoulder = 0.f;   // 头宽(耳/眼) / 肩宽（头大小 vs 肩宽）
    float ub_neck_shoulder = 0.f;   // (鼻→肩中点) / 肩宽（颈+头长 vs 肩宽）
    bool  has_upper = false;

    static constexpr int DIM = 14;
    float data[DIM];

    // 维度权重（加权余弦相似度用）
    // 身份核心特征 ×3，中稳特征 ×2，基础特征 ×1
    static constexpr float weights[DIM] = {
        1.f, 1.f,                       // shoulder_width, hip_width
        1.f, 1.f, 1.f, 1.f,            // 上臂、前臂
        1.f, 1.f, 1.f, 1.f,            // 大腿、小腿
        2.f,                             // torso_ratio（中等稳定）
        3.f, 3.f, 3.f                   // arm_ratio, leg_ratio, shoulder_hip_ratio（最稳定）
    };

    void flatten();
};

// ──────────────────────────────────────────────────────────
// 配置
// ──────────────────────────────────────────────────────────
struct PoseEstimatorConfig {
    size_t   sizeModel;
    uint8_t* bufferModel;
    bool     bgr2rgb;
    int      deviceId;
    float    kpConfThresh;
};

// ──────────────────────────────────────────────────────────
// 姿态匹配工具函数（静态）
// ──────────────────────────────────────────────────────────
class PoseMatch {
public:
    // OKS: 衡量两组关键点的空间吻合度，范围 [0,1]
    // area = 目标框面积（像素²），用于归一化
    static float compute_oks(
        const PoseKeypoint kps_gt[NUM_KEYPOINTS],
        const PoseKeypoint kps_dt[NUM_KEYPOINTS],
        float area,
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

    // 从关键点提取骨骼比例描述符
    static BodyProportionDescriptor extract_body_proportions(
        const PoseKeypoint kps[NUM_KEYPOINTS],
        float kp_conf_thresh = 0.4f);

    // 综合 pose 相似度 = w_oks * oks + w_bp * bp_sim
    // 只在两者都有效时混合，否则只用有效的那个
    static float compute_pose_score(
        const PoseKeypoint kps_a[NUM_KEYPOINTS],
        const PoseKeypoint kps_b[NUM_KEYPOINTS],
        const BodyProportionDescriptor& bp_a,
        const BodyProportionDescriptor& bp_b,
        float area,
        float kp_conf_thresh = 0.4f);
};

// ──────────────────────────────────────────────────────────
// 推理类
// ──────────────────────────────────────────────────────────
class PoseEstimator {
public:
    PoseEstimator();
    ~PoseEstimator();

    int init(const PoseEstimatorConfig& config);

    int run(const cv::Mat& image, const cv::Mat& body_box,
            PoseResult& result);

    // 找到与给定 bbox 最匹配的 PoseResult（IoU 最大）
    // 返回 -1 表示没找到
    static int find_best_match(
        const std::vector<PoseResult>& poses,
        const cv::Mat& bbox,            // [1,4] (x1,y1,x2,y2)
        float iou_thresh = 0.3f);

    void release();

private:
    int preProcess(const cv::Mat& image, const cv::Rect2f& body_box);
    int inference();
    int postProcess(cv::Size src_img_size, const cv::Rect2f& body_box,
                    PoseResult& result);
    bool build_topdown_affine(const cv::Rect2f& body_box);
    cv::Point2f map_point_to_original(float x, float y) const;
    static int argmax(const float* data, int length, float& max_value);

private:
    bool            initialized_;
    PoseEstimatorConfig config_;
    cv::Size        netSize_;

    ModelProcess    modelProcess_;
    void*           picDevBuffer_;
    size_t          devBufferSize_;

    std::vector<int> inputDims_;
    cv::Mat affineMat_;
    cv::Mat inverseAffineMat_;

    static constexpr int kInputWidth = 192;
    static constexpr int kInputHeight = 256;
    static constexpr int kSimccXLength = 384;
    static constexpr int kSimccYLength = 512;
    static constexpr float kSimccSplitRatio = 2.f;

};

} // namespace VISION_ENGINE

#endif // POSE_ESTIMATOR_H
