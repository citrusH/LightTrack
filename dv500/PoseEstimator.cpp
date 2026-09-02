#include "PoseEstimator.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ModelProfiler.h"

using namespace cv;
using namespace VISION_ENGINE;

#define PLOGI(fmt, ...) printf("[PoseEstimator][I] " fmt "\n", ##__VA_ARGS__)
#define PLOGE(fmt, ...) printf("[PoseEstimator][E] " fmt "\n", ##__VA_ARGS__)

/* ================================================================== *
 *  COCO OKS 标准 σ（每个关键点的 scale factor）
 * ================================================================== */

// ════════════════════════════════════════════════════════════
// COCO OKS 标准σ值（per-keypoint scale factor）
// 来源: COCO evaluation 官方定义
// ════════════════════════════════════════════════════════════
static const float kOksSigma[NUM_KEYPOINTS] = {
    0.026f,  // nose
    0.025f,  // left_eye
    0.025f,  // right_eye
    0.035f,  // left_ear
    0.035f,  // right_ear
    0.079f,  // left_shoulder
    0.079f,  // right_shoulder
    0.072f,  // left_elbow
    0.072f,  // right_elbow
    0.062f,  // left_wrist
    0.062f,  // right_wrist
    0.107f,  // left_hip
    0.107f,  // right_hip
    0.087f,  // left_knee
    0.087f,  // right_knee
    0.089f,  // left_ankle
    0.089f   // right_ankle
};

// ════════════════════════════════════════════════════════════
// 工具函数
// ════════════════════════════════════════════════════════════
static inline float point_dist(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

static inline float box_iou(const Rect_<float>& a, const Rect_<float>& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.f;
    float inter = (x2 - x1) * (y2 - y1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    return inter / (area_a + area_b - inter + 1e-6f);
}

// ════════════════════════════════════════════════════════════
// PoseMatch: OKS 计算
// ════════════════════════════════════════════════════════════
float PoseMatch::compute_oks(
    const PoseKeypoint kps_gt[NUM_KEYPOINTS],
    const PoseKeypoint kps_dt[NUM_KEYPOINTS],
    float area,
    float kp_conf_thresh)
{
    if (area <= 0.f) return 0.f;

    float sum = 0.f;
    int count = 0;

    for (int i = 0; i < NUM_KEYPOINTS; ++i) {
        // 两边都需要可见
        if (kps_gt[i].confidence < kp_conf_thresh ||
            kps_dt[i].confidence < kp_conf_thresh)
            continue;

        float dx = kps_gt[i].x - kps_dt[i].x;
        float dy = kps_gt[i].y - kps_dt[i].y;
        float d2 = dx * dx + dy * dy;

        float s2 = kOksSigma[i] * kOksSigma[i];
        float var = 2.0f * s2 * area;  // COCO 标准: 2 * σ² * area

        // 置信度加权：两侧中较低的作为权重
        float w = std::min(kps_gt[i].confidence, kps_dt[i].confidence);

        sum += w * std::exp(-d2 / (var + 1e-6f));
        count++;
    }

    if (count < 3) return 0.f;
    return sum / count;
}

// ════════════════════════════════════════════════════════════
// PoseMatch: 骨骼比例描述符提取
// ════════════════════════════════════════════════════════════
BodyProportionDescriptor PoseMatch::extract_body_proportions(
    const PoseKeypoint kps[NUM_KEYPOINTS],
    float kp_conf_thresh)
{
    BodyProportionDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.valid = false;

    auto visible = [&](int idx) -> bool {
        return kps[idx].confidence >= kp_conf_thresh;
    };

    // 双肩为一切比例的归一基础：无肩 → 无法计算任何描述符
    if (!visible(LEFT_SHOULDER) || !visible(RIGHT_SHOULDER))
        return desc;

    // ── 上半身比例（仅需 肩 + 头部点，抗桌面/半身遮挡）────────────────
    //   桌遮挡时髋不可见 → 全身描述符失效，但肩+头通常仍在 → 提供"头大小/颈长
    //   相对肩宽"的弱判别（按肩宽归一，距离无关）。设 has_upper，valid 仍按全身判定。
    {
        float ub_sw = point_dist(kps[LEFT_SHOULDER].x, kps[LEFT_SHOULDER].y,
                                 kps[RIGHT_SHOULDER].x, kps[RIGHT_SHOULDER].y);
        if (ub_sw >= 3.f) {
            float msx = (kps[LEFT_SHOULDER].x + kps[RIGHT_SHOULDER].x) * 0.5f;
            float msy = (kps[LEFT_SHOULDER].y + kps[RIGHT_SHOULDER].y) * 0.5f;
            float head_w = -1.f;          // 头宽：双耳优先，退双眼
            if (visible(LEFT_EAR) && visible(RIGHT_EAR))
                head_w = point_dist(kps[LEFT_EAR].x, kps[LEFT_EAR].y,
                                    kps[RIGHT_EAR].x, kps[RIGHT_EAR].y);
            else if (visible(LEFT_EYE) && visible(RIGHT_EYE))
                head_w = point_dist(kps[LEFT_EYE].x, kps[LEFT_EYE].y,
                                    kps[RIGHT_EYE].x, kps[RIGHT_EYE].y);
            float neck_len = -1.f;        // 颈+头长：鼻（退眼中点）→ 肩中点
            if (visible(NOSE))
                neck_len = point_dist(kps[NOSE].x, kps[NOSE].y, msx, msy);
            else if (visible(LEFT_EYE) && visible(RIGHT_EYE))
                neck_len = point_dist((kps[LEFT_EYE].x + kps[RIGHT_EYE].x) * 0.5f,
                                      (kps[LEFT_EYE].y + kps[RIGHT_EYE].y) * 0.5f, msx, msy);
            if (head_w  > 1.f) desc.ub_head_shoulder = head_w  / ub_sw;
            if (neck_len > 1.f) desc.ub_neck_shoulder = neck_len / ub_sw;
            if (desc.ub_head_shoulder > 1e-6f || desc.ub_neck_shoulder > 1e-6f)
                desc.has_upper = true;
        }
    }

    // 全身描述符需双髋（躯干高度）：缺髋 → 仅返回上半身（valid=false, has_upper 可能为真）
    if (!visible(LEFT_HIP) || !visible(RIGHT_HIP))
        return desc;

    // 躯干高度 = midpoint(shoulders) → midpoint(hips)
    float mid_sh_x = (kps[LEFT_SHOULDER].x + kps[RIGHT_SHOULDER].x) * 0.5f;
    float mid_sh_y = (kps[LEFT_SHOULDER].y + kps[RIGHT_SHOULDER].y) * 0.5f;
    float mid_hp_x = (kps[LEFT_HIP].x + kps[RIGHT_HIP].x) * 0.5f;
    float mid_hp_y = (kps[LEFT_HIP].y + kps[RIGHT_HIP].y) * 0.5f;
    float torso_h  = point_dist(mid_sh_x, mid_sh_y, mid_hp_x, mid_hp_y);

    if (torso_h < 5.f) return desc;  // 太小不可靠

    float sh_w = point_dist(kps[LEFT_SHOULDER].x, kps[LEFT_SHOULDER].y,
                            kps[RIGHT_SHOULDER].x, kps[RIGHT_SHOULDER].y);
    float hp_w = point_dist(kps[LEFT_HIP].x, kps[LEFT_HIP].y,
                            kps[RIGHT_HIP].x, kps[RIGHT_HIP].y);

    desc.shoulder_width  = sh_w / torso_h;
    desc.hip_width       = hp_w / torso_h;

    // torso_ratio：全身高度 / 躯干高度
    // 直接反映身高特征 — 高个子腿长 → 比值大，矮个子 → 比值小
    // （同一帧同一距离下，不同身高的人此值差异明显）
    {
        float top_y = mid_sh_y;
        float bot_y = mid_hp_y;
        if (visible(NOSE))         top_y = std::min(top_y, kps[NOSE].y);
        if (visible(LEFT_EYE))     top_y = std::min(top_y, kps[LEFT_EYE].y);
        if (visible(RIGHT_EYE))    top_y = std::min(top_y, kps[RIGHT_EYE].y);
        if (visible(LEFT_ANKLE))   bot_y = std::max(bot_y, kps[LEFT_ANKLE].y);
        if (visible(RIGHT_ANKLE))  bot_y = std::max(bot_y, kps[RIGHT_ANKLE].y);
        if (visible(LEFT_KNEE))    bot_y = std::max(bot_y, kps[LEFT_KNEE].y);
        if (visible(RIGHT_KNEE))   bot_y = std::max(bot_y, kps[RIGHT_KNEE].y);
        desc.torso_ratio = (bot_y - top_y) / torso_h;
    }

    // 四肢（不可见的用对侧镜像，再不行设 0）
    auto limb_len = [&](int a, int b) -> float {
        if (visible(a) && visible(b))
            return point_dist(kps[a].x, kps[a].y, kps[b].x, kps[b].y) / torso_h;
        return 0.f;
    };

    desc.left_upper_arm  = limb_len(LEFT_SHOULDER,  LEFT_ELBOW);
    desc.right_upper_arm = limb_len(RIGHT_SHOULDER, RIGHT_ELBOW);
    desc.left_forearm    = limb_len(LEFT_ELBOW,     LEFT_WRIST);
    desc.right_forearm   = limb_len(RIGHT_ELBOW,    RIGHT_WRIST);
    desc.left_thigh      = limb_len(LEFT_HIP,       LEFT_KNEE);
    desc.right_thigh     = limb_len(RIGHT_HIP,      RIGHT_KNEE);
    desc.left_shin       = limb_len(LEFT_KNEE,      LEFT_ANKLE);
    desc.right_shin      = limb_len(RIGHT_KNEE,     RIGHT_ANKLE);

    // 对称补全：如果一侧不可见，用另一侧
    if (desc.left_upper_arm  < 1e-6f) desc.left_upper_arm  = desc.right_upper_arm;
    if (desc.right_upper_arm < 1e-6f) desc.right_upper_arm = desc.left_upper_arm;
    if (desc.left_forearm    < 1e-6f) desc.left_forearm    = desc.right_forearm;
    if (desc.right_forearm   < 1e-6f) desc.right_forearm   = desc.left_forearm;
    if (desc.left_thigh      < 1e-6f) desc.left_thigh      = desc.right_thigh;
    if (desc.right_thigh     < 1e-6f) desc.right_thigh     = desc.left_thigh;
    if (desc.left_shin       < 1e-6f) desc.left_shin       = desc.right_shin;
    if (desc.right_shin      < 1e-6f) desc.right_shin      = desc.left_shin;

    // 至少需要躯干 + 2条四肢有效数据
    int valid_limbs = 0;
    if (desc.left_upper_arm  > 1e-6f) valid_limbs++;
    if (desc.right_upper_arm > 1e-6f) valid_limbs++;
    if (desc.left_thigh      > 1e-6f) valid_limbs++;
    if (desc.right_thigh     > 1e-6f) valid_limbs++;

    if (valid_limbs < 2) return desc;

    // ═══════════════════════════════════════════════
    // 新增维度：肢体内部比例（基因决定，极其人格化）
    // ═══════════════════════════════════════════════

    // 上臂/前臂比例（左右取均值；单侧不可见用另一侧）
    float arm_ratio_L = (desc.left_forearm  > 1e-6f) ? desc.left_upper_arm  / desc.left_forearm  : 0.f;
    float arm_ratio_R = (desc.right_forearm > 1e-6f) ? desc.right_upper_arm / desc.right_forearm : 0.f;
    if (arm_ratio_L < 1e-6f) arm_ratio_L = arm_ratio_R;
    if (arm_ratio_R < 1e-6f) arm_ratio_R = arm_ratio_L;
    desc.arm_ratio = (arm_ratio_L + arm_ratio_R) * 0.5f;

    // 大腿/小腿比例
    float leg_ratio_L = (desc.left_shin  > 1e-6f) ? desc.left_thigh  / desc.left_shin  : 0.f;
    float leg_ratio_R = (desc.right_shin > 1e-6f) ? desc.right_thigh / desc.right_shin : 0.f;
    if (leg_ratio_L < 1e-6f) leg_ratio_L = leg_ratio_R;
    if (leg_ratio_R < 1e-6f) leg_ratio_R = leg_ratio_L;
    desc.leg_ratio = (leg_ratio_L + leg_ratio_R) * 0.5f;

    // 肩宽/髋宽（V型 vs 梨型体型）
    desc.shoulder_hip_ratio = (hp_w > 1e-6f) ? sh_w / hp_w : 0.f;

    desc.valid = true;
    desc.flatten();
    return desc;
}

void BodyProportionDescriptor::flatten() {
    data[0]  = shoulder_width;
    data[1]  = hip_width;
    data[2]  = left_upper_arm;
    data[3]  = right_upper_arm;
    data[4]  = left_forearm;
    data[5]  = right_forearm;
    data[6]  = left_thigh;
    data[7]  = right_thigh;
    data[8]  = left_shin;
    data[9]  = right_shin;
    data[10] = torso_ratio;
    data[11] = arm_ratio;
    data[12] = leg_ratio;
    data[13] = shoulder_hip_ratio;
}

// C++14 需要 constexpr 静态成员的外部定义
constexpr float BodyProportionDescriptor::weights[];

// ════════════════════════════════════════════════════════════
// PoseMatch: 骨骼比例相似度（逐维相对偏差，替代"全正向量余弦"）
//
// 旧实现用"全正比例向量的加权余弦"：所有分量为正 → 向量几乎同向 →
// 任意两个成年人余弦都 ≈0.99（即便体型差异很大），判别区间 <0.01 被姿态
// 噪声淹没（"人人相似"），既无法识别冒充者，作硬门槛时反而误否决真目标。
//
// 新实现：只取抗 2D 投影的"比值的比值"维（arm/leg/shoulder_hip + 低权 torso），
// 逐维计算对称相对偏差并按容差归一，得到有真实动态范围的相似度 [0,1]：
//   reldev_i = |a_i - b_i| / (0.5·(a_i + b_i))      （对称相对差）
//   s_i      = max(0, 1 - reldev_i / tol_i)          （单维相似度）
//   sim      = Σ w_i·s_i / Σ w_i                      （加权平均）
// 同一人各维偏差小 → sim 高（~0.85+）；体型不同（如肩髋比差 ~20%）→ sim 明显下降。
// 稳健维有效数 <2 → 返回 -1（无法可靠比较，由调用方据此判定 has_shape）。
// 容差为器件调参起点，建议在真机上按"同人/异人"分布标定。
// ════════════════════════════════════════════════════════════
float PoseMatch::body_proportion_similarity(
    const BodyProportionDescriptor& a,
    const BodyProportionDescriptor& b)
{
    if (!a.valid || !b.valid) return -1.f;

    // 仅用抗 2D 投影的稳健维：data 索引 + 容差 tol + 权重 w
    //   [11]=arm_ratio, [12]=leg_ratio, [13]=shoulder_hip_ratio（体型主判别），
    //   [10]=torso_ratio（身高/腿长，较噪 → 低权 + 宽容差）
    struct Dim { int idx; float tol; float w; };
    static const Dim dims[] = {
        { 11, 0.15f, 3.f },
        { 12, 0.15f, 3.f },
        { 13, 0.18f, 3.f },
        { 10, 0.20f, 1.f },
    };

    float ssum = 0.f, sw = 0.f;
    int   used = 0;
    for (const auto& d : dims) {
        float av = a.data[d.idx], bv = b.data[d.idx];
        if (av < 1e-6f || bv < 1e-6f) continue;          // 任一方缺失该维 → 跳过
        float denom = 0.5f * (av + bv);
        if (denom < 1e-6f) continue;
        float reldev = std::fabs(av - bv) / denom;        // 对称相对偏差
        float s = 1.f - reldev / d.tol;
        if (s < 0.f) s = 0.f;
        ssum += d.w * s;
        sw   += d.w;
        used++;
    }

    if (used < 2 || sw < 1e-6f) return -1.f;              // 稳健维不足 → 无法可靠比较
    return ssum / sw;
}

// ════════════════════════════════════════════════════════════
// PoseMatch: 上半身比例相似度（全身描述符失效时的弱 tie-breaker）
//   仅头宽/肩宽、颈长/肩宽两维，逐维相对偏差 → [0,1]；任一方无 has_upper 返回 -1。
//   诚实警示：两维相近的同建之人此值仍会偏高，仅作微弱区分，不可作硬否决依据。
// ════════════════════════════════════════════════════════════
float PoseMatch::upper_body_similarity(
    const BodyProportionDescriptor& a,
    const BodyProportionDescriptor& b)
{
    if (!a.has_upper || !b.has_upper) return -1.f;
    struct D { float av, bv, tol; };
    const D dims[2] = {
        { a.ub_head_shoulder, b.ub_head_shoulder, 0.15f },
        { a.ub_neck_shoulder, b.ub_neck_shoulder, 0.18f },
    };
    float ssum = 0.f; int used = 0;
    for (const auto& d : dims) {
        if (d.av < 1e-6f || d.bv < 1e-6f) continue;
        float denom = 0.5f * (d.av + d.bv);
        if (denom < 1e-6f) continue;
        float rel = std::fabs(d.av - d.bv) / denom;
        float s = 1.f - rel / d.tol;
        if (s < 0.f) s = 0.f;
        ssum += s; used++;
    }
    if (used < 1) return -1.f;
    return ssum / (float)used;
}

// ════════════════════════════════════════════════════════════
// PoseMatch: 综合 pose score
// ════════════════════════════════════════════════════════════
float PoseMatch::compute_pose_score(
    const PoseKeypoint kps_a[NUM_KEYPOINTS],
    const PoseKeypoint kps_b[NUM_KEYPOINTS],
    const BodyProportionDescriptor& bp_a,
    const BodyProportionDescriptor& bp_b,
    float area,
    float kp_conf_thresh)
{
    float oks = compute_oks(kps_a, kps_b, area, kp_conf_thresh);
    float bp_sim = body_proportion_similarity(bp_a, bp_b);

    bool has_oks = (oks > 1e-6f);
    bool has_bp  = (bp_sim >= 0.f);   // -1 = 稳健维不足，无法可靠比较

    if (has_oks && has_bp) {
        // OKS 权重大（帧间位置匹配），骨骼比例辅助确认身份
        return 0.65f * oks + 0.35f * bp_sim;
    } else if (has_oks) {
        return oks;
    } else if (has_bp) {
        return bp_sim;
    }
    return 0.f;
}

/* ==================================================================
 *  ============  PoseEstimator (DV500) 实现部分  ====================
 * ================================================================== */

/* ================================================================== *
 *  init —— 复用 Detector 的 DV500 初始化流程
 * ================================================================== */
int PoseEstimator::init(const PoseEstimatorConfig& cfg)
{
    if (inited_) { PLOGE("already inited"); return -1; }
    destroy();
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
    cfg_         = cfg;
    thresh_conf_ = cfg.threshold_score;

    if (load_model_() != 0) { destroy(); return -1; }
    model_loaded_ = true;

    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(POSE_MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        PLOGE("get model info failed");
        destroy(); return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;

    svp_acl_mdl_io_dims in_dims{};
    svp_acl_error acl_ret = svp_acl_mdl_get_input_dims(desc, 0, &in_dims);
    if (acl_ret != SVP_ACL_SUCCESS) {
        PLOGE("get input dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }

    input_nchw_ = in_dims.dims[1] == 3 || in_dims.dims[1] == 1;
    if (input_nchw_) {
        net_h_ = (int)in_dims.dims[2];
        net_w_ = (int)in_dims.dims[3];
    } else {
        net_h_ = (int)in_dims.dims[1];
        net_w_ = (int)in_dims.dims[2];
    }
    if (net_w_ != kInputWidth || net_h_ != kInputHeight) {
        PLOGE("unexpected RTMPose input %dx%d, expected %dx%d",
              net_w_, net_h_, kInputWidth, kInputHeight);
        destroy(); return -1;
    }
    if (model_info->output_num != 2) {
        PLOGE("unexpected RTMPose output count=%zu, expected 2", model_info->output_num);
        destroy(); return -1;
    }
    PLOGI("RTMPose input: %s %dx%d, output_num=%zu",
          input_nchw_ ? "NCHW" : "NHWC/AIPP", net_w_, net_h_, model_info->output_num);

    task_info_.cfg.max_batch_num     = 1;
    task_info_.cfg.dynamic_batch_num = 1;
    task_info_.cfg.total_t           = 0;
    task_info_.cfg.is_cached         = TD_TRUE;
    task_info_.cfg.model_idx         = POSE_MODEL_IDX;

    task_initialized_ = true;
    if (init_task_() != 0) { destroy(); return -1; }

    inited_ = true;
    PLOGI("PoseEstimator init ok: %s", cfg_.model_path.c_str());
    return 0;
}

/* ================================================================== *
 *  run
 * ================================================================== */
int PoseEstimator::run(const cv::Mat& bgr_image, const cv::Mat& body_box,
                       PoseResult& result)
{
    if (!inited_) { PLOGE("not inited"); return -1; }
    if (bgr_image.empty()) { PLOGE("empty image"); return -1; }
    if (body_box.empty() || body_box.cols < 4) { PLOGE("invalid body box"); return -1; }
    result = PoseResult{};
    cv::Rect2f roi(body_box.at<float>(0, 0), body_box.at<float>(0, 1),
                   body_box.at<float>(0, 2) - body_box.at<float>(0, 0),
                   body_box.at<float>(0, 3) - body_box.at<float>(0, 1));
    if (roi.width <= 1.f || roi.height <= 1.f) { PLOGE("degenerate body box"); return -1; }
    fxprof::add_call(fxprof::Model::Pose);

    int ret;
    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Pre);
        ret = preProcess(bgr_image, roi);
    }
    if (ret < 0) return -1;

    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Infer);
        ret = inference();
    }
    if (ret < 0) return -1;

    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Post);
        ret = postProcess(bgr_image.size(), roi, result);
    }
    if (ret < 0) return -1;
    return 0;
}

/* RTMPose TopdownAffine：BODY xyxy → 192×256，越界区域填 0。 */

bool PoseEstimator::build_topdown_affine(const cv::Rect2f& body_box)
{
    constexpr float kBBoxPadding = 1.25f;
    const float center_x = body_box.x + body_box.width * 0.5f;
    const float center_y = body_box.y + body_box.height * 0.5f;
    float scale_w = body_box.width * kBBoxPadding;
    float scale_h = body_box.height * kBBoxPadding;
    const float target_aspect = (float)net_w_ / (float)net_h_;
    if (scale_w > scale_h * target_aspect) scale_h = scale_w / target_aspect;
    else scale_w = scale_h * target_aspect;
    if (scale_w <= 1.f || scale_h <= 1.f) return false;
    const float sx = (float)net_w_ / scale_w;
    const float sy = (float)net_h_ / scale_h;
    affine_mat_ = (cv::Mat_<double>(2, 3) <<
        sx, 0.0, (double)net_w_ * 0.5 - center_x * sx,
        0.0, sy, (double)net_h_ * 0.5 - center_y * sy);
    cv::invertAffineTransform(affine_mat_, inverse_affine_mat_);
    return true;
}

int PoseEstimator::preProcess(const cv::Mat& image, const cv::Rect2f& body_box)
{
    if (image.type() != CV_8UC3) {
        PLOGE("preProcess expects CV_8UC3 src, got type=%d", image.type());
        return -1;
    }
    if (!build_topdown_affine(body_box)) {
        PLOGE("build topdown affine failed");
        return -1;
    }

    /* ---- 先取设备输入缓冲（cached MMZ，CPU 可直接寻址）：
     *      让 resize/填边/cvtColor 直接写进去，省掉 copyMakeBorder 的整图分配
     *      和逐行 memset+memcpy_s 的上传 ---- */
    td_u8  *dev_ptr  = nullptr;
    td_u32  buf_size = 0;
    td_u32  stride   = 0;
    td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
        &task_info_, 0, &dev_ptr, &buf_size, &stride);
    if (ret != TD_SUCCESS || dev_ptr == nullptr) {
        PLOGE("get input buffer info failed 0x%x", ret);
        return -1;
    }

    const int    row_bytes  = net_w_ * 3;
    const td_u32 valid_size = stride * net_h_;
    if ((td_u32)row_bytes > stride) {
        PLOGE("stride too small: stride=%u need=%d", stride, row_bytes);
        return -1;
    }
    if (valid_size > buf_size) {
        PLOGE("buffer not enough: stride=%u buf=%u", stride, buf_size);
        return -1;
    }

    cv::Mat dev_img(net_h_, net_w_, CV_8UC3, dev_ptr, (size_t)stride);

    /* 行尾 stride 填充字节清零：仅当 stride>row_bytes 时存在；与原始逐行 memset 等价。
     * 有效列区域随后会被 resize/填边完整覆盖，这里整块清零最简单。 */
    if ((td_u32)row_bytes < stride)
        memset(dev_ptr, 0, valid_size);

    cv::warpAffine(image, dev_img, affine_mat_, cv::Size(net_w_, net_h_),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (cfg_.bgr2rgb)
        cv::cvtColor(dev_img, dev_img, cv::COLOR_BGR2RGB);

    /* cached MMZ：CPU 写完刷 cache，NPU 才能读到（与原始一致） */
    svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, valid_size);
    if (flush_ret != SVP_ACL_SUCCESS) {
        PLOGE("mem flush failed 0x%x", flush_ret);
        return -1;
    }

    ret = sample_common_svp_npu_update_input_data_buffer_info(
        dev_ptr, valid_size, stride, 0, &task_info_);
    if (ret != TD_SUCCESS) {
        PLOGE("update input buffer failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* ================================================================== *
 *  inference —— 与 Detector 完全一致
 * ================================================================== */
int PoseEstimator::inference()
{
    td_s32 ret = sample_common_svp_npu_model_execute(&task_info_);
    if (ret != TD_SUCCESS) {
        PLOGE("model execute failed 0x%x", ret);
        return -1;
    }
    return 0;
}

int PoseEstimator::argmax(const float* data, int length, float& max_value)
{
    if (data == nullptr || length <= 0) return -1;
    int index = 0;
    max_value = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_value) { max_value = data[i]; index = i; }
    }
    return index;
}

cv::Point2f PoseEstimator::map_point_to_original(float x, float y) const
{
    if (inverse_affine_mat_.empty()) return cv::Point2f(-1.f, -1.f);
    return cv::Point2f(
        (float)(inverse_affine_mat_.at<double>(0, 0) * x
              + inverse_affine_mat_.at<double>(0, 1) * y
              + inverse_affine_mat_.at<double>(0, 2)),
        (float)(inverse_affine_mat_.at<double>(1, 0) * x
              + inverse_affine_mat_.at<double>(1, 1) * y
              + inverse_affine_mat_.at<double>(1, 2)));
}

int PoseEstimator::postProcess(cv::Size src_img_size, const cv::Rect2f& body_box,
                               PoseResult& result)
{
    result = PoseResult{};
    result.box = body_box;
    const float* simcc_x = nullptr;
    const float* simcc_y = nullptr;
    size_t simcc_x_stride = 0;
    size_t simcc_y_stride = 0;
    for (int output_idx = 0; output_idx < 2; ++output_idx) {
        td_u8* ptr = nullptr;
        td_u32 size = 0, stride = 0;
        if (sample_common_svp_npu_get_output_data_buffer_info(
                &task_info_, output_idx, &ptr, &size, &stride) != TD_SUCCESS || ptr == nullptr) {
            PLOGE("get output %d failed", output_idx);
            return -1;
        }
        if (svp_acl_rt_mem_invalidate(ptr, size) != SVP_ACL_SUCCESS) {
            PLOGE("invalidate output %d failed", output_idx);
            return -1;
        }
        sample_svp_npu_model_info* info = sample_common_svp_npu_get_model_info(POSE_MODEL_IDX);
        svp_acl_mdl_io_dims dims{};
        if (!info || !info->model_desc
            || svp_acl_mdl_get_output_dims(info->model_desc, output_idx, &dims) != SVP_ACL_SUCCESS) {
            PLOGE("get output %d dims failed", output_idx);
            return -1;
        }
        size_t elems = 1;
        for (size_t d = 0; d < dims.dim_count; ++d) elems *= (size_t)dims.dims[d];
        const size_t row_stride = stride > 0 ? stride / sizeof(float) : 0;
        if (elems == (size_t)NUM_KEYPOINTS * kSimccXLength) {
            simcc_x = reinterpret_cast<const float*>(ptr);
            simcc_x_stride = row_stride >= kSimccXLength ? row_stride : kSimccXLength;
        } else if (elems == (size_t)NUM_KEYPOINTS * kSimccYLength) {
            simcc_y = reinterpret_cast<const float*>(ptr);
            simcc_y_stride = row_stride >= kSimccYLength ? row_stride : kSimccYLength;
        } else {
            PLOGE("unexpected RTMPose output %d elements=%zu", output_idx, elems);
            return -1;
        }
    }
    if (simcc_x == nullptr || simcc_y == nullptr) return -1;
    float score_sum = 0.f;
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
        float max_x = 0.f, max_y = 0.f;
        const int ix = argmax(simcc_x + k * simcc_x_stride, kSimccXLength, max_x);
        const int iy = argmax(simcc_y + k * simcc_y_stride, kSimccYLength, max_y);
        if (ix < 0 || iy < 0) continue;
        const cv::Point2f p = map_point_to_original(ix / kSimccSplitRatio,
                                                    iy / kSimccSplitRatio);
        const float score = std::min(max_x, max_y);
        result.keypoints[k].x = p.x;
        result.keypoints[k].y = p.y;
        result.keypoints[k].confidence = score;
        const bool inside = p.x >= 0.f && p.x < src_img_size.width
                         && p.y >= 0.f && p.y < src_img_size.height;
        if (inside && score >= cfg_.kp_conf_thresh) {
            ++result.valid_kp_count;
            score_sum += score;
        }
    }
    result.score = result.valid_kp_count > 0 ? score_sum / result.valid_kp_count : 0.f;
    return result.valid_kp_count > 0 ? 0 : -1;
}

/* ================================================================== *
 *  destroy
 * ================================================================== */
void PoseEstimator::destroy()
{
    PLOGI("destroy called, inited=%d", inited_);

    if (task_initialized_) {
        deinit_task_();
        task_initialized_ = false;
    }
    if (model_loaded_) {
        unload_model_();
        model_loaded_ = false;
    }

    inited_ = false;
    PLOGI("destroyed");
}


// ════════════════════════════════════════════════════════════
// find_best_match: 根据 IoU 找与给定 bbox 最匹配的 PoseResult
// ════════════════════════════════════════════════════════════
int PoseEstimator::find_best_match(
    const std::vector<PoseResult>& poses,
    const cv::Mat& bbox,
    float iou_thresh)
{
    if (poses.empty() || bbox.empty()) return -1;

    float bx1 = bbox.at<float>(0, 0);
    float by1 = bbox.at<float>(0, 1);
    float bx2 = bbox.at<float>(0, 2);
    float by2 = bbox.at<float>(0, 3);       
    Rect_<float> query(bx1, by1, bx2 - bx1, by2 - by1);

    int best_idx = -1;
    float best_iou = iou_thresh;

    for (int i = 0; i < (int)poses.size(); ++i) {
        float iou = box_iou(query, poses[i].box);
        if (iou > best_iou) {
            best_iou = iou;
            best_idx = i;
        }
    }

    return best_idx;
}


/* ================================================================== *
 *  —— ACL / 模型 / Task 生命周期函数（与 Detector 完全一致） ——
 * ================================================================== */
int PoseEstimator::load_model_()
{
    td_s32 ret = sample_common_svp_npu_load_model(
        cfg_.model_path.c_str(), POSE_MODEL_IDX, TD_FALSE);
    if (ret != TD_SUCCESS) {
        PLOGE("load model failed %s 0x%x", cfg_.model_path.c_str(), ret);
        return -1;
    }
    // PLOGI("model loaded: %s", cfg_.model_path.c_str());
    return 0;
}

void PoseEstimator::unload_model_()
{
    sample_common_svp_npu_unload_model(POSE_MODEL_IDX);
}

int PoseEstimator::init_task_()
{
    td_s32 ret;

    ret = sample_common_svp_npu_create_input(&task_info_);
    if (ret != TD_SUCCESS) {
        PLOGE("create input failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_output(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_input(&task_info_);
        PLOGE("create output failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_task_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input (&task_info_);
        PLOGE("create task buf failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_work_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_task_buf(&task_info_);
        sample_common_svp_npu_destroy_output  (&task_info_);
        sample_common_svp_npu_destroy_input   (&task_info_);
        PLOGE("create work buf failed 0x%x", ret); return -1;
    }

    // PLOGI("task init ok");
    return 0;
}

void PoseEstimator::deinit_task_()
{
    sample_common_svp_npu_destroy_work_buf(&task_info_);
    sample_common_svp_npu_destroy_task_buf(&task_info_);
    sample_common_svp_npu_destroy_output  (&task_info_);
    sample_common_svp_npu_destroy_input   (&task_info_);
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}
