#include "PoseEstimator.h"
#include "ModelProfiler.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;
using namespace VISION_ENGINE;

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

// ════════════════════════════════════════════════════════════
// PoseEstimator 构造/析构
// ════════════════════════════════════════════════════════════
PoseEstimator::PoseEstimator()
    : initialized_(false)
    , picDevBuffer_(nullptr)
    , devBufferSize_(0)
{
}

PoseEstimator::~PoseEstimator() {
    release();
}

// ════════════════════════════════════════════════════════════
// PoseEstimator::init — ACL 模型加载
// ════════════════════════════════════════════════════════════
int PoseEstimator::init(const PoseEstimatorConfig& config) {
    if (initialized_) return 0;
    release();
    config_ = config;

    // 从内存加载模型
    Result ret = modelProcess_.LoadModelFromMem(config.bufferModel, config.sizeModel);
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: LoadModelFromMem failed");
        release();
        return -1;
    }

    ret = modelProcess_.CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: CreateModelDesc failed");
        release();
        return -1;
    }

    // 获取输入维度
    ret = modelProcess_.GetInputDimsByIndex(0, inputDims_);
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: GetInputDimsByIndex failed");
        release();
        return -1;
    }
    if (inputDims_.size() != 4) {
        ERROR_LOG("PoseEstimator: invalid input dims size=%zu", inputDims_.size());
        release();
        return -1;
    }
    if (inputDims_[1] == 3 || inputDims_[1] == 1)
        netSize_ = Size(inputDims_[3], inputDims_[2]);
    else
        netSize_ = Size(inputDims_[2], inputDims_[1]);
    if (netSize_.width != kInputWidth || netSize_.height != kInputHeight) {
        ERROR_LOG("PoseEstimator: unexpected RTMPose input %dx%d, expected %dx%d",
                  netSize_.width, netSize_.height, kInputWidth, kInputHeight);
        release();
        return -1;
    }
    INFO_LOG("PoseEstimator RTMPose input size: %dx%d", netSize_.width, netSize_.height);

    // 获取输入大小
    ret = modelProcess_.GetInputSizeByIndex(0, devBufferSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: GetInputSizeByIndex failed");
        release();
        return -1;
    }
    const size_t expected_input_size = (size_t)kInputWidth * kInputHeight * 3;
    if (devBufferSize_ < expected_input_size) {
        ERROR_LOG("PoseEstimator: AIPP input buffer too small, bytes=%zu, expected>=%zu",
                  devBufferSize_, expected_input_size);
        release();
        return -1;
    }

    // 分配设备缓冲区
    aclError aclRet = aclrtMalloc(&picDevBuffer_, devBufferSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("PoseEstimator: malloc device buffer failed, size=%zu", devBufferSize_);
        picDevBuffer_ = nullptr;
        release();
        return -1;
    }

    ret = modelProcess_.CreateInput(picDevBuffer_, devBufferSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: CreateInput failed");
        release();
        return -1;
    }

    ret = modelProcess_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: CreateOutput failed");
        release();
        return -1;
    }

    initialized_ = true;
    INFO_LOG("[Pose] init");
    return 0;
}

// ════════════════════════════════════════════════════════════
// PoseEstimator::run
// ════════════════════════════════════════════════════════════
int PoseEstimator::run(const Mat& image, const Mat& body_box, PoseResult& result) {
    if (!initialized_) {
        ERROR_LOG("PoseEstimator: not initialized");
        return -1;
    }
    if (image.empty()) {
        ERROR_LOG("PoseEstimator: empty image");
        return -1;
    }

    if (body_box.empty() || body_box.cols < 4) {
        ERROR_LOG("PoseEstimator: invalid body box");
        return -1;
    }
    result = PoseResult{};
    Rect2f roi(body_box.at<float>(0, 0), body_box.at<float>(0, 1),
               body_box.at<float>(0, 2) - body_box.at<float>(0, 0),
               body_box.at<float>(0, 3) - body_box.at<float>(0, 1));
    if (roi.width <= 1.f || roi.height <= 1.f) return -1;
    fxprof::add_call(fxprof::Model::Pose);

    int ret;
    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Pre);
        ret = preProcess(image, roi);
    }
    if (ret < 0) return -1;

    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Infer);
        ret = inference();
    }
    if (ret < 0) return -1;

    {
        fxprof::ScopedPhase _p(fxprof::Model::Pose, fxprof::Phase::Post);
        ret = postProcess(image.size(), roi, result);
    }
    if (ret < 0) return -1;

    return 0;
}

// ════════════════════════════════════════════════════════════
// RTMPose TopdownAffine：BODY xyxy → 192×256，越界区域填 0。
// ════════════════════════════════════════════════════════════
bool PoseEstimator::build_topdown_affine(const Rect2f& body_box) {
    constexpr float kBBoxPadding = 1.25f;
    const float center_x = body_box.x + body_box.width * 0.5f;
    const float center_y = body_box.y + body_box.height * 0.5f;
    float scale_w = body_box.width * kBBoxPadding;
    float scale_h = body_box.height * kBBoxPadding;
    const float aspect = (float)netSize_.width / (float)netSize_.height;
    if (scale_w > scale_h * aspect) scale_h = scale_w / aspect;
    else scale_w = scale_h * aspect;
    if (scale_w <= 1.f || scale_h <= 1.f) return false;
    const float sx = (float)netSize_.width / scale_w;
    const float sy = (float)netSize_.height / scale_h;
    affineMat_ = (Mat_<double>(2, 3) <<
        sx, 0.0, (double)netSize_.width * 0.5 - center_x * sx,
        0.0, sy, (double)netSize_.height * 0.5 - center_y * sy);
    invertAffineTransform(affineMat_, inverseAffineMat_);
    return true;
}

int PoseEstimator::preProcess(const Mat& image, const Rect2f& body_box) {
    if (image.type() != CV_8UC3 || !build_topdown_affine(body_box)) return -1;
    Mat dst;
    warpAffine(image, dst, affineMat_, netSize_, INTER_LINEAR,
               BORDER_CONSTANT, Scalar(0, 0, 0));

    if (config_.bgr2rgb)
        cvtColor(dst, dst, COLOR_BGR2RGB);
    if (!dst.isContinuous()) dst = dst.clone();

    const size_t data_len = dst.total() * dst.elemSize();
    if (data_len != (size_t)kInputWidth * kInputHeight * 3
        || data_len > devBufferSize_) {
        ERROR_LOG("PoseEstimator: invalid input bytes=%zu buffer=%zu",
                  data_len, devBufferSize_);
        return -1;
    }
    Result ret = Utils::MemcpyHostToDevice(dst.data, picDevBuffer_, data_len);
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: memcpy to device failed");
        return -1;
    }

    return 0;
}

// ════════════════════════════════════════════════════════════
// inference
// ════════════════════════════════════════════════════════════
int PoseEstimator::inference() {
    Result ret = modelProcess_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("PoseEstimator: inference failed");
        return -1;
    }
    return 0;
}

// ════════════════════════════════════════════════════════════
int PoseEstimator::argmax(const float* data, int length, float& max_value) {
    if (data == nullptr || length <= 0) return -1;
    int index = 0;
    max_value = data[0];
    for (int i = 1; i < length; ++i) {
        if (data[i] > max_value) { max_value = data[i]; index = i; }
    }
    return index;
}

Point2f PoseEstimator::map_point_to_original(float x, float y) const {
    if (inverseAffineMat_.empty()) return Point2f(-1.f, -1.f);
    return Point2f(
        (float)(inverseAffineMat_.at<double>(0, 0) * x
              + inverseAffineMat_.at<double>(0, 1) * y
              + inverseAffineMat_.at<double>(0, 2)),
        (float)(inverseAffineMat_.at<double>(1, 0) * x
              + inverseAffineMat_.at<double>(1, 1) * y
              + inverseAffineMat_.at<double>(1, 2)));
}

int PoseEstimator::postProcess(Size src_img_size, const Rect2f& body_box,
                               PoseResult& result) {
    result = PoseResult{};
    result.box = body_box;
    vector<vector<float>> output;
    if (modelProcess_.OutputModelResult(output) != SUCCESS) return -1;
    const vector<float>* simcc_x = nullptr;
    const vector<float>* simcc_y = nullptr;
    const size_t expected_x = (size_t)NUM_KEYPOINTS * kSimccXLength;
    const size_t expected_y = (size_t)NUM_KEYPOINTS * kSimccYLength;
    for (const auto& tensor : output) {
        if (tensor.size() == expected_x) simcc_x = &tensor;
        else if (tensor.size() == expected_y) simcc_y = &tensor;
    }
    if (simcc_x == nullptr || simcc_y == nullptr) {
        ERROR_LOG("PoseEstimator: invalid SimCC outputs count=%zu", output.size());
        return -1;
    }
    float score_sum = 0.f;
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
        float max_x = 0.f, max_y = 0.f;
        int ix = argmax(simcc_x->data() + k * kSimccXLength, kSimccXLength, max_x);
        int iy = argmax(simcc_y->data() + k * kSimccYLength, kSimccYLength, max_y);
        if (ix < 0 || iy < 0) continue;
        Point2f p = map_point_to_original(ix / kSimccSplitRatio,
                                          iy / kSimccSplitRatio);
        float score = std::min(max_x, max_y);
        result.keypoints[k].x = p.x;
        result.keypoints[k].y = p.y;
        result.keypoints[k].confidence = score;
        bool inside = p.x >= 0.f && p.x < src_img_size.width
                   && p.y >= 0.f && p.y < src_img_size.height;
        if (inside && score >= config_.kpConfThresh) {
            ++result.valid_kp_count;
            score_sum += score;
        }
    }
    result.score = result.valid_kp_count > 0
                 ? score_sum / result.valid_kp_count : 0.f;
    return result.valid_kp_count > 0 ? 0 : -1;
}

// ════════════════════════════════════════════════════════════
// find_best_match: 根据 IoU 找与给定 bbox 最匹配的 PoseResult
// ════════════════════════════════════════════════════════════
int PoseEstimator::find_best_match(
    const vector<PoseResult>& poses,
    const Mat& bbox,
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

// ════════════════════════════════════════════════════════════
// release
// ════════════════════════════════════════════════════════════
void PoseEstimator::release() {
    const bool had_resources = initialized_ || picDevBuffer_ != nullptr;
    initialized_ = false;
    modelProcess_.Release();
    if (picDevBuffer_) {
        aclError ret = aclrtFree(picDevBuffer_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("[Pose] free input buffer failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        picDevBuffer_ = nullptr;
    }
    devBufferSize_ = 0;
    inputDims_.clear();
    if (had_resources) INFO_LOG("[Pose] release");
}
