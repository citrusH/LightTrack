#pragma once

#ifndef LIGHT_TRACKER_HPP
#define LIGHT_TRACKER_HPP

#include "SvpAclRuntime.h"
#include "KalmanBoxTracker.h"
#include "Detector.h"
#include "Facekps.h"
#include "FaceRecognitionSystem.h"
#include "PersonReID.h"
#include "PoseEstimator.h"
#include <vector>
#include <memory>
#include <deque>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#ifdef USE_HISI_IVE
#include "IveGmc.h"   // GMC 的 IVE 硬件实现（仅 USE_HISI_IVE 编译）
#endif

// ════════════════════════════════════════════════════════════════════
// 本轮改造标签索引（与 .cpp 中的 [N#] 注释一一对应）：
//  [N1]  帧率无关化：盲跟时长/长盲门槛/teleport 预算全部改墙钟 ms（原帧计数在
//        27~374ms 帧间隔波动下语义漂移 14 倍）。
//  [N2]  spatial_cont 增加"归一化中心距"第三臂 + 统一空间门参考点（lead 优先）。
//        无 GMC 时云台移动帧 KF 预测框系统性滞后，纯 IoU 臂会误杀真目标。
//  [N3]  遮挡 onset 的 overlap 参考框改用主目标 last_observation（仅滞后≤1帧），
//        不再用滞后的 KF 预测框 → 云台移动期 FSM 也能正确武装危险期防线。
//  [N4]  暂定闸（provisional hold）期间，被延迟的候选检测从二级轨迹收集中排除，
//        防止主目标检测被写进"他人"轨迹/emb。
//  [N5]  occ_kf_clean_ 改 sticky 语义：遮挡窗口内一旦被非人脸观测更新即置脏、
//        直到下次 onset 才复位（原实现会被下一帧纯预测"漂白"）。
//  [N6]  人脸最小尺寸门（识别 / 注册 / sweep 统一 14px）+ 初始注册复用延迟注册的质量评分
//        （质量不达标 → 不注册，交给延迟通道）。
//  [N7]  剩余身份相关寿命改 ms：轨迹寿命、RECOVERING 时长、可见度迟滞、人脸注册间隔、
//        main_present/front_follow/incumbent/onset 新鲜度双闸。
//  [N8]  emergence 浮现点加 ms 时效（零检测帧不进匹配函数 → 原值会无限陈旧）。
//  [N9]  NullSink::verbose 改 static constexpr（兑现编译期消除）；kMatchTrace
//        改为 64KB 缓冲文本文件 + 事件触发/心跳 flush（UART 阻塞输出退出热路径）。
//  [N10] 自运动前馈（无 GMC 的替代先验）：相机运动由我们自己的输出引起、方向恒指
//        向画面中心 → 用 -β·e 预测本帧表观位移，平移 lead/smooth/emergence 参考点
//        并放宽相关门预算；β 在线估计。只动"软参考点"，绝不碰 KF 状态。
//  [N11] 头锚定瞄准点 get_aim_point()：App 端云台应把该点（而非框中心）驱动到画面
//        中心 → 消除"家具自下而上遮挡/半身框"导致的 tilt 抖动。
//  [N12] 过载降级模式：frame_dt 持续 >250ms 时自动收紧 ReID/人脸预算，
//        打断"慢帧 → 危险期逐帧推理 → 更慢"的正反馈螺旋。
//  [N13] Pose 改为候选级 0/1/2 按需调度，OCCLUDED 不自动提频。
//  [N14] 死代码清理：颜色直方图函数/常量、compute_candidate_pose_score、
//        kStationaryPxSec / kHeadVetoIou / kFaceLockTTL / kMaxOcclusionFrames /
//        kAlertTimeout / kSecAssocColorMin / kSecExclColorMin / kSecColorHistAlpha。
// ════════════════════════════════════════════════════════════════════


struct MatchQualityRecord {
    float reid_sim;
    float anchor_sim;
    float total_score;
    bool  from_face;
    int   frame_id;
};

struct MainNonMainSplit {
    cv::Mat main_trks;           // [1,4] 行向量，float类型
    cv::Mat nm_trks;             // [n,4] 矩阵，float类型
    std::vector<int> main_idx;
    std::vector<int> nm_idx;
};

struct TrackerInfo {
    cv::Mat trks;               // [n,4] 矩阵，float类型
    cv::Mat velocities;         // [n,2] 矩阵，float类型
    cv::Mat speeds;             // [n,1] 列向量或[n,] 矩阵，float类型
    cv::Mat last_boxes;         // [n,4] 矩阵，float类型
    cv::Mat k_observations;     // [n,4] 矩阵，float类型
};



struct LightTrackerConfig {
    float det_thresh = 0.7;
    float appearance_thresh = 0.7;
    int max_age = 30;            // [N7] 帧数兜底；主寿命判定改 kTrackMaxAgeMs（墙钟）
    int min_hits = 3;
    float iou_threshold = 0.4;
    int delta_t = 3;
    float inertia = 0.2;
    float gate = 0.2;
};

// 新增输出结构体，替代单独的 near_box
struct ProximityInfo {
    cv::Mat match_candidates;   // 宽范围，用于匹配，6列 [x1,y1,x2,y2,source,index]
    cv::Mat all_candidates;     // 全画面人体候选，同格式；供丢失/拥挤期跨帧公平 ReID 扫描
    int     close_det_count;    // 紧贴主目标的检测框数量（距离），用于 embedding 保护
    int     overlap_count;      // 与主目标检测框 IoU > 阈值的数量（真正遮挡）
};



// ============================================================
// 匹配结果 + 质量评估
// ============================================================
enum class MeasurementReliability {
    NONE,
    UNCERTAIN,
    RELIABLE
};

struct MainMatchResult {
    bool matched = false;
    int  source  = -1;
    int  index   = -1;
    float reid_sim    = 0.f;
    float anchor_sim  = 0.f;
    float total_score = 0.f;
    float head_match  = 0.f;
    bool  has_head    = false;
    // 已通过「绝对 ReID + 已评估次优候选分差」的强外观直证。
    // 后续重捕/KF 信任必须复用该结论，不能仅按绝对分数再次放行。
    bool  strong_reid_direct = false;
    bool  from_face   = false;
    bool  face_from_sweep = false;  // B1：人脸命中来自全画面扫描（须走暂定闸，不即时硬覆盖副作用）
    bool  from_global_body_scan = false; // 全图人体探索槽命中；普通强分需同空间假设复验
    int   body_hyp_id = -1;          // 跨帧物理 BODY hypothesis；detector source/index 不可代替
    bool  provisional_continuation = false; // 身份 continuation 已成立，尚需 measurement 安全门提交
    bool  provisional_weak_spatial = false;
    float provisional_center_dist = -1.f;
    float provisional_box_iou = -1.f;
    float provisional_size_ratio = -1.f;
    MeasurementReliability reliability = MeasurementReliability::NONE;
    cv::Mat emb;                    // C2：赢家候选的 ReID 特征（人脸覆盖后失效时为空 → 下游按需重算）
};

// 分离 OKS（位置跟踪信号）和 body_shape（身份识别信号）
struct PoseScoreDetail {
    float oks         = 0.f;   // 关键点空间吻合度 — 帧间位置追踪
    float body_shape  = 0.f;   // 骨骼比例余弦相似度 — 人体结构身份指纹
    bool  has_oks     = false;
    bool  has_shape   = false;
    // ── 肩中点（供转身期肩部连续性信号；免费，复用已匹配的候选 pose）──
    cv::Point2f shoulder_mid = cv::Point2f(-1.f, -1.f);   // 候选肩中点（像素）
    float shoulder_w   = 0.f;                             // 候选肩宽（像素）
    bool  has_shoulder = false;
};

// [N11] 云台瞄准点（App 消费）：位置 + 是否有效 + 当前输出控制权重。
//   App 应把 (x,y) 驱动到画面中心；真实身体/严格接受的头脸观测均为 1，0=无输出。
struct AimPoint {
    float x = -1.f, y = -1.f;
    bool  valid  = false;
    float weight = 0.f;
};

class TrackerRuntimeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};


class LightTracker {
private:
    float det_thresh;
    float appearance_thresh;
    int max_age;
    int min_hits;
    float iou_threshold;
    int delta_t;
    float inertia;
    float gate;

    // 必须先于所有模型声明，使其在模型成员全部析构后最后释放 ACL。
    SvpAclRuntime acl_runtime_;
    CFaceKeypoint106 detector_fk;
    FaceRecognitionSystem face_recognizer;
    PersonReID_PCB emb_model;
    Detector detector;
    PoseEstimator pose_estimator;
    bool initialized_ = false;

    void release_models();

    enum class PoseReason { TARGET_REFRESH, AMBIGUITY_RESOLVE };
    enum class PoseRequestStatus { AVAILABLE, FAILED, BUDGET_DENIED };
    struct PoseCacheEntry {
        int source = -1;
        int index = -1;
        cv::Mat body_box;
        PoseResult pose{};
        PoseRequestStatus status = PoseRequestStatus::FAILED;
    };
    std::vector<PoseCacheEntry> pose_cache_;
    int pose_budget_used_ = 0;
    static constexpr int kPoseBudgetPerFrame = 2;
    int last_committed_pose_frame_ = -1;
    int64_t last_committed_pose_ms_ = -1;
    cv::Mat committed_pose_box_;

    int frame_count;

    LightTrackerConfig config;
    std::vector<std::shared_ptr<KalmanBoxTracker>> trackers;

    cv::Mat compute_embedding(const cv::Mat& img,
        const cv::Mat& bbox);  // bbox为[1,4]行向量，返回[1, n]行向量

    int main_track_unmatched_time;

    int64_t last_frame_time = 0;
    // 当前输入帧的单调时间戳。每次 update() 入口覆盖，使同帧 BODY/HEAD/FACE
    // MotionObservation 共用捕获时刻，不把不同模型路径的处理耗时误当成人体运动。
    int64_t current_frame_timestamp_ms_ = -1;
    // 帧间隔（秒，由时间戳测得）：驱动 KF 的时间尺度，并用于帧率无关空间门。
    // 首帧/异常间隔回退 kDefaultDtSec。
    float frame_dt_sec_ = 0.04f;
    static constexpr float kDefaultDtSec = 0.04f;   // 回退帧间隔（≈25fps）

    // ── [N1] 真盲时长（墙钟 ms，取代原 main_blind_frames_ 帧计数）──
    //   语义：距离主目标最后一次"真实观测"（真实身体、头部连续性或人脸确认）的墙钟
    //   毫秒数：blind_ms() = now − last_real_obs_ms_（<0 视为极大，即从未观测）。
    //   消费者：C-identity 长盲门（≥kReacqProbationMs）、teleport 预算随盲时长增长、
    //   [MATCH] 诊断行。帧计数在 27~374ms 帧间隔波动下语义漂移 14 倍，故弃用。
    //   历史 MainTargetPredictor 已删除。短时身份不可观测窗口可由安全真实 Motion History
    //   启动有限时间/距离的 PTZ-only 预测框；它不刷新本真实观测时钟或任何身份状态。
    int64_t last_real_obs_ms_ = -1;
    int64_t last_body_observation_ms_ = -1;
    // 头部轨迹与人脸身份必须分开：二者都可证明遮挡状态仍有真实观测，但只有
    // 人脸是身份直证，头部不能借此放宽人体 anchor/ReID 身份门。
    int64_t last_head_continuity_ms_ = -1;
    int64_t last_face_identity_ms_ = -1;
    // 仅 BODY 绑定的人脸在本帧真正通过 FaceReco 后刷新。face-only 身份命中不能
    // 冒充“当前 BODY 已确认”，否则错误 BODY 只要包住真脸就会关闭全局扫脸。
    int64_t last_body_face_identity_ms_ = -1;
    static constexpr int64_t kPartObsRecentMs = 350;
    // 完全没有身体/头/脸真实观测超过此时长后，人体/头部 KF 停止速度外推，
    // 但继续推进轨迹寿命并保留全画面身份重捕。下一强观测会直接重建数值状态。
    static constexpr int64_t kKfLongBlindFreezeMs = 1500;

    int face_recognition_every_n_frames = 15;

    // ── 性能：减少每帧模型推理（仅 CLEAR 期生效；危险/拥挤/警报自动回全量）──
    static constexpr int kReidMaxCandClear = 3;   // CLEAR 期最多对前 K 个候选跑 ReID（新 ReID 更可信，适度增加预算）
    static constexpr int   kReidMaxCandDanger = 4;
    static constexpr float kPrerankDistW      = 0.5f;
    static constexpr float kPrerankHeadW      = 0.5f;
    // 多人/丢失期人体全图轮转：Detector 仍只跑一次，人体 ReID 总预算仍为上面的 3/4。
    // 局部槽维持原轨迹能力，探索槽按“最久未推理 + 以上次可信位置由近及远”覆盖全画面。
    static constexpr int     kBodyReidExploreSlots       = 2;
    static constexpr int     kBodyReidLongBlindLocalSlots = 1;
    static constexpr int64_t kBodyReidLongBlindMs        = 700;
    static constexpr int     kBodyReidHypMaxAgeFrames    = 8;
    static constexpr float   kBodyReidHypMatchDiag       = 1.25f;
    static constexpr float   kBodyReidHypAmbigCostGap    = 0.12f;
    struct BodyReidHypothesis {
        int id = -1;
        float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
        int first_seen_frame = -1;
        int last_seen_frame = -1;
        int last_attempt_frame = -1000000;
        int last_inferred_frame = -1000000;
    };
    std::vector<BodyReidHypothesis> body_reid_hypotheses_;
    int  next_body_reid_hyp_id_ = 1;
    bool body_reid_global_active_ = false;
    int  body_reid_scan_epoch_ = 0;
    int  body_reid_scan_epoch_frame_ = -1;
    // 搜索圆心只接受初始指定、可靠单人 BODY、强 ReID/anchor 或人脸身份观测。
    // 弱 BODY/纯 KF 预测不会移动它，避免错锁路人后把全图扫描永久带偏。
    float body_reid_anchor_cx_ = -1.f;
    float body_reid_anchor_cy_ = -1.f;
    float body_reid_anchor_diag_ = -1.f;
    int64_t body_reid_anchor_ms_ = -1;
    static constexpr int kPoseInferEveryN  = 3;
    static constexpr int kFaceMaxCand      = 3;   // 每次人脸验证最多候选数（CLEAR 期）
    static constexpr int kFaceDangerEveryN = 3;   // 危险期人脸每 N 帧（常规节流）
    static constexpr int kFaceDangerMaxCand = 4;  // 危险期人脸候选上限

    // ── 10 FPS 帧级模型预算 ──
    // 人脸与候选 Pose 分别有独立硬预算，Pose 单帧最多两次。
    // 所有人脸路径（近场/扫脸/独立脸/延迟注册）共享 face_model_budget_，
    // 不允许各分支分别叠加预算。
    static constexpr int kFaceBudgetNormal   = 1;
    static constexpr int kFaceBudgetPriority = 3;
    int  face_model_budget_ = kFaceBudgetNormal; // 剩余“FaceKps + FaceReco”配额
    bool face_priority_frame_ = false;
    int  face_priority_streak_ = 0;
    bool frame_allow_secondary_reid_ = false;    // 二级 ReID 只使用普通帧余量
    enum class IdentityEvidence { UNKNOWN, POSITIVE, NEGATIVE };
    struct FaceInferenceCacheEntry {
        cv::Rect box;             // xyxy 语义
        float sim = 0.f;
        std::string name = "unknown";
        IdentityEvidence evidence = IdentityEvidence::UNKNOWN;
        std::string reason = "not_evaluated";
    };
    // 同一检测脸可能先出现在人体候选、后又进入部件恢复；每帧只允许实际推理一次。
    std::vector<FaceInferenceCacheEntry> face_inference_cache_;

    // ── [N12] 过载降级模式（打断"慢帧→更多推理→更慢"正反馈）──
    //   frame_dt 连续 kOverloadOnN 帧 > kOverloadDtHi → 进入降级：
    //   ReID/人脸预算收紧、全画面扫描停、二级 ReID 轮询停。
    //   连续 kOverloadOffN 帧 < kOverloadDtLo → 退出。全部是"预算收紧"，无状态污染。
    bool overload_mode_ = false;
    int  overload_hi_ = 0, overload_lo_ = 0;
    static constexpr float kOverloadDtHi       = 0.25f;
    static constexpr float kOverloadDtLo       = 0.15f;
    static constexpr int   kOverloadOnN        = 3;
    static constexpr int   kOverloadOffN       = 5;

    // ── 延迟/重新人脸注册 ──
    bool face_registered_ = false;             // 是否已成功注册过人脸
    // [N7] 注册间隔改墙钟（原帧计数：15f 慢帧下=5.6s、快帧下=0.4s，语义漂移）
    int64_t last_face_register_ms_ = -999999;  // 上次尝试注册的时刻（ms）
    static constexpr int64_t kFaceRegisterRetryMs = 600;    // 未注册时：每 600ms 尝试（≈15f@25fps）
    static constexpr int64_t kFaceReregisterMs    = 12000;  // 已注册后：每 12s 重注册（≈300f@25fps）

    // ── [N6] 人脸尺寸门 + 初始注册质量门 ──
    //   识别门：脸高 < kFaceRecogMinFacePx 识别必败 → 跳过（省 2 次 NPU/脸 + 堵死远距
    //   小脸单张误识 → face_lock/emb 大步刷新的污染入口）。三个像素门统一为 14px。
    //   初始注册（setMainTarget）复用延迟注册的质量评分：质量 < kFaceInitRegisterMinQ
    //   → 本次不注册（侧脸/小脸初注册拖垮冷启动识别率），交延迟通道补注册。
    static constexpr float kFaceRecogMinFacePx    = 14.f;
    static constexpr float kFaceRegisterMinFacePx = 14.f;
    static constexpr float kFaceInitRegisterMinQ  = 0.45f;
    // 识别同样先过五点质量门：只把朝向/完整性足够的人脸送入 FaceReco。
    // 尺寸门与注册统一为 14px，遮挡期较小但质量合格的人脸仍可进入识别。
    static constexpr float kFaceRecognitionMinQ   = 0.40f;

    // 平滑中心（PTZ 中心先验）
    float smooth_cx_ = -1.f;
    float smooth_cy_ = -1.f;
    float center_ema_alpha_ = 0.3f;

    // ── 丢失期重捕搜索门的引导中心 ──
    // 引导中心 = 上次真实观测中心；遮挡期由浮现点(emergence)推进；[N10] 每帧开头由
    // 自运动前馈平移。仅内部搜索参考，从不作为主框输出。
    float lead_cx_ = -1.f;   // 引导中心 x
    float lead_cy_ = -1.f;   // 引导中心 y

    // 外部 PTZ 已实际滑动后的本地搜索锚点。它只由最后可靠 BODY 更新；
    // blind 期绝不回写 KF、lead、motion history 或身份模板。
    enum class PtzBlindPhase { IDLE = 0, SLIDING = 1, STOPPED = 2 };
    PtzBlindPhase ptz_blind_phase_ = PtzBlindPhase::IDLE;
    bool spatial_prior_invalid_ = false;
    cv::Mat ptz_blind_anchor_box_;       // [1,4] xyxy，最后可靠 BODY
    int64_t ptz_blind_anchor_ms_ = -1;
    int ptz_blind_explore_rotor_ = 0;
    int ptz_reacq_body_streak_ = 0;

    // ── 当前主框控制权重与观测来源 ──
    // 输出权重只控制云台，不能再兼任身份/KF 信任标志。严格接受的 BODY/HEAD/FACE
    // 都是实时观测，控制权重为 1；PREDICTED 仅用于短遮挡 PTZ 连续性且随可信度
    // 衰减；NONE 为 0。ego 参数学习仍只认 BODY。
    enum class OutputSource { NONE, BODY, HEAD, FACE, PREDICTED };
    OutputSource frame_output_source_ = OutputSource::NONE;
    float coast_weight_ = 0.0f;
    // 本帧主目标 measurement 的最终语义。只有 RELIABLE 能更新主 BODY KF、
    // 刷新真实观测时钟并终止 short prediction；UNCERTAIN 仅保留候选假设。
    MeasurementReliability frame_measurement_reliability_ = MeasurementReliability::NONE;

    // ── 真实人体运动历史 ──
    // BODY/HEAD/FACE 全部使用人体 xyxy 框语义。仅最终安全接受的真实观测可写入；
    // prediction/UNCERTAIN/provisional 不得写入。同帧按 BODY > FACE_IDENTITY >
    // FACE_TRACK > HEAD 仲裁。构造时为空，真实观测后更新，reset/换目标清空。
    enum class MotionObservationSource { HEAD, FACE_TRACK, FACE_IDENTITY, BODY };
    enum class MotionQuality { LOW, MEDIUM, HIGH };
    struct MotionObservation {
        int frame_id = -1;
        int64_t timestamp_ms = -1;
        float x1 = -1.f, y1 = -1.f, x2 = -1.f, y2 = -1.f;
        MotionObservationSource source = MotionObservationSource::HEAD;
        int clip_mask = 0;
    };
    struct MotionEstimate {
        bool valid = false;
        float history_vx = 0.f, history_vy = 0.f;  // px/ms
        float residual_ratio = 0.f;
        float direction_consistency = 0.f;
        float kf_vx = 0.f, kf_vy = 0.f;            // px/ms
        float kf_direction_cos = 0.f;
        int observation_count = 0;
        int valid_pair_count = 0;
        bool kf_valid = false;
        bool kf_strong_conflict = false;
        bool clip_risk = false;
        bool source_transition_risk = false;
        MotionQuality history_quality = MotionQuality::LOW;
        MotionQuality effective_quality = MotionQuality::LOW;
    };
    static constexpr size_t kMotionHistoryMaxCount = 10;
    static constexpr int64_t kMotionHistoryMaxAgeMs = 1000;
    static constexpr int64_t kMotionAdjacentMaxGapMs = 350;
    static constexpr int64_t kMotionStartFreshnessMaxAgeMs = 500;
    static constexpr float kMotionResidualHighRatio = 0.10f;
    static constexpr float kMotionResidualMediumRatio = 0.25f;
    static constexpr float kMotionDirectionHighRatio = 0.75f;
    static constexpr float kMotionDirectionMediumRatio = 0.50f;
    static constexpr float kMotionStationaryDiagRatio = 0.005f;
    static constexpr float kMotionKfReverseCosThreshold = -0.50f;
    static constexpr float kMotionMediumVelocityScale = 0.50f;
    std::deque<MotionObservation> motion_history_;

    // 非主轨迹本帧真实 detection association。它描述 observation freshness，
    // 不属于 KF 数学状态；每帧入口清空，仅 update_secondary_tracks() 的真实配对写入。
    struct SecondaryFrameObservation {
        int frame_id = -1;
        int64_t timestamp_ms = -1;
        cv::Mat box;  // raw detector BODY，xyxy
        int detection_source = -1;
        int detection_index = -1;
        float association_iou = -1.f;
        float row_second_iou = -1.f;
        float column_second_iou = -1.f;
        int row_candidate_count = 0;
        int column_candidate_count = 0;
        float relative_selected_cost = -1.f;
        float relative_second_cost = -1.f;
        int64_t relative_previous_dt_ms = -1;
    };
    std::unordered_map<int, SecondaryFrameObservation>
        secondary_frame_observations_;

    // 主目标与某条非主轨迹同帧、同 timestamp 的双真实 BODY 相对中心历史。
    // 任意一侧 prediction/provisional/UNCERTAIN 或 secondary quarantine 都不得写入。
    // 当前只记录 validator 所需诊断量，尚不据此阻止写入。后续若启用 segment gate：
    // AMBIGUOUS 清空旧 segment 且当前点不作 seed；DISCONTINUOUS 清空旧 segment
    // 并允许当前点作新 seed；新 segment 至少 3 个可靠点/2 个有效速度才可 emergence。
    struct RelativeCenterObservation {
        int frame_id = -1;
        int64_t timestamp_ms = -1;
        float main_cx = 0.f, main_cy = 0.f;
        float secondary_cx = 0.f, secondary_cy = 0.f;
        float relative_x = 0.f, relative_y = 0.f;
        float reference_diag = 1.f;
        float main_width = 1.f, main_height = 1.f;
        float secondary_width = 1.f, secondary_height = 1.f;
        float normalized_x = 0.f, normalized_y = 0.f;
        float width_ratio = 1.f, height_ratio = 1.f;
        float secondary_aspect = 1.f;
    };
    static constexpr size_t kRelativeMotionHistoryMaxCount = 5;
    std::unordered_map<int, std::deque<RelativeCenterObservation>>
        relative_motion_history_;

    // ── 短时遮挡预测输出 ──
    // Motion History 仅在首次进入 PRED 前估计一次。ACTIVE 后完整冻结 anchor、
    // velocity、quality 与安全边界，不再读 history/KF，也不被未确认候选牵引。
    // 它只影响 PTZ 输出，不更新 KF measurement、ReID/Anchor/Face/Pose、
    // hit/tsu、真实观测时钟、Motion History 或遮挡状态机。
    static constexpr int64_t kShortPredictionMaxDurationMs = 1200;
    static constexpr float kShortPredictionNominalDtMs = 40.0f;
    static constexpr float kShortPredictionMaxBodyDiag = 0.75f;
    static constexpr float kShortPredictionMaxFrameDiag = 0.12f;
    static constexpr int64_t kShortPredictionLowHoldMs = 180;
    enum class PredictionLifecycle { IDLE, ACTIVE, EXHAUSTED };
    enum class PredictionMode { MOVE_HIGH, MOVE_MEDIUM, HOLD_LOW };
    struct FrozenPredictionState {
        PredictionLifecycle lifecycle = PredictionLifecycle::IDLE;
        PredictionMode mode = PredictionMode::HOLD_LOW;
        MotionQuality motion_quality = MotionQuality::LOW;
        float anchor_x1 = -1.f, anchor_y1 = -1.f;
        float anchor_x2 = -1.f, anchor_y2 = -1.f;
        float anchor_cx = -1.f, anchor_cy = -1.f;
        int64_t anchor_timestamp_ms = -1;
        int anchor_frame_id = -1;
        MotionObservationSource anchor_source = MotionObservationSource::HEAD;
        int64_t prediction_start_ms = -1;
        float frozen_vx = 0.f, frozen_vy = 0.f;     // px/ms
        float max_displacement = 0.f;
        float last_output_cx = -1.f, last_output_cy = -1.f;
        int64_t last_output_ms = -1;
    };
    FrozenPredictionState frozen_prediction_;

    // 每个有主目标但无可靠 BODY/HEAD/FACE 的帧互斥计数。reset/换目标清零。
    uint64_t pred_missing_real_frames_ = 0;
    uint64_t pred_move_high_frames_ = 0;
    uint64_t pred_move_medium_frames_ = 0;
    uint64_t pred_hold_low_frames_ = 0;
    uint64_t pred_output_none_frames_ = 0;

    // ── HEAD/FACE 重构框的画面裁剪与整框稳定 ──
    // 先按正常头身/脸身比例重构，再与画面求交：远距遮挡保留完整人体，近距越界
    // 自然得到画面内可见部分。裁剪框统一进入 KF/lead/身份锁和 PTZ 输出稳定层。
    // BODY→部件与部件→BODY 连续过渡；全局人脸强重定位直接 snap，不拖慢找回。
    bool    part_output_box_valid_ = false;
    float   part_output_cx_ = -1.f, part_output_cy_ = -1.f;
    float   part_output_w_ = -1.f,  part_output_h_ = -1.f;
    int64_t part_output_update_ms_ = -1;
    int64_t part_output_pending_since_ms_ = -1;
    float   last_returned_x1_ = -1.f, last_returned_y1_ = -1.f;
    float   last_returned_x2_ = -1.f, last_returned_y2_ = -1.f;
    int64_t last_returned_box_ms_ = -1;
    OutputSource last_returned_source_ = OutputSource::NONE;
    // 近距离目标通常只露肩膀以上，框底贴近画面边缘。此时转头会显著改变
    // FACE/HEAD/BODY 框尺寸，却不代表人体真的上下移动。跨来源保留一个近场状态，
    // 纵向只跟随本帧真实头/脸观测；匹配阶段仍可使用预测轨迹，但预测值绝不能
    // 驱动云台输出。无部件的短空档保持，不跟随重构比例抖动。
    bool    closeup_output_active_ = false;
    float   closeup_top_y_ = -1.f;
    float   closeup_last_head_cy_ = -1.f;
    int64_t closeup_update_ms_ = -1;
    int64_t closeup_last_head_ms_ = -1;
    float   closeup_last_face_cy_ = -1.f;
    int64_t closeup_last_face_ms_ = -1;
    int     closeup_anchor_kind_ = 0;  // 0=无，1=头，2=脸估算头顶
    static constexpr float   kPartBoxPosDeadbandFrac     = 0.08f;
    static constexpr float   kPartBoxSizeDeadbandFrac    = 0.06f;
    static constexpr float   kPartBoxDeadbandMinPx       = 3.f;
    static constexpr float   kPartBoxDeadbandMaxPx       = 16.f;
    static constexpr int64_t kPartBoxDwellMs             = 100;
    static constexpr int64_t kPartBoxBridgeMaxGapMs      = 500;
    static constexpr float   kPartBoxPosSpeedScaleSec    = 4.0f;
    static constexpr float   kPartBoxSizeSpeedScaleSec   = 6.0f;
    static constexpr float   kPartBoxBodyReturnSpeedMul  = 2.0f;
    static constexpr float   kCloseupEnterBottomFrac     = 0.07f;
    static constexpr float   kCloseupExitBottomFrac      = 0.12f;
    static constexpr float   kCloseupMinHeadHeightFrac   = 0.10f;
    static constexpr float   kCloseupMinBodyWidthFrac    = 0.28f;
    static constexpr float   kCloseupMinVisibleHeightFrac= 0.30f;
    static constexpr float   kCloseupRelocateDiagFrac    = 0.18f;
    static constexpr int     kCloseupControlHeadMaxTsu   = 0;
    static constexpr int64_t kCloseupHeadHoldMs          = 300;
    static constexpr float   kCloseupPosSpeedScaleSec    = 3.0f;
    static constexpr float   kCloseupSizeSpeedScaleSec   = 2.0f;
    static constexpr float   kCloseupHeadJitterFrac      = 0.20f;
    static constexpr float   kFaceToHeadTopFrac          = 0.25f;

    // ── [N10] 自运动前馈（无 GMC 的替代先验）──
    // 事实：相机只在我们输出 box 后才动；动的效果永远是把该 box 中心拉向画面中心。
    // 故上一次输出后本帧表观位移 ≈ -f·e（e=上次输出中心−画面中心，f=clamp(β·dt)）。
    // 每帧开头把该位移施加到 lead/smooth/emergence 三个软参考点，并把 |位移|/2 作为
    // 额外松弛加进 teleport/spatial_cont 预算。⚠ 绝不平移 KF 状态（估错→门略宽，安全）。
    // β_sec 在线估计：连续命中帧目标表观位移在 -ê 方向投影 / |e| / dt，有界 EMA。
    bool    ego_enabled_ = false;                // 默认关闭；可由 set_ego_enabled(true) 显式开启
    float   ego_ex_ = 0.f, ego_ey_ = 0.f;        // 未闭合误差向量（px；输出时置 out−center，逐帧几何衰减）
    bool    ego_active_ = false;                 // 是否存在有效未闭合误差
    int64_t ego_out_ms_ = 0;                     // 最近一次输出主框的时刻
    float   ego_beta_sec_ = 5.0f;                // 在线估计的闭合速率（1/s）；相机属性，跨 reset 保留
    float   ego_sx_ = 0.f, ego_sy_ = 0.f;        // 本帧预测表观位移（px；诊断/预算用）
    float   ego_shift_mag_ = 0.f;                // |本帧预测位移|（px）
    // 头部连续性专用的累计自运动补偿。主 KF 不接受估计位移，避免错误前馈污染身份
    // 轨迹；但头部在“无身体、云台仍跟随”时必须在当前图像坐标系中作空间匹配。
    float   head_ego_dx_ = 0.f, head_ego_dy_ = 0.f;
    float   prev_match_cx_ = 0.f, prev_match_cy_ = 0.f;   // β 估计：上个命中帧输出中心
    float   prev_match_ex_ = 0.f, prev_match_ey_ = 0.f;   // β 估计：上个命中帧误差向量
    int64_t prev_match_ms_ = -1;
    bool    prev_match_valid_ = false;
    static constexpr float   kEgoBetaSecMin      = 0.5f;
    static constexpr float   kEgoBetaSecMax      = 15.f;
    static constexpr float   kEgoBetaEma         = 0.15f;
    static constexpr float   kEgoMaxFracPerFrame = 0.85f;  // 单帧最大闭合比例（clamp β·dt）
    static constexpr float   kEgoDeadbandFrac    = 0.04f;  // 死区（×画面对角线）：|e| 小于此认为云台不动
    static constexpr int64_t kEgoTimeoutMs       = 700;    // 最近输出超时 → 前馈失效
    static constexpr int64_t kEgoObsMaxGapMs     = 500;    // β 采样：两命中帧间隔上限

    // ── [N11] 头锚定瞄准点 ──
    // 框中心在"家具自下而上遮挡/半身框"时系统性上移 → 云台误抬。改用头锚定点：
    // 头部 KF 新鲜 → 头中心下方 kAimChestFrac×参考身高；无头 → 框顶 + kAimTopFrac×参考身高。
    // 参考身高优先 main_h_hold_。每帧在输出主框后由 compute_aim_point 更新。
    float aim_x_ = -1.f, aim_y_ = -1.f;
    bool  aim_valid_ = false;
    static constexpr float kAimChestFrac = 0.18f;
    static constexpr float kAimTopFrac   = 0.30f;

    bool reset_flag = false;

    // ── 遮挡状态管理（宁可不匹配也不匹配错；不可靠 measurement → KF 纯预测，
    //    稳定快照存在时仅向 PTZ 输出有限 short prediction）──
    enum class OcclusionState { CLEAR, OCCLUDED, RECOVERING };
    OcclusionState occlusion_state_ = OcclusionState::CLEAR;
    int occlusion_start_frame_ = 0;
    int64_t recovery_start_ms_ = 0;                // [N7] RECOVERING 起始时刻（原帧计数改墙钟）
    int separation_streak_ = 0;                    // OCCLUDED 期连续"疑似已分离"帧数（F7；证据计数，保留帧单位）
    cv::Mat pre_occ_emb_;                          // 遮挡前的干净 embedding 快照
    static constexpr int64_t kRecoveryMs = 0;    // 分离后的短保护期；长恢复期会造成可见目标前/后穿人时卡顿
    static constexpr int kSeparationConfirmFrames = 1; // 分离 1 帧即恢复；不要设 0（会使条件恒成立）
    // 人物身份歧义独立于 OCC/RECOVERING 和“本帧目标已由强身份找回”。只有真实人物竞争
    // 证据才能刷新；close>=2 不能单独触发。强身份可恢复输出，但 scene risk 只能自然过期。
    enum PersonIdentityRiskReason : uint32_t {
        kPersonRiskNone             = 0,
        kPersonRiskOverlap          = 1u << 0,
        kPersonRiskMergeOnset       = 1u << 1,
        kPersonRiskOwnerCompetition = 1u << 2,
        kPersonRiskKnownOther       = 1u << 3,
        kPersonRiskAlertCompetition = 1u << 4
    };
    struct PersonIdentityAmbiguityContext {
        int64_t last_evidence_ms = -1;
        int last_evidence_frame = -1;
        int last_direct_competition_frame = -1;
        uint32_t reasons = kPersonRiskNone;
        int close_count = 0;
    };
    PersonIdentityAmbiguityContext person_identity_context_;
    static constexpr int64_t kPersonIdentityRiskHoldMs = 200;
    // B2：进入 OCCLUDED 需主目标"近期被看到"。[N7] 帧数 AND 墙钟双闸。
    static constexpr int     kOcclusionOnsetMaxTsu = 2;
    static constexpr int64_t kOcclusionOnsetMaxMs  = 300;

    // ── [N9] 匹配决策 trace：完整时间线写 64KB 缓冲文件（取代 UART）──
    //   每条 trace 先进入 stdio 用户态缓冲；事件立即 flush，平稳期每
    //   kTraceHeartbeatFrames 帧 flush。文件保留每帧 CAND/MATCH/OUTPUT，能定位算法未主动
    //   报警的静默错锁；环仍保留为固定内存窗口，供后续扩展/现场检查。
    static constexpr int kTraceRingN            = 96;
    static constexpr int kTraceLineLen          = 320;
    static constexpr int kTraceHeartbeatFrames  = 25;
    static constexpr int kTraceDumpMinGapFrames = 10;
    char trace_ring_[kTraceRingN][kTraceLineLen];
    int  trace_head_ = 0, trace_count_ = 0;
    int  trace_last_dump_frame_ = -999;
    OcclusionState trace_prev_occ_ = OcclusionState::CLEAR;
    bool trace_prev_alert_ = false;
    bool trace_event_pending_ = false;   // 各机制置位 → 本帧强制刷出
    void trace_push(const char* line);
    void trace_flush(const char* reason, bool full);

    // ── 遮挡者追踪（分离后排除法基础）──
    int  occluder_tracker_id_ = -1;
    cv::Mat pre_occ_velocity_;
    // [N5] occ_kf_clean_（sticky）："本次遮挡窗口内主 KF 是否只吃过 人脸确认/纯预测"。
    //   onset 复位 true；遮挡期任一次【非人脸】检测真正更新 KF → 置 false 并锁存到下次 onset。
    //   原实现会被后续帧纯预测重新置 true（"最后一帧干净"≠"全程干净"）→ RECOVERING 权重表漂白。
    bool occ_kf_clean_ = true;

    // ── 遮挡期"浮现点"预测（仅作恢复搜索/ReID 调度/separation HOLD 提示）──
    //   无 GMC 时朝向只来自 main/secondary 同帧双真实 BODY 的相对中心历史；浮现点
    //   不写 lead、不作身份否决或 KF measurement。[N8] 以 ms 时效防陈旧搜索提示。
    bool  emergence_valid_ = false;
    float emergence_cx_ = -1.f, emergence_cy_ = -1.f;
    float emergence_dir_x_ = 0.f, emergence_dir_y_ = 0.f;
    int64_t emergence_update_ms_ = -1;                     // [N8] 最近刷新时刻
    static constexpr float   kEmergencePushDiag = 0.6f;
    static constexpr int64_t kEmergenceMaxAgeMs = 800;     // [N8] 超此未刷新 → 浮现点作废

    // ── 主目标身体可见度（第二维度，独立于遮挡状态）──
    //   部位深度：头≈0.1<肩≈0.28<髋≈0.52<膝≈0.78<踝=1.0；可见到的最深部位=visible_ratio。
    enum class VisibilityState { FULL, MOSTLY_FULL, HALF, UPPER, HEAD_ONLY };
    VisibilityState visibility_state_   = VisibilityState::FULL;
    VisibilityState visibility_pending_ = VisibilityState::FULL;
    float visible_ratio_ema_            = 1.0f;
    int64_t visibility_pending_since_ms_ = 0;      // [N7] pending 起始时刻（原帧计数改墙钟）
    static constexpr float   kVisEmaAlpha     = 0.4f;
    static constexpr int64_t kVisHysteresisMs = 160;   // [N7] 状态切换需稳定时长（≈4f@25fps）

    // ── 人脸硬锚定（#1）──
    bool    face_locked_ = false;
    int     last_face_lock_frame_ = -999;
    cv::Mat face_lock_box_;
    static constexpr int64_t kFaceLockTTLMs   = 2400;   // 人脸锁有效期（≈60f@25fps）
    static constexpr int64_t kAlertTimeoutMs  = 3600;   // alert 自动解除（≈90f@25fps）
    static constexpr int64_t kMaxOcclusionMs  = 3600;   // OCCLUDED 超时强制回 CLEAR（≈90f）
    static constexpr int64_t kCoexistVetoMaxMs = 500;   // 共存排除新鲜度墙钟上限
    int64_t last_face_lock_ms_ = -999999;
    int64_t alert_start_ms_    = 0;
    int64_t occ_start_ms_      = 0;
    static constexpr float kFaceConfirmEmbAlpha = 0.6f; // 人脸确认刷新主 emb 的 EMA 系数

    // ── GMC 相机运动补偿（#3b；当前关闭，见 gmc_enabled_）──
    cv::Mat prev_gray_;
    float   gmc_scale_   = 1.0f;
    // 关闭后 estimate_camera_motion 早退（gmc_M 恒空），依赖世界系稳定的机制降级到安全模式：
    //   M1 运动一致性否决 → 不启用（需可信 KF 世界系速度，前馈给不了）。身份保护转交
    //   共存/头部/anchor/人脸。[N10] 自运动前馈补上"表观位移预测"，但只作用于软参考点/门预算。
    bool    gmc_enabled_ = false;            // 总开关（当前：关闭）
    static constexpr int   kGmcWorkWidth   = 640;
    static constexpr int   kGmcMinInliers  = 12;
    static constexpr float kGmcMaxScaleDev = 0.5f;

    // ── GMC Y 平面零拷贝（IVE 硬件加速）──
    unsigned long long cur_y_phys_ = 0, cur_y_virt_ = 0;
    int   cur_y_w_ = 0, cur_y_h_ = 0;
    int   cur_y_stride_ = 0;
    bool  cur_y_valid_  = false;

    // ── 匹配质量监控 ──
    std::deque<MatchQualityRecord> quality_history_;
    static constexpr int kQualityWindowSize = 30;
    float baseline_reid_ = 0.0f;
    float baseline_anchor_sim_ = 0.0f;
    int   stable_frame_count_ = 0;
    int   suspect_streak_ = 0;
    bool  id_switch_alert_ = false;
    int   alert_frame_start_ = 0;
    static constexpr int kSuspectThresh = 3;

    // ── 检测器类别约定（新模型）：0=face,1=body,2=head（与旧模型相反）──
    static constexpr int LABEL_FACE = 0;
    static constexpr int LABEL_BODY = 1;
    static constexpr int LABEL_HEAD = 2;

    // ── 头部轨迹判别（辅助遮挡期防 ID switch）──
    static constexpr int   kHeadPredMaxAge = 15;   // 头部预测可信最大滞后帧
    static constexpr int   kHeadVetoMaxAge = 8;    // 头部硬否决仅在预测足够新鲜时启用
    // 头部空间匹配（标准化中心距，替代裸 IoU）：head_match=1-(center_dist/head_size)/falloff。
    static constexpr float kHeadMatchFalloff = 2.5f;
    static constexpr float kHeadMatchVetoMin = 0.20f;  // F6：否决距离 ≈2.0 头径

    // ── 同衣运动否决（相对空间一致性）──
    static constexpr float kMotionVetoFactor     = 2.5f;
    static constexpr float kMotionVetoMinGapDiag = 1.0f;

    // ── 头部连续性（A）：身体被家具遮挡但头部可见时维持跟踪 ──
    //   检测不到人体、但有"独立头部"（未关联任何人体）落在头部 KF 预测附近 → 判定主目标仍在
    //   （仅头可见）：真实头校正头部 KF、按学习的"头→体"几何重建身体框并输出。头也丢 → 盲 coast。
    std::vector<cv::Rect> standalone_heads_;
    std::vector<float>    standalone_head_scores_;
    // ── 人脸连续性（B）：未关联任何人体的"独立人脸"（身体被遮挡、仅露脸时的关键信号）──
    //   与 standalone_heads_ 对称；matchPersonFaces 每帧重建（xyxy 塞 Rect）。
    std::vector<cv::Rect> standalone_faces_;
    std::vector<float>    standalone_face_scores_;
    // 恢复候选保留本帧全部检测部件，包括 standalone、绑定在最终未接受人体上的
    // 部件，以及因人体几何异常未建立正式关联但本身仍有效的部件。
    std::vector<cv::Rect> recovery_heads_;
    std::vector<float>    recovery_head_scores_;
    // 与 recovery_heads_ 对齐；-1=standalone/几何异常，>=0=正式绑定的人体索引。
    std::vector<int>      recovery_head_owner_person_;
    std::vector<char>     recovery_head_owner_ambiguous_;
    // 与 matchPersonFaces 中 persons 顺序一致，供 head-only 判断是否处于人体交错。
    std::vector<cv::Rect> recovery_body_boxes_;
    std::vector<char>     recovery_body_valid_;
    std::vector<IdentityEvidence> recovery_body_identity_evidence_;
    std::vector<std::string> recovery_body_identity_reason_;
    std::vector<cv::Rect> recovery_faces_;
    std::vector<float>    recovery_face_scores_;
    std::vector<int>      recovery_face_owner_person_;
    std::vector<char>     recovery_face_owner_ambiguous_;
    bool  head_body_geom_valid_ = false;
    float hb_h_ratio_  = 7.0f;   // body_h / head_h
    float hb_w_ratio_  = 2.5f;   // body_w / head_w
    float hb_dx_ratio_ = 0.0f;   // (body_cx - head_cx) / head_w
    float last_main_bw_ = -1.f, last_main_bh_ = -1.f;   // 最近真实体框尺寸（重建钳制基准）
    static constexpr float kHeadReconSizeLo   = 0.35f;
    static constexpr float kHeadReconSizeHi   = 5.0f;
    static constexpr float kHbGeomAlpha       = 0.2f;
    // 可靠 BODY+FACE 帧学习脸→完整人体几何；FACE-only 始终用当前脸尺度重构。
    // 尚未学到目标个体比例时使用下列保守默认值；近距离大脸由画面裁剪自然得到可见部分。
    bool  face_body_geom_valid_ = false;
    float fb_h_ratio_  = 8.0f;   // body_h / face_h
    float fb_w_ratio_  = 3.2f;   // body_w / face_w
    float fb_dx_ratio_ = 0.0f;   // (body_cx - face_cx) / face_w
    static constexpr float kFbGeomAlpha = 0.15f;
    static constexpr float kHeadReacqGateRatio = 2.5f;  // 头重捕门：标准化中心距上限（×头径）
    static constexpr float kHeadReacqMinScore  = 0.5f;  // 独立头维持连续性的最低检测置信度
    static constexpr float kHeadReacqMinScoreSolo = 0.30f; // 单头、无竞争时允许较弱但连续的头检测
    static constexpr float kHeadReacqAmbigRatio = 0.5f; // 门内次近头 d1>此×d2 → 歧义拒绝
    // 头 KF 陈旧后严格门会收缩；若目标仍在运动，真实头可能稳定地落在门外。允许一个
    // 更宽但必须“无歧义 + 同一位置假设连续两帧”的恢复通道，避免直接扩大严格门导致吸错头。
    static constexpr int   kHeadReacqExtendedMinTsu = 4;
    // 扩展门按预测年龄增加不确定度：tsu=8/15 时约为 1.32/1.60×基础头门；
    // 与严格门的随龄收缩相反，但只服务两帧确认通道。
    static constexpr float kHeadReacqExtendedAgeGrow = 0.04f;
    static constexpr int   kHeadReacqExtendedConfirmFrames = 2;
    static constexpr float kHeadReacqExtendedStepRatio = 1.5f;
    static constexpr int64_t kHeadReacqExtendedMaxGapMs = 250;
    float head_reacq_pending_cx_ = -1.f, head_reacq_pending_cy_ = -1.f;
    int   head_reacq_pending_streak_ = 0;
    int64_t head_reacq_pending_ms_ = -1;
    static constexpr float kObservedPartWeight = 1.0f;  // 已接受真实头/脸观测：云台满控制增益
    // B7：头部连续性无身份证据 → 尺寸、歧义、归属和局部竞争护栏；连续真观测不再固定超时。
    static constexpr float   kHeadSizeRatioMin  = 0.55f;
    static constexpr float   kHeadSizeRatioMax  = 1.8f;
    static constexpr int64_t kHeadOnlySuspectMs = 1000;
    static constexpr float   kHeadBodyResumeMin = 0.55f;
    int64_t head_only_since_ms_ = -1;   // 头部维持连续段起始时刻；-1=未维持。真实身体命中/reset 清

    // ── 人脸连续性（B）：身体不可见、仅露独立人脸时的强身份维持 ──
    //   FaceReco 命中后建立短时物理脸连续性；后续帧仍持续按预算复核身份，但偶发
    //   Kps/角度/预算失败可由“真实脸检测 + 唯一空间连续性”桥接，避免 PTZ 脉冲启停。
    //   命中/桥接 → 由脸 + 最近真实体尺寸预测整身体框，以满控制权重输出。
    // 原始脸池独立于人体高/低分组；CLEAR 中若脸未被健康单人 BODY 的近期人脸身份解释，
    // 也会使用普通 1-slot 持续全局观察，防错误 BODY 刷新状态后永久跳过远处真脸。
    // 无人体的身份优先帧没有 Pose/ReID：Detector + Face×3 + logic 约 57ms，仍低于 10 FPS。
    // 多于三张时首轮局部优先，随后按短期物理脸假设的最近推理时间跨帧公平覆盖，
    // 避免 detector 数组重排后仍反复处理“最近三张”。每帧预算不增加。
    static constexpr int   kFaceOnlyMaxCand  = 3;      // 每帧最多对 N 个调度脸跑识别（过载降到 1）
    static constexpr float kFaceOnlyGateDiag = 3.0f;   // 独立脸到预测中心的搜索门（×最近真实体框对角线）
    // 全局找脸：旧位置只决定优先级，不再永久排除门外脸。候选按稳定空间假设跨帧覆盖，
    // 高相似度可即时重定位；普通正匹配须同一空间假设连续两帧，控制多人全画面误识风险。
    static constexpr int64_t kFaceRecoveryRefMaxAgeMs    = 900;
    static constexpr float   kFaceGlobalDirectSim        = 0.65f;
    static constexpr int     kFaceGlobalConfirmFrames    = 2;
    static constexpr int64_t kFaceGlobalConfirmMaxGapMs  = 350;
    static constexpr float   kFaceGlobalSameHypFaceDiag  = 2.5f;
    static constexpr float   kFaceRelocateBodyDiag       = 0.75f;
    // 人脸身份已由 FaceReco 确认后，允许用“本帧真实检测到的同一物理脸”桥接偶发的
    // Kps/角度/预算失败。不是复用冻结框：每帧都必须有尺寸一致、空间连续且不歧义的脸。
    // 身份最长沿用 1.8s；多人脸靠近到无法明确区分时立即停桥接，强制回 FaceReco。
    static constexpr int64_t kFaceTrackIdentityMaxAgeMs = 1800;
    static constexpr int64_t kFaceTrackMaxGapMs         = 260;
    static constexpr float   kFaceTrackGateDiag         = 1.25f;
    static constexpr float   kFaceTrackAmbigRatio       = 0.55f;
    static constexpr float   kFaceTrackSizeRatioMin     = 0.55f;
    static constexpr float   kFaceTrackSizeRatioMax     = 1.80f;
    int face_recovery_rotor_ = 0;
    int face_recovery_fail_streak_ = 0;
    // 本帧 BODY 尚未被人脸身份解释时，禁止 update() 的 CLEAR 提交清空公平扫描状态；
    // 否则普通 1-slot 会每帧重新从旧位置最近脸开始，远处主脸永久饥饿。
    bool preserve_face_search_state_ = false;
    // 本帧已提交 BODY 已有强身份，或其关联脸连续、脸体几何合理并能明确归属时，
    // 人脸连续性只维护身份/部件轨迹，不用近似重构框覆盖真实人体检测框。
    // 这是逐帧仲裁结果：update() 帧首清零，match_main_target_unified() 计算，reset() 全清。
    bool prefer_body_geometry_output_ = false;
    // 多人脸恢复调度不能依赖 detector 每帧返回的数组下标：人脸顺序一变，按下标
    // rotor 会反复处理同一批物理人脸。用短生命周期空间假设记录每张脸最近一次
    // 真正获得识别机会的帧，优先调度最久未处理者。
    struct FaceRecoveryHypothesis {
        int id = -1;
        cv::Rect box;                  // xyxy 语义
        int last_seen_frame = -1;
        int last_inferred_frame = -1000000;
    };
    std::vector<FaceRecoveryHypothesis> face_recovery_hypotheses_;
    int next_face_recovery_hyp_id_ = 1;
    static constexpr int   kFaceRecoveryHypMaxAgeFrames = 8;
    static constexpr float kFaceRecoveryHypMatchDiag    = 1.75f;
    static constexpr int   kFaceRecoveryLocalRetryFrames = 3;
    cv::Rect last_confirmed_face_box_;        // xyxy；最近身份确认/连续观测到的物理脸
    int64_t last_confirmed_face_ms_ = -1;     // 最近物理脸真实观测；身份时间另见 last_face_identity_ms_
    int last_confirmed_face_frame_ = -1;      // 输出控制只允许本帧真实脸，禁止旧脸持续拉动框
    bool  face_global_pending_ = false;
    float face_global_pending_cx_ = -1.f, face_global_pending_cy_ = -1.f;
    int   face_global_pending_streak_ = 0;
    int64_t face_global_pending_ms_ = -1;

    // B1：全画面扫描人脸命中走暂定闸（2 帧同假设确认）；暂定期扫描逐帧重试。
    static constexpr float kFaceSimMargin          = 0.08f;
    static constexpr int   kFaceSweepConfirmFrames = 2;
    bool pending_from_sweep_ = false;
    int  pending_sweep_frame_ = -1;

    // ── 主目标朝向 + 肩部几何（转身/光照期外观退化的"第三可靠性轴"）──
    float   frontalness_ = -1.f;               // EMA；-1=本帧无法评估（门控空操作）
    bool    shoulder_geom_valid_ = false;
    float   sb_dx_ratio_ = 0.0f;
    float   sb_dy_ratio_ = -0.30f;
    float   sb_w_ratio_  = 0.60f;
    int64_t orient_low_since_ms_ = -1;
    float   prev_incumbent_anchor_ = -1.f;

    // ── 长盲 coast 后重捕的身份复核（C-identity）──
    // [N1] "真正盲跟时长"改墙钟（blind_ms = now − last_real_obs_ms_）。
    int reacq_defer_count_ = 0;
    float pending_cx_ = -1.f, pending_cy_ = -1.f;
    bool  pending_active_ = false;
    // [N4] 暂定假设对应的检测（source/index）：暂定期该检测从二级轨迹收集中排除。
    int   pending_src_ = -1, pending_idx_ = -1;
    // 仅 pending_body_hyp_id_ >= 0 表示 long-blind BODY provisional。
    // generic UNCERTAIN / face sweep / 纯调度 pending 不得获得 continuation 优先权。
    int   pending_body_hyp_id_ = -1;
    float pending_body_x1_ = -1.f, pending_body_y1_ = -1.f;
    float pending_body_x2_ = -1.f, pending_body_y2_ = -1.f;
    float pending_body_reid_ = -1.f, pending_body_anchor_ = -1.f;
    int   pending_body_start_frame_ = -1;
    int64_t pending_body_start_ms_ = -1;
    static constexpr float kProvisionalScoreEpsilon = 1e-4f;
    static constexpr float   kProvisionalPosTolFactor = 1.5f;
    static constexpr int64_t kReacqProbationMs   = 500;  // [N1] blind_ms ≥ 此 → 启用复核（≈12f@25fps）
    static constexpr float   kReacqAnchorConfident = 0.58f;
    // issue 2「从前面经过」：连续在场+全身可见+高 anchor → 信任跟随 KF。[N7] 帧数 AND 墙钟双闸。
    static constexpr int     kFrontFollowMaxTsu  = 5;
    static constexpr int64_t kFrontFollowMaxMs   = 600;
    static constexpr int     kReacqMaxDefer      = 1;    // K：非强身份高惊奇重捕同假设确认帧数
    // ── 强 ReID 直证：新外观模型可信时，允许强外观证据即时提交/跟随，减少穿人卡顿。
    //   OccludedDataset：同人 0.4186±0.1397、异人 0.0488±0.0841；危险尾部
    //   同人 P5=0.1918、异人 P95=0.1988，单一低阈值无法兼顾召回与误认。
    //   全身恢复保持严格档 0.40；半身强直证取 0.36，并继续要求可靠 anchor 与 0.10 领先差。
    //   低相似度仍可作为融合软分，但绝不能绕过遮挡/歧义/空间安全闸。
    static constexpr float   kReidDirectConfirmFull    = 0.40f;
    static constexpr float   kReidDirectConfirmPartial = 0.36f;
    static constexpr float   kReidDirectAnchorFloor = 0.36f;
    static constexpr float   kReidDirectMargin      = 0.10f;
    static constexpr float   kAnchorDirectConfirm   = 0.68f;

    // ── 在位者迟滞（G1：防同衣/近似目标逐帧震荡）── [N7] 帧数 AND 墙钟双闸。
    static constexpr float   kIncumbentHysteresis = 0.07f;
    static constexpr float   kIncumbentMinIou     = 0.30f;
    static constexpr int     kIncumbentMaxTsu     = 3;
    static constexpr int64_t kIncumbentMaxMs      = 250;

    // ── 人脸锁硬保持（FACE_HOLD）──
    static constexpr float kFaceHoldMinIou      = 0.20f;
    static constexpr float kFaceHoldBonus       = 1.0f;
    static constexpr float kFaceHoldAnchorFloor = 0.05f;

    // ── 分离期消歧（S1：遮挡后重现防"干净遮挡者夺锁"）──
    static constexpr float kSepTrajGateDiag = 2.0f;
    static constexpr float kSepTrajMargin   = 0.5f;
    static constexpr float kSepAppDominant  = 0.15f;

    // ── 主匹配歧义门限 ──
    // ReID 阈值按当前模型的实测分布校准；anchor 是多样本画廊最大值，保留独立门限。
    // 两人都落在“可能同人”的区域且差距不足 0.08，按歧义处理而非拿低分外观抢锁。
    static constexpr float kReidAmbiguousMin   = 0.20f;
    static constexpr float kReidAmbiguousGap   = 0.08f;
    static constexpr float kAnchorAmbiguousMin = 0.65f;
    static constexpr float kAnchorAmbiguousGap = 0.04f;
    static constexpr float kAmbiguousGapDanger = 0.04f;
    static constexpr float kAmbiguousGapClear  = 0.02f;
    static constexpr float kFaceLockVetoBoost  = 0.08f;

    // ── [N1] 短盲跳变身份复核（teleport gate）── 预算帧率无关（对角线 × 盲秒数[有上限]）+ 自运动松弛。
    // [N1-fix] 基础预算也帧率无关化：单帧合法位移 ∝ frame_dt，故基础项 = 每秒率×frame_dt。
    //   固定基础项在高帧率下过松（短间隔内允许大跳变）、低帧率下过紧。20/s×0.04s ≈ 旧 0.8diag@25fps。
    static constexpr float   kTeleportBaseDiagPerSec = 20.0f;  // 单帧基础预算率（×对角线/秒；×frame_dt）
    static constexpr float   kTeleportRateDiagPerSec = 12.5f;  // 每秒追加（≈旧 0.5diag/f@25fps）
    static constexpr int64_t kTeleportElapsedCapMs   = 400;    // 盲时长计入预算上限（之后 C-identity 接管）

    // ── [N2] spatial_cont 中心距臂 ──
    // 无 GMC 时云台移动帧 KF 预测框滞后 → IoU 臂失效；若又无关联头 → 危险期严格档误杀真目标。
    // 第三臂：候选中心到统一参考点（lead 优先，含自运动前馈）距离 ≤ 对角线×(基础+速率×dt)+自运动松弛。
    static constexpr float kSpatialContCenterDiag     = 1.0f;
    static constexpr float kSpatialContRateDiagPerSec = 1.5f;

    // ── 背向/侧向/远距宽容（防"转身/走远即丢"）──
    static constexpr float kAnchorVetoRelaxed = 0.28f;  // 宽容档（CLEAR 孤立转身/走远）
    static constexpr float kAnchorVetoDanger  = 0.22f;  // 危险/半身档：保留真目标退化余量，不接纳近零外观
    static constexpr float kAnchorVetoDangerCrowded = 0.32f; // 有他人竞争时宁可短暂停住，也不以低外观夺锁
    static constexpr float kVetoRelaxIou      = 0.20f;  // "空间连续"最小 IoU（PTZ/穿人时 KF 滞后，适度放宽）
    // §1 防误检夺锁：放宽仅在"主目标连续在场"时生效。[N7] 帧数 AND 墙钟双闸。
    static constexpr int     kRelaxMaxMissFrames = 6;
    static constexpr int64_t kRelaxMaxMissMs     = 350;
    static constexpr float kEmbAdaptAnchorFloor = 0.35f;
    // "新视角"样本入廊（时空连续性认证；解决画廊学不到背面/侧面死锁）
    static constexpr int   kViewAddMinStreak = 15;
    static constexpr float kViewAddQuality   = 0.55f;

    // ── 多样本锚点画廊填充（防注册污染；anchor_sim=max(画廊) 会放宽身份门 → 添加须保守）──
    static constexpr float kAnchorAddAnchorMin = 0.70f;  // 非人脸入廊：anchor 置信下限
    static constexpr int   kAnchorAddPeriod    = 15;     // 非人脸入廊：最小周期（帧）

    // ── 非主目标轻量运动轨迹（纯 IoU 关联，不存 emb）── [N14] 颜色门常量已删。
    static constexpr int   kSecTrkMax        = 8;
    static constexpr float kSecTrkIou        = 0.30f;
    static constexpr float kSecTrkNewMainIou = 0.50f;   // 新建轨迹与主框 IoU 超此不建（防影子）
    static constexpr int   kCoexistConfirm   = 10;      // full：可单臂/位置否决
    static constexpr int   kCoexistProvisional = 3;     // P1：暂定他人（否决须位置 AND 外观双证）
    static constexpr float kCoexistVetoIou   = 0.50f;
    static constexpr int   kCoexistVetoMaxTsu = 2;
    // B3：隔离轨迹（出生即压主框）转正条件
    static constexpr float kQuarantineClearIou    = 0.30f;
    static constexpr int   kQuarantineClearFrames = 5;

    // ── 非主轨迹外观特征 + 外观排除（"已知他人"负证据）── [N14] 颜色相关常量已删。
    static constexpr int   kSecReidBudgetPerFrame = 1;
    static constexpr int   kSecEmbRefreshFrames   = 30;
    static constexpr float kSecExclSimMin   = 0.65f;
    static constexpr float kSecExclMargin   = 0.10f;    // 同衣护栏：明显更像他人才排除
    static constexpr float kSecPollutionSim = 0.55f;    // 与主 anchor ≥ 此 → 疑主目标误裂，禁作排除源
    // 二级轨迹纯 IoU 关联可能在交错时换到主目标身上。若当前候选已满足强主身份直证，
    // 且它比该二级轨迹保存的身份至少更接近主 anchor 此余量，则判定为“所有权反转”，
    // 撤销该二级轨迹的位置否决并隔离重建。只复用已有 embedding，不增加 ReID 调度。
    static constexpr float kSecOwnershipMainMargin = 0.15f;

    // ── 朝向门 + 肩部连续性（防转身/光照 lock-loss）──
    static constexpr float   kFrontalLowThresh      = 0.55f;
    static constexpr float   kFrontalnessAlpha      = 0.4f;
    static constexpr float   kOrientMinTorsoPx      = 60.f;
    static constexpr float   kOrientReidMul         = 0.55f;
    static constexpr float   kOrientShapeMul        = 0.5f;
    static constexpr float   kShoulderContBudget    = 0.35f;
    static constexpr float   kShoulderContFalloff   = 2.0f;
    static constexpr float   kShoulderContIncumbent = 0.5f;
    static constexpr float   kRelEngage             = 0.99f;
    static constexpr float   kAnchorVetoRelaxTurned = 0.20f;
    static constexpr int64_t kOrientHoldMaxMs       = 1500;
    static constexpr float   kLightingPrevAnchorMin = 0.5f;
    static constexpr float   kAnchorDropSuspect     = 0.25f;
    static constexpr float   kLightingRel           = 0.3f;

    // ── M1：主目标静止时否决"压在移动他人轨迹上"的候选（GMC 关闭时不启用）──
    // [N14] kStationaryPxSec 已删（单阈值化后无引用）；仅保留 mv_move_thr 用的 kCandMovingPxSec。
    static constexpr float kCandMovingPxSec    = 75.f;   // 候选/主目标速度 > 此(px/s) → 视为移动
    static constexpr float kMotionVetoAssocIou = 0.30f;
    static constexpr float kMotionDirCosVeto   = 0.30f;  // 方向余弦 < 此（夹角>72°）→ 相悖否决

    // ── M2：可见度退化的不确定匹配 → 须人脸或高空间连续 ──
    static constexpr float kUncertainVisIou = 0.30f;

    // ── 疑惑期全画面人脸扫描（加速错锁找回）──
    static constexpr int   kFaceSweepPeriod      = 15;
    static constexpr float kFaceSweepDoubtAnchor = 0.70f;
    static constexpr int   kFaceSweepMaxFaces    = 2;
    static constexpr float kFaceSweepMinFacePx   = 14.f;
    int face_sweep_rotor_ = 0;

    // ── 部分遮挡框补全（防 PTZ 因半身框中心上移而"抬头"）──
    float main_h_hold_ = 0.f;
    static constexpr float kBoxCompleteTrig    = 0.85f;
    static constexpr float kBoxCompleteAlpha   = 0.15f;
    static constexpr float kBoxCompleteEvidIou = 0.10f;
    static constexpr float   kBoxCompleteMaxAspect = 3.4f;
    static constexpr int64_t kBoxCompleteHoldMaxMs = 5000;
    int64_t main_h_hold_ms_ = 0;

    // ── 人脸模板质量锁 ──
    float face_template_quality_ = 0.f;
    static constexpr float kFaceTemplateGoodEnough    = 0.80f;
    static constexpr float kFaceTemplateUpgradeMargin = 0.05f;

    // ── [N7] 轨迹寿命墙钟（共存否决轨迹的存活语义在 14× dt 波动下漂移严重）──
    static constexpr int64_t kTrackMaxAgeMs = 900;   // 非主轨迹无观测超此 → 清理（≈max_age 30f@33fps）

    int img_h;
    int img_w;
    int main_id = -1;

public:
    LightTracker(const LightTrackerConfig& config = LightTrackerConfig());
    ~LightTracker();

    int init();
    void set_maxAge(int maxAge) { max_age = maxAge; };

    // 返回 {跟踪结果, 人体检测数}。正常帧的人体检测数为 dets_one + dets_second；
    // 用户指定主目标的注册帧没有检测分组，返回指定目标数 1。
    std::pair<cv::Mat, int> update(const cv::Mat& img, const cv::Rect& mainBox, const float& z);

    void reset();

    bool get_reset_flag() { return reset_flag; };
    void set_reset_flag(bool flag) { reset_flag = flag; };
    int get_main_id() { return main_id; };

    /// 当前云台控制权重：1.0=真实身体或严格接受的头/脸观测；(0,1)=短时预测
    /// 的衰减可信度；0.0=本帧无主框。
    float get_coast_weight() const { return coast_weight_; }

    /// [N11] 云台推荐瞄准点（头锚定，消除半身/家具遮挡的 tilt 抖动）。
    ///   App 应把 (x,y) 驱动到画面中心；aim_valid=false 时（本帧无输出/信息不足）
    ///   回退用 get_coast_weight 决定是否保持。weight 即 coast_weight（控制增益乘子）。
    AimPoint get_aim_point() const {
        AimPoint a; a.x = aim_x_; a.y = aim_y_; a.valid = aim_valid_; a.weight = coast_weight_;
        return a;
    }

    /// [N1] 距上次真实观测的墙钟毫秒（诊断/上层丢失恢复策略用）。<0 → 尚无观测。
    int64_t get_blind_ms() const {
        return (last_real_obs_ms_ < 0) ? -1 : (now_ms_const() - last_real_obs_ms_);
    }
    int64_t get_body_blind_ms() const {
        return (last_body_observation_ms_ < 0) ? -1
             : (now_ms_const() - last_body_observation_ms_);
    }

    /// [N10] 自运动前馈开关（现场若发现前馈方向估计不稳，可一键回退到纯几何门控）。
    void set_ego_enabled(bool en) { ego_enabled_ = en; }

    // 外部 PTZ 的只读 blind 状态。SLIDING/STOPPED 只在 PTZ 已实际移动后上报；
    // 坐标系失效标志直到可靠 BODY 重捕后才清除。
    void set_ptz_blind_phase(int phase);

    /// 设置本帧 Y(亮度)平面，供 GMC 零拷贝硬件加速使用（每帧在 update 前调用）。
    void set_frame_yplane(unsigned long long y_phys, unsigned long long y_virt,
                          int w, int h, int y_stride) {
        bool ok = (y_phys != 0ULL) && (y_virt != 0ULL) && w > 0 && h > 0 && y_stride >= w;
        cur_y_phys_   = ok ? y_phys   : 0ULL;
        cur_y_virt_   = ok ? y_virt   : 0ULL;
        cur_y_w_      = ok ? w        : 0;
        cur_y_h_      = ok ? h        : 0;
        cur_y_stride_ = ok ? y_stride : 0;
        cur_y_valid_  = ok;
    }

private:

    struct DetectionGroups {
        cv::Mat dets_one;        // [n,5]
        std::vector<cv::Mat> dets_one_face;   // 每个人体对应的人脸框集合
        cv::Mat dets_second;     // [n,5]
        std::vector<cv::Mat> dets_second_face;
        // 头部框（与 dets_one / dets_second 行对齐）：无头则全 0
        cv::Mat dets_one_head;    // [n,5]
        cv::Mat dets_second_head; // [n,5]
    };

    bool isFaceFullyInsidePerson(const cv::Rect& face, const cv::Rect& person);

    std::vector<PersonWithFace> matchPersonFaces(const std::vector<ObjDetInfo>& detect_list);

    void add_other_det(cv::Mat& result, cv::Mat dets_one, cv::Mat dets_second);

    cv::Mat build_main_row(const cv::Mat& box4);

    // 将重构框裁到真实画面；source_tag 仅用于 trace。
    cv::Mat clip_reconstructed_body_to_frame(const cv::Mat& box4,
                                             const char* source_tag);
    // 返回框专用 cx/cy/w/h 整体稳定。调用方先把同一个画面内可见框写入
    // KF/face_lock/lead，本函数只对交给云台的最终整框做防抖。
    cv::Mat stabilize_returned_box(const cv::Mat& raw_return_box,
                                   OutputSource source, bool force_snap);
    void note_returned_box(const cv::Mat& box4);

    // BODY/FACE/HEAD 均无真实观测时，尝试从最近真实 Motion History 启动冻结预测。
    // 成功仅写 out_box、frame_output_source_ 和 coast_weight_；不修改身份状态。
    bool try_short_prediction(cv::Mat& out_box, const char* context);
    void clear_short_prediction(const char* reason);
    void exhaust_short_prediction(const char* reason);
    void record_motion_observation(const cv::Mat& body_box,
                                   MotionObservationSource source,
                                   int64_t timestamp_ms);
    void prune_motion_history(int64_t now);
    MotionEstimate estimate_motion(int64_t now) const;
    void clear_motion_history();
    static int motion_source_priority(MotionObservationSource source);
    static const char* motion_source_name(MotionObservationSource source);
    static const char* motion_quality_name(MotionQuality quality);
    static const char* prediction_mode_name(PredictionMode mode);
    static int box_clip_mask(const cv::Mat& body_box, int width, int height);
    void trace_prediction_coverage(bool has_main_target);
    const SecondaryFrameObservation* secondary_frame_observation(
        int tracker_id) const;
    void record_relative_motion_observation(
        int tracker_id, const cv::Mat& main_box,
        const cv::Mat& secondary_box);
    bool estimate_relative_emergence_direction(
        int tracker_id, float& dir_x, float& dir_y,
        int* sample_count = nullptr,
        float* direction_consistency = nullptr);
    void prune_relative_motion_history(int64_t now);
    bool recovery_search_center(float& cx, float& cy) const;

    static const char* identity_evidence_name(IdentityEvidence evidence);
    static const char* person_identity_risk_reason_name(uint32_t reasons);
    void note_person_identity_ambiguity(uint32_t reasons, int close_count,
                                        bool direct_competition);
    bool person_identity_ambiguity_active(int64_t now) const;
    void expire_person_identity_ambiguity(int64_t now);
    void mark_body_identity_evidence(const cv::Mat& body_box,
                                     IdentityEvidence evidence,
                                     const char* reason);
    IdentityEvidence body_identity_evidence_for_box(
        const cv::Mat& body_box, std::string* reason = nullptr,
        int* owner_person = nullptr) const;
    IdentityEvidence body_identity_evidence_for_owner(
        int owner_person, std::string* reason = nullptr) const;
    IdentityEvidence face_identity_evidence_for_box(
        const cv::Rect& face_box, std::string* reason = nullptr) const;
    void trace_continuity_gate(const char* type, const char* action,
                               bool scene_risk, IdentityEvidence part_evidence,
                               IdentityEvidence owner_evidence, int owner,
                               const char* reason);
    bool body_provisional_geometry(const cv::Mat& body_box,
                                   float& center_distance,
                                   float& box_iou,
                                   float& size_ratio) const;
    bool body_provisional_scores_stable(float reid, float anchor) const;
    void trace_body_provisional(const char* action, int hyp, int prev_hyp,
                                int source, int index, float reid, float anchor,
                                float prev_reid, float prev_anchor,
                                int streak, int need, float center_distance,
                                float box_iou, float size_ratio,
                                const char* reason, const char* gate,
                                const char* override_reason = "none");
    void clear_body_provisional(const char* reason, const char* gate,
                                int source = -1, int index = -1,
                                float reid = -999.f, float anchor = -999.f,
                                float center_distance = -1.f,
                                float box_iou = -1.f,
                                float size_ratio = -1.f,
                                int current_hyp = -1);

    // 记录引导中心（丢失期重捕搜索门用）。box 为 [1,4] xyxy，空则不更新。
    void update_lead_center(const cv::Mat& box);

    // ── [N10] 自运动前馈辅助 ──
    // 每帧开头调用：把上次输出误差按 β·dt 闭合，得到本帧预测表观位移 (ego_sx_,ego_sy_)，
    // 并平移 lead/smooth/emergence 三个软参考点（不碰 KF）。无有效误差/超时 → 位移置 0。
    void apply_ego_feedforward();
    // 每帧在确定主框输出后调用：更新未闭合误差向量 + 在线估计 β（连续命中帧）。
    // out_valid=false（本帧无输出）→ 误差失活（云台将停）。
    void note_output_for_ego(bool out_valid, float out_cx, float out_cy, bool matched_real);
    // [N11] 由本帧主框 + 头部预测 + 高度保持值算头锚定瞄准点，写 aim_*。
    void compute_aim_point(const cv::Mat& main_box4);

    // ── 头部连续性（A）辅助 ──
    void learn_head_body_geom(const cv::Mat& body_box, const cv::Mat& head_box);
    void learn_face_body_geom(const cv::Mat& body_box, const cv::Mat& face_box);
    cv::Mat reconstruct_body_from_head(const cv::Mat& head_box) const;
    int find_recovery_head_near(const cv::Mat& head_pred, float gate_scale = 1.f,
                                float* out_d1 = nullptr, float* out_d2 = nullptr,
                                float* out_gate = nullptr) const;
    bool try_head_continuity(cv::Mat& out_box);

    // ── 人脸连续性（B）辅助 ──
    // 由独立人脸框按当前脸尺度预测完整身体框；退化框 → 返回空。
    cv::Mat reconstruct_body_from_face(const cv::Rect& face_box) const;
    // 判断人体检测是否以合理的“上部、居中”几何拥有该脸；同时参考脸重构框
    // 的水平中心，避免宽大/重叠的错误人体仅凭包含关系截走已识别的主目标脸。
    bool face_body_geometry_consistent(const cv::Rect& face_box,
                                       const cv::Mat& body_box,
                                       float* out_cost = nullptr) const;
    // 身体不可见、仅露独立人脸时：识别命中即重建体框写 out_box、置人脸锁、返回 true。
    bool try_face_only_continuity(const cv::Mat& img, cv::Mat& out_box);
    // FaceReco 已确认后的短时物理脸连续性：只复用本帧真实脸检测，不新增模型调用。
    // measured_body_box 非空时，仅用连续脸维护身份/头轨迹，几何输出保留真实 BODY。
    bool try_confirmed_face_track_continuity(
        cv::Mat& out_box, const cv::Mat& measured_body_box = cv::Mat(),
        bool* used_measured_body = nullptr);
    // 人脸确认时，若同帧独立头包住该脸，则同步校正头部 KF，保证脸消失后可无缝降级到头轨迹。
    void sync_head_track_from_confirmed_face(const cv::Rect& face_box,
                                             bool relocate = false);
    // 首次注册帧漏脸时的安全补偿：仅“已被单头轨迹确认 + 单张脸在该头内”才允许补建模板。
    void try_register_face_from_confirmed_head(const cv::Mat& img);
    float head_match_score(const cv::Mat& head_pred, const cv::Mat& head_cand) const;

    // ── 主目标朝向 + 肩部连续性 ──
    void update_orientation_state(const PoseResult* main_pose, const cv::Mat& body_box);
    float shoulder_cont_score(const cv::Point2f& cand_mid, const cv::Mat& pred_body_box) const;
    static bool extract_shoulder_geom(const PoseKeypoint* kps, float conf_thresh,
                                      cv::Point2f& mid, float& width);

    bool should_update_embedding(const MainMatchResult& match_result, int close_det_count);

    // ── 非主目标轻量轨迹维护 ──
    void update_secondary_tracks(const DetectionGroups& det_groups,
                                 const MainMatchResult& best,
                                 const cv::Mat& img);
    void update_secondary_features(const DetectionGroups& det_groups, const cv::Mat& img);

    // [N14] compute_color_hist / color_hist_sim 已删（颜色直方图整体弃用）。

    int64_t now_ms();
    // const 版（供 get_blind_ms 等 const 成员用；实现同 now_ms）
    int64_t now_ms_const() const;

    DetectionGroups extract_detections(const std::vector<PersonWithFace>& output_results);

    TrackerInfo get_predicted_tracks();

    std::pair<cv::Mat, int> setMainTarget(const cv::Mat& img, const cv::Rect& mainBox);

    // [N6] 由 5 点关键点评估人脸质量（正脸程度+尺寸+完整性）∈[0,1]；退化返回 -1。
    // 初始注册与延迟注册共用（原逻辑内联在 try_deferred_face_register，抽出复用）。
    float evaluate_face_quality(const std::vector<float>& kps_10,
                                const cv::Rect& face_box_xyxy,
                                int crop_w, int crop_h) const;

    Verification_Result face_recognition_verification(
        const cv::Mat& near_box, int box_n,
        const DetectionGroups& det_groups, int main_idx, cv::Mat img,
        cv::Mat matched_one, cv::Mat matched_second,
        const cv::Point2f* standalone_ref = nullptr,
        float standalone_gate = 0.f,
        int standalone_budget = 0,
        const cv::Point2f* standalone_alt_ref = nullptr,
        bool standalone_identity_only = false);

    std::pair<float, std::string> face_recognition_inference(const cv::Mat& face_boxes,  cv::Mat img);
    bool take_face_model_slot();

    MainNonMainSplit split_main_nomain(const cv::Mat& trks);

    void update_quality_monitor(const MainMatchResult& result, int close_det_count);
    void update_smooth_center(const cv::Mat& trk_box);
    void update_body_reid_search_anchor(const cv::Mat& box);
    void update_ptz_blind_anchor(const cv::Mat& box, int64_t timestamp_ms);
    void finish_ptz_blind_reacquisition(int main_trk_idx, const cv::Mat& body_box);

    ProximityInfo collect_nearby_dets(
        const cv::Mat& main_trk,
        const cv::Mat& dets_one,
        const cv::Mat& dets_second);

    void update_trackers_unified(
        const cv::Mat& matches_one,
        const cv::Mat& matches_second,
        const std::vector<int>& unmatched_trks,
        const cv::Mat& dets_one,
        const cv::Mat& dets_second,
        const std::vector<int>& picked_idx,
        const MainMatchResult& match_result,
        const cv::Mat& img);

    cv::Mat assign_cascade(const DetectionGroups& det_groups, const cv::Mat& img);

    cv::Mat match_main_target_unified(
        const TrackerInfo& info,
        const DetectionGroups& det_groups,
        const ProximityInfo& proximity,
        const std::vector<int>& main_indices,
        const cv::Mat& img);

    Face_Match face_recognition_match(const cv::Mat& near_box, int box_n,  const DetectionGroups& det_groups, cv::Mat matched_one, cv::Mat matched_second, int main_idx, cv::Mat img);

    cv::Mat generate_final_results();

    std::vector<float> get_kps10(std::vector<FaceKeypoint> kps_106);

    void cleanup_expired_trackers();

    void try_deferred_face_register(const cv::Mat& img,
                                    const DetectionGroups& det_groups,
                                    const MainMatchResult& match_result,
                                    int close_det_count);

    // ── Pose ──
    PoseCacheEntry* request_pose(const cv::Mat& img, const cv::Mat& body_box,
                                 int source, int index, PoseReason reason);
    PoseCacheEntry* find_cached_pose(int source, int index);
    void commit_pose(const PoseResult& pose, const cv::Mat& body_box,
                     const cv::Mat& img, const char* reason);
    PoseScoreDetail compute_candidate_pose_detail(
        const PoseResult& candidate_pose, const cv::Mat& candidate_bbox,
        const cv::Mat& predicted_box);

    // ── 主目标身体可见度评估 ──
    VisibilityState assess_visibility(const cv::Mat& main_box,
                                      const PoseResult* pose,
                                      int frame_w, int frame_h);

    // ── GMC 相机运动补偿（当前关闭）──
    cv::Mat estimate_camera_motion(const cv::Mat& img, const cv::Mat& trks);

#ifdef USE_HISI_IVE
    IveGmc ive_gmc_;
#endif

    void warp_box_inplace(cv::Mat& box4, const cv::Mat& M);
    cv::Point2f warp_point(float x, float y, const cv::Mat& M);
};

#endif
