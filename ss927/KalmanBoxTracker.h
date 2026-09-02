
#pragma once
#ifndef KALMANBOXTRACKER_H
#define KALMANBOXTRACKER_H

#include "KalmanFilter.h"
#include "PoseEstimator.h"

#include <iostream>
#include <memory>
#include <optional>
#include <map>
#include <opencv2/opencv.hpp>
#include <deque>
#include <string>
#include <vector>

using namespace VISION_ENGINE;



class KalmanBoxTracker {
public:



    /*method*/
    KalmanBoxTracker(cv::Mat bbox_, int delta_t_ = 3, cv::Mat emb = cv::Mat(), bool is_main = false);

    // Disable copy (prevent double-free)
    KalmanBoxTracker(const KalmanBoxTracker&) = delete;
    KalmanBoxTracker& operator=(const KalmanBoxTracker&) = delete;

    // Enable move semantics
    KalmanBoxTracker(KalmanBoxTracker&&) = default;
    KalmanBoxTracker& operator=(KalmanBoxTracker&&) = default;

    void update(std::optional<cv::Mat> bbox_);
    // 头/脸重建出的身体框只作为弱几何观测校正 KF 状态：不重置
    // time_since_update，不写 last_observation/observations，也不增加 hits。
    // 因而真实人体观测的新鲜度语义保持不变，同时遮挡期间位置/速度不会漂走。
    void correct_body_from_part(const cv::Mat& bbox_, bool from_face,
                                bool relocate_center = false);
    int get_id() { return id; };
    cv::Mat get_emb() { return emb; };

    void reset_mainEmb() { emb = cv::Mat(); };


    // void update_emb(const cv::Mat& emb_new,
    //                                float quality_score);

//     cv::Mat get_reliable_emb() const {
//     // 当前 emb 质量过差时，回退到 anchor
//     if (!anchor_emb.empty() && current_visibility < 0.45f) {
//         return anchor_emb;
//     }
//     return emb;
// }

    void update_emb(const cv::Mat& new_emb, float alpha);

    // ── 多样本锚点画廊（替代单一 anchor_emb_）──────────────────────
    // anchor_sim = 对画廊所有样本取最大余弦。画廊按"质量 + 多样性"维护，
    // 使一次糟糕的注册（半身/遮挡）能被后续高质量样本逐步替换（自愈）。
    // 注意：max 取相似度会"放宽"身份门（更易通过），故仅在身份证据最强
    // （人脸确认 / 高 anchor + 全身可见 + 无贴近人）时才向画廊添加样本。
    void    set_anchor_emb(const cv::Mat& e);              // 注册：清空并以基线质量加入首样本
    cv::Mat get_anchor_emb() const;                        // 返回最高质量样本（danger 回退 / alert 回滚 用）
    void    add_anchor_sample(const cv::Mat& e, float quality, float vis = 1.f);  // vis=采样时可见比例（可见度分带用）
    float   anchor_sim_max(const cv::Mat& feat) const;     // 对画廊取最大余弦；空画廊返回 -1
    // Q1：仅对"与当前可见度同带"的样本取最大余弦（半身查询比半身参考，避免与全身参考误低）；
    //   本带尚无样本 → 回退 anchor_sim_max（= 原行为，无回归）。
    float   anchor_sim_vis(const cv::Mat& feat, float cur_vis) const;
    bool    has_anchor() const { return !anchor_gallery_.empty(); }
    int     get_anchor_gallery_size() const { return (int)anchor_gallery_.size(); }
    void set_confirmed_emb(const cv::Mat& e) { confirmed_emb_ = e.clone(); }
    cv::Mat get_confirmed_emb() const { return confirmed_emb_; }

    cv::Mat get_last_observation() { return last_observation; };
    cv::Mat get_velocity() { return velocity; };


    int get_time_since_update() { return time_since_update; };
    int get_time_since_update_emb() { return time_since_update_emb; };
    int get_hit_streak() { return hit_streak; };
    bool get_is_main() { return is_main; };

    std::map<int, cv::Mat> get_observations() { return observations; };

    float get_speed() { return speed; };

    // KF 内部速度幅值（px / 标称帧；标称帧长为 40ms，已含 GMC 状态级补偿）。
    // 注意 get_speed() 返回的是 1-IoU（OC-SORT 方向置信），非速度；
    // 本函数取 KF 状态 [.,.,.,.,vx,vy] 的模长，是真正的运动量度量。
    float get_kf_speed() const;
    // KF 速度向量 (vx,vy)，px / 标称帧，已含 GMC 补偿。用于运动方向一致性判别。
    cv::Vec2f get_kf_velocity() const;
    // 物理速度接口（px/s），供帧率无关的运动阈值使用。
    float get_kf_speed_per_sec() const;
    cv::Vec2f get_kf_velocity_per_sec() const;

    // 用真实处理间隔更新下一次预测。速度状态仍保留为 40ms 标称帧单位，
    // 因此不破坏既有方向/历史语义。
    void set_frame_interval(float dt_sec);
    // 主目标长时间没有任何真实身体/头/脸观测时冻结速度，仍推进 age/tsu，
    // 并限制协方差；下一次强观测会直接重建状态，身份模板不受影响。
    void set_long_coast(bool enabled) { long_coast_ = enabled; }

    // 数值保护事件由 LightTracker 消费并写入 match trace。
    // 字符串以 "NAN " 或 "RESET " 开头，避免生产构建依赖 stdout。
    std::vector<std::string> consume_numerical_events();

    cv::Mat predict();  // Returns a (1,4) row vector
    cv::Mat get_state(); // Returns state vector x (1,4)


    static void set_count(int val){ count = val;};

public:
    /*variable*/
    static int count;
    cv::Mat bbox;               // [4,1] 列向量，float类型
    std::unique_ptr<KalmanFilterNew> kf; // kalman predictor
    int time_since_update;
    int id;
    int hits = 0;
    int hit_streak = 0;
    int age = 0;
    int time_since_update_emb = 0;
    float speed = 0;
    bool is_main = false;
    int max_observations = 30;
    cv::Mat emb = cv::Mat();
    // ── 共存帧计数（仅非主轨迹使用）──
    // 在"主目标本帧真实匹配"且本轨迹同帧匹配到自己的检测时 +1。
    // 计数足够高 = 此人曾与主目标同时可见 → 物理上不可能是主目标本人
    // → 主目标匹配时对该轨迹覆盖的候选实施"共存排除"否决（防同衣夺锁）。
    int coexist_with_main = 0;
    // ── 隔离标记（B3，仅非主轨迹）──
    // 出生时与主框强重叠的轨迹（可能是主目标匹配被门控延迟时其自身检测生成的
    // "影子"，也可能是正面逼近的遮挡者）：允许创建（遮挡者识别/anti_occ 需要它
    // 存在），但剥夺一切否决权（不积累 coexist、不作共存/外观排除源、不输出、
    // 不占 ReID 预算），直到连续 kQuarantineClearFrames 帧用"明显离开主框"的
    // 自有检测自证是独立他人（影子做不到：它的"自有检测"就是主目标的检测）。
    bool quarantined_ = false;
    int  quarantine_clear_streak_ = 0;
    struct AnchorSample { cv::Mat emb; float quality = 0.f; float vis = 1.f; };  // vis=采样时可见比例[0,1]（Q1 可见度分带）
    std::vector<AnchorSample> anchor_gallery_;   // 多样本锚点画廊（容量/阈值见 .cpp）
    cv::Mat confirmed_emb_;

    // ── 非主轨迹外观特征（"已知他人"排除用；主轨迹亦可持有但不依赖）──
    // 颜色直方图：上身 HSV，纯 CPU（无 NPU），随观测 EMA 融合，抗光照抖动。
    // emb_update_frame_：最近一次 ReID 嵌入刷新的帧号，供轮询调度挑最陈旧者刷新。
    cv::Mat color_hist_;
    int     emb_update_frame_ = -1;   // -1 = 从未计算过嵌入（轮询时优先注册）
    float   last_det_score_ = 0.f;    // 最近关联检测的置信度（注册优先级：高分先注册）
    int64_t last_update_ms_ = -1;     // 最近真实观测时刻(ms)，由 LightTracker 盖戳；
                                      // 共存排除"新鲜度"的墙钟上限用（B6，-1=尚无观测）

    void    set_emb_update_frame(int f) { emb_update_frame_ = f; }
    int     get_emb_update_frame() const { return emb_update_frame_; }
    const cv::Mat& get_color_hist() const { return color_hist_; }
    bool    has_color_hist() const { return !color_hist_.empty(); }
    // 颜色直方图 EMA 融合（首帧直接赋值）。两个直方图 bins/type 由 compute_color_hist 统一。
    void    update_color_hist(const cv::Mat& h, float alpha) {
        if (h.empty()) return;
        if (color_hist_.empty() || color_hist_.size() != h.size()
                                || color_hist_.type() != h.type()) {
            color_hist_ = h.clone();
            return;
        }
        cv::addWeighted(color_hist_, 1.f - alpha, h, alpha, 0.0, color_hist_);
    }

    // ── Pose 数据 ──
    PoseKeypoint last_keypoints_[NUM_KEYPOINTS];       // 最近帧的关键点
    BodyProportionDescriptor body_proportions_;         // 骨骼比例描述符
    BodyProportionDescriptor anchor_body_proportions_;  // 注册时的锚点骨骼比例
    bool has_pose_ = false;

    void set_keypoints(const PoseKeypoint kps[NUM_KEYPOINTS]) {
        memcpy(last_keypoints_, kps, sizeof(PoseKeypoint) * NUM_KEYPOINTS);
        has_pose_ = true;
    }
    const PoseKeypoint* get_keypoints() const { return last_keypoints_; }
    bool get_has_pose() const { return has_pose_; }

    void set_body_proportions(const BodyProportionDescriptor& bp) { body_proportions_ = bp; }
    const BodyProportionDescriptor& get_body_proportions() const { return body_proportions_; }

    void set_anchor_body_proportions(const BodyProportionDescriptor& bp) { anchor_body_proportions_ = bp; }
    const BodyProportionDescriptor& get_anchor_body_proportions() const { return anchor_body_proportions_; }

    // ── 头部轨迹（独立 Kalman 滤波器）──
    // 遮挡期人体框互相重叠（body IoU 失效/陷阱），但头部很少重叠：
    // 即使头被挡住，头部 KF 纯预测出的框也仍在正确区域、不与他人头部重叠。
    // → 用主目标头部预测框与候选头部检测框的 IoU 作为遮挡期可靠空间判别信号。
    std::unique_ptr<KalmanFilterNew> head_kf_;   // 头部 6 维 KF [cx,cy,w,h,vx,vy]
    cv::Mat head_last_obs_;                       // 最近头部观测框 [1,4] xyxy
    cv::Mat head_pred_box_;                       // 本帧头部预测框 [1,4] xyxy（可被 GMC 补偿）
    int     head_time_since_update_ = 0;          // 头部 KF 距上次观测的帧数
    bool    head_valid_ = false;                  // 头部轨迹是否已建立

    // 头部 KF 预测一帧；未建立轨迹时返回空矩阵。结果同时写入 head_pred_box_
    cv::Mat predict_head();
    // 头部 KF 更新：有观测则校正（首次观测懒初始化），无观测则纯预测推进
    void    update_head(std::optional<cv::Mat> head_bbox);
    // 强人脸在全局新位置确认且同帧有真实头时，重建头 KF 到该真实头；用于清除
    // 遮挡前旧位置/速度，避免脸消失后的 head-only 又跳回旧轨迹。
    void    rebase_head(const cv::Mat& head_bbox);
    // 强人脸已在全局新位置确认、但该处没有可靠头检测时，旧头 KF 不能继续提供瞄准点。
    void    clear_head_track();
    // 头部 KF 当前状态框 [1,4] xyxy
    cv::Mat get_head_state();

    bool    has_head_track() const { return head_valid_; }
    cv::Mat get_head_pred_box() const { return head_pred_box_; }
    void    set_head_pred_box(const cv::Mat& b) { head_pred_box_ = b.clone(); }
    int     get_head_time_since_update() const { return head_time_since_update_; }

    // GMC 状态级补偿：把相机运动仿射 M（2x3, CV_64F）施加到 body/head KF 内部状态。
    // 行级补偿只校正"本帧输出副本"，且 head_pred_box_ 的写回会被下帧 predict_head()
    // 从未补偿状态重新生成而覆盖 → 多帧丢失 + 云台移动时预测累计滞后。
    // 状态级补偿（BoT-SORT 做法）使补偿逐帧累计，预测始终在当前帧坐标系。
    void    apply_camera_motion(const cv::Mat& M);

    // Below variables are newly added in ocsort
    // A6：哨兵初始化为 -1（OC-SORT 原语义）。update() 以 sum>=0 判"已有历史观测"，
    // 旧 zeros 初始化使该哨兵恒真 → 首次 update 用零框算 speed_direction（一帧垃圾速度）。
    // generate_final_results 的"有观测才输出观测框"判定已同步改为 sum>=0。
    cv::Mat last_observation = cv::Mat(1, 4, CV_32F, cv::Scalar(-1.0f)); // [1,4] 行向量，float类型
    std::map<int, cv::Mat> observations; // 存储4x1列向量，float类型
    cv::Mat velocity = cv::Mat::zeros(1, 2, CV_32F); // [1,2] 行向量，float类型
    int delta_t;
    bool faceVerified = false;

private:
    cv::Mat body_good_x_;
    cv::Mat body_good_P_;
    cv::Mat head_good_x_;
    cv::Mat head_good_P_;
    bool long_coast_ = false;
    std::deque<std::string> numerical_events_;

    void record_numerical_event(const std::string& event);
    bool body_filter_healthy() const;
    bool head_filter_healthy() const;
    void save_body_snapshot();
    void save_head_snapshot();
    bool restore_body_snapshot(const char* stage);
    bool restore_head_snapshot(const char* stage);
    bool reinitialize_body_filter(const cv::Mat& bbox, const char* reason);
};

#endif // KALMANBOXTRACKER_H
