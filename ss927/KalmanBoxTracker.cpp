#include "KalmanBoxTracker.h"
#include "utils.h"
#include <utility>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cfloat>   // FLT_MAX（add_anchor_sample 分带淘汰用）

int KalmanBoxTracker::count = 0;

namespace {
// 速度状态保留为“每个 40ms 标称帧”的像素位移；预测时按真实帧间隔缩放，
// 这样不破坏既有速度历史，同时修复模型调度导致的变帧间隔预测误差。
constexpr float kNominalFrameDtSec = 0.040f;
constexpr float kMinDtScale = 0.50f;
constexpr float kMaxDtScale = 4.00f;
constexpr float kBodyVelDecayPerNominalFrame = 0.88f;
constexpr float kHeadVelDecayPerNominalFrame = 0.90f;
constexpr float kCovarianceMaxAbs = 1.0e6f;
constexpr float kCovarianceMinDiag = 1.0e-3f;
constexpr float kBoxCoordinateMaxAbs = 1.0e7f;
constexpr std::size_t kNumericalEventMax = 8;

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

void configure_dynamics(KalmanFilterNew& filter, float dt_scale, bool head_track) {
    const float decay = std::pow(head_track ? kHeadVelDecayPerNominalFrame
                                             : kBodyVelDecayPerNominalFrame,
                                 dt_scale);
    filter.F = cv::Mat::eye(6, 6, CV_32F);
    filter.F.at<float>(0, 4) = dt_scale;
    filter.F.at<float>(1, 5) = dt_scale;
    filter.F.at<float>(4, 4) = decay;
    filter.F.at<float>(5, 5) = decay;

    // 像素方差。头框相对更小、抖动占比更高，故允许更大的位置与速度变化。
    const float pos_q  = head_track ? 1.5f  : 1.0f;
    const float size_q = 1.0f;
    const float vel_q  = head_track ? 16.0f : 9.0f;
    filter.Q = cv::Mat::zeros(6, 6, CV_32F);
    filter.Q.at<float>(0, 0) = pos_q * dt_scale;
    filter.Q.at<float>(1, 1) = pos_q * dt_scale;
    filter.Q.at<float>(2, 2) = size_q * dt_scale;
    filter.Q.at<float>(3, 3) = size_q * dt_scale;
    filter.Q.at<float>(4, 4) = vel_q * dt_scale;
    filter.Q.at<float>(5, 5) = vel_q * dt_scale;
}

void configure_measurement_noise(KalmanFilterNew& filter, const cv::Mat& bbox,
                                 bool head_track) {
    if (bbox.empty() || bbox.cols < 4) return;
    const float w = std::max(1.f, std::abs(bbox.at<float>(0, 2) - bbox.at<float>(0, 0)));
    const float h = std::max(1.f, std::abs(bbox.at<float>(0, 3) - bbox.at<float>(0, 1)));
    const float diag = std::sqrt(w * w + h * h);
    const float center_sigma = head_track
        ? clampf(diag * 0.040f, 1.5f, 10.0f)
        : clampf(diag * 0.025f, 2.0f, 12.0f);
    const float size_ratio = head_track ? 0.060f : 0.040f;
    const float width_sigma = clampf(w * size_ratio, head_track ? 1.5f : 2.0f,
                                     head_track ? 10.0f : 12.0f);
    const float height_sigma = clampf(h * size_ratio, head_track ? 1.5f : 2.0f,
                                      head_track ? 10.0f : 14.0f);
    filter.R = cv::Mat::zeros(4, 4, CV_32F);
    filter.R.at<float>(0, 0) = center_sigma * center_sigma;
    filter.R.at<float>(1, 1) = center_sigma * center_sigma;
    filter.R.at<float>(2, 2) = width_sigma * width_sigma;
    filter.R.at<float>(3, 3) = height_sigma * height_sigma;
}

// 部件重建框的中心通常可靠，但宽高来自历史比例而非真实人体检测。使用明显更大的
// 测量噪声，避免连续头/脸观测把人体 KF 错误收紧成“看似高置信”的伪人体轨迹。
void configure_part_measurement_noise(KalmanFilterNew& filter, const cv::Mat& bbox,
                                      bool from_face) {
    if (bbox.empty() || bbox.cols < 4) return;
    const float w = std::max(1.f, std::abs(bbox.at<float>(0, 2) - bbox.at<float>(0, 0)));
    const float h = std::max(1.f, std::abs(bbox.at<float>(0, 3) - bbox.at<float>(0, 1)));
    const float diag = std::sqrt(w * w + h * h);
    const float center_sigma = clampf(diag * (from_face ? 0.05f : 0.06f), 4.f, 28.f);
    const float width_sigma  = std::max(8.f,  w * 0.25f);
    const float height_sigma = std::max(12.f, h * 0.30f);
    filter.R = cv::Mat::zeros(4, 4, CV_32F);
    filter.R.at<float>(0, 0) = center_sigma * center_sigma;
    filter.R.at<float>(1, 1) = center_sigma * center_sigma;
    filter.R.at<float>(2, 2) = width_sigma * width_sigma;
    filter.R.at<float>(3, 3) = height_sigma * height_sigma;
}

void configure_initial_covariance(KalmanFilterNew& filter, bool head_track) {
    const float pos_p = head_track ? 16.f : 25.f;
    const float size_p = head_track ? 16.f : 36.f;
    const float vel_p = head_track ? 64.f : 100.f;
    filter.P = cv::Mat::eye(6, 6, CV_32F);
    filter.P.at<float>(0, 0) = pos_p;
    filter.P.at<float>(1, 1) = pos_p;
    filter.P.at<float>(2, 2) = size_p;
    filter.P.at<float>(3, 3) = size_p;
    filter.P.at<float>(4, 4) = vel_p;
    filter.P.at<float>(5, 5) = vel_p;
}

bool valid_bbox_xyxy(const cv::Mat& bbox) {
    if (bbox.empty() || bbox.type() != CV_32F || bbox.rows < 1 || bbox.cols < 4)
        return false;
    if (!cv::checkRange(bbox.row(0).colRange(0, 4), true, nullptr, -FLT_MAX, FLT_MAX))
        return false;
    const float x1 = bbox.at<float>(0, 0);
    const float y1 = bbox.at<float>(0, 1);
    const float x2 = bbox.at<float>(0, 2);
    const float y2 = bbox.at<float>(0, 3);
    const float w = x2 - x1;
    const float h = y2 - y1;
    return std::isfinite(w) && std::isfinite(h)
        && std::abs(x1) <= kBoxCoordinateMaxAbs
        && std::abs(y1) <= kBoxCoordinateMaxAbs
        && std::abs(x2) <= kBoxCoordinateMaxAbs
        && std::abs(y2) <= kBoxCoordinateMaxAbs
        && w >= 1.f && h >= 1.f;
}

bool finite_filter_state(const KalmanFilterNew* filter) {
    if (!filter || filter->x.type() != CV_32F || filter->P.type() != CV_32F
        || filter->x.rows != 6 || filter->x.cols != 1
        || filter->P.rows != 6 || filter->P.cols != 6) {
        return false;
    }
    if (!cv::checkRange(filter->x, true, nullptr, -FLT_MAX, FLT_MAX)
        || !cv::checkRange(filter->P, true, nullptr, -FLT_MAX, FLT_MAX)) {
        return false;
    }
    for (int i = 0; i < 6; ++i) {
        if (std::abs(filter->x.at<float>(i, 0)) > kBoxCoordinateMaxAbs)
            return false;
    }
    const float w = filter->x.at<float>(2, 0);
    const float h = filter->x.at<float>(3, 0);
    if (!std::isfinite(w) || !std::isfinite(h) || w < 1.f || h < 1.f)
        return false;
    for (int i = 0; i < 6; ++i) {
        const float d = filter->P.at<float>(i, i);
        if (!std::isfinite(d) || d < 0.f) return false;
    }
    return true;
}

bool stabilize_covariance(KalmanFilterNew& filter) {
    if (filter.P.empty() || filter.P.type() != CV_32F
        || filter.P.rows != 6 || filter.P.cols != 6
        || !cv::checkRange(filter.P, true, nullptr, -FLT_MAX, FLT_MAX)) {
        return false;
    }

    // 保持对称；若整体尺度过大则等比例缩放，避免逐元素裁剪破坏半正定结构。
    filter.P = 0.5f * (filter.P + filter.P.t());
    cv::Mat abs_p;
    cv::absdiff(filter.P, cv::Scalar(0), abs_p);
    double max_abs = 0.0;
    cv::minMaxLoc(abs_p, nullptr, &max_abs);
    if (!std::isfinite(max_abs)) return false;
    if (max_abs > kCovarianceMaxAbs)
        filter.P *= (float)(kCovarianceMaxAbs / max_abs);

    for (int i = 0; i < 6; ++i) {
        float& d = filter.P.at<float>(i, i);
        if (!std::isfinite(d)) return false;
        d = clampf(d, kCovarianceMinDiag, kCovarianceMaxAbs);
    }
    return cv::checkRange(filter.P, true, nullptr, -FLT_MAX, FLT_MAX);
}

std::unique_ptr<KalmanFilterNew> build_box_kf(const cv::Mat& bbox, bool head_track) {
    if (!valid_bbox_xyxy(bbox)) return nullptr;

    auto filter = std::make_unique<KalmanFilterNew>(6, 4);
    configure_dynamics(*filter, 1.0f, head_track);

    filter->H = cv::Mat::zeros(4, 6, CV_32F);
    filter->H.at<float>(0, 0) = 1.0f;
    filter->H.at<float>(1, 1) = 1.0f;
    filter->H.at<float>(2, 2) = 1.0f;
    filter->H.at<float>(3, 3) = 1.0f;

    configure_measurement_noise(*filter, bbox, head_track);
    configure_initial_covariance(*filter, head_track);

    cv::Mat z = Utils::convert_bbox_to_z(bbox);
    if (z.type() != CV_32F || z.rows != 4 || z.cols != 1
        || !cv::checkRange(z, true, nullptr, -FLT_MAX, FLT_MAX)) {
        return nullptr;
    }
    filter->x = cv::Mat::zeros(6, 1, CV_32F);
    z.copyTo(filter->x.rowRange(0, 4));
    filter->x_prior = filter->x.clone();
    filter->P_prior = filter->P.clone();
    filter->x_post = filter->x.clone();
    filter->P_post = filter->P.clone();
    if (!finite_filter_state(filter.get())) return nullptr;
    return filter;
}
} // namespace

KalmanBoxTracker::KalmanBoxTracker(cv::Mat bbox_, int delta_t_,
    cv::Mat emb_new, bool ismain) {
    is_main = ismain;
    delta_t = 1;  // PTZ 单目标场景建议固定为 1
    emb = emb_new.clone();

    // ═══════════════════════════════════════════════════
    // 6 维状态：[cx, cy, w, h, vx, vy]
    // ═══════════════════════════════════════════════════
    kf = build_box_kf(bbox_, false);
    if (!kf) {
        // 正常检测路径不会进入这里；保留一个有限的最小状态，避免构造期异常
        // 直接制造永久 NaN。该轨迹随后只能靠有效强观测重建。
        cv::Mat fallback = (cv::Mat_<float>(1, 4) << 0.f, 0.f, 1.f, 1.f);
        kf = build_box_kf(fallback, false);
    }

    speed = 1;
    time_since_update = 0;
    id = KalmanBoxTracker::count;
    KalmanBoxTracker::count += 1;
    hits = 0;
    hit_streak = 0;
    age = 0;
    last_observation = cv::Mat(1, 4, CV_32F, cv::Scalar(-1.0f));   // A6：-1 哨兵（见 .h）
    observations.clear();
    velocity = cv::Mat::zeros(1, 2, CV_32F);
    save_body_snapshot();
}

void KalmanBoxTracker::record_numerical_event(const std::string& event) {
    if (numerical_events_.size() >= kNumericalEventMax)
        numerical_events_.pop_front();
    numerical_events_.push_back(event);
}

std::vector<std::string> KalmanBoxTracker::consume_numerical_events() {
    std::vector<std::string> out;
    out.reserve(numerical_events_.size());
    while (!numerical_events_.empty()) {
        out.push_back(std::move(numerical_events_.front()));
        numerical_events_.pop_front();
    }
    return out;
}

bool KalmanBoxTracker::body_filter_healthy() const {
    return finite_filter_state(kf.get());
}

bool KalmanBoxTracker::head_filter_healthy() const {
    return head_valid_ && finite_filter_state(head_kf_.get());
}

void KalmanBoxTracker::save_body_snapshot() {
    if (!body_filter_healthy()) return;
    body_good_x_ = kf->x.clone();
    body_good_P_ = kf->P.clone();
}

void KalmanBoxTracker::save_head_snapshot() {
    if (!head_filter_healthy()) return;
    head_good_x_ = head_kf_->x.clone();
    head_good_P_ = head_kf_->P.clone();
}

bool KalmanBoxTracker::restore_body_snapshot(const char* stage) {
    if (body_good_x_.empty() || body_good_P_.empty()
        || body_good_x_.type() != CV_32F || body_good_P_.type() != CV_32F
        || body_good_x_.rows != 6 || body_good_x_.cols != 1
        || body_good_P_.rows != 6 || body_good_P_.cols != 6
        || !cv::checkRange(body_good_x_, true, nullptr, -FLT_MAX, FLT_MAX)
        || !cv::checkRange(body_good_P_, true, nullptr, -FLT_MAX, FLT_MAX)) {
        return false;
    }

    kf->x = body_good_x_.clone();
    kf->P = body_good_P_.clone();
    kf->x.at<float>(4, 0) = 0.f;
    kf->x.at<float>(5, 0) = 0.f;
    kf->P *= 4.f;
    const float floors[6] = {100.f, 100.f, 144.f, 196.f, 100.f, 100.f};
    for (int i = 0; i < 6; ++i)
        kf->P.at<float>(i, i) = std::max(kf->P.at<float>(i, i), floors[i]);
    if (!stabilize_covariance(*kf) || !body_filter_healthy()) return false;

    kf->x_prior = kf->x.clone();
    kf->P_prior = kf->P.clone();
    kf->x_post = kf->x.clone();
    kf->P_post = kf->P.clone();
    save_body_snapshot();
    record_numerical_event(std::string("RESET body action=restore_snapshot stage=")
                           + (stage ? stage : "unknown"));
    return true;
}

bool KalmanBoxTracker::restore_head_snapshot(const char* stage) {
    if (!head_kf_ || head_good_x_.empty() || head_good_P_.empty()
        || head_good_x_.type() != CV_32F || head_good_P_.type() != CV_32F
        || head_good_x_.rows != 6 || head_good_x_.cols != 1
        || head_good_P_.rows != 6 || head_good_P_.cols != 6
        || !cv::checkRange(head_good_x_, true, nullptr, -FLT_MAX, FLT_MAX)
        || !cv::checkRange(head_good_P_, true, nullptr, -FLT_MAX, FLT_MAX)) {
        return false;
    }

    head_kf_->x = head_good_x_.clone();
    head_kf_->P = head_good_P_.clone();
    head_kf_->x.at<float>(4, 0) = 0.f;
    head_kf_->x.at<float>(5, 0) = 0.f;
    head_kf_->P *= 4.f;
    const float floors[6] = {64.f, 64.f, 64.f, 64.f, 100.f, 100.f};
    for (int i = 0; i < 6; ++i)
        head_kf_->P.at<float>(i, i) =
            std::max(head_kf_->P.at<float>(i, i), floors[i]);
    if (!stabilize_covariance(*head_kf_) || !head_filter_healthy()) return false;

    head_kf_->x_prior = head_kf_->x.clone();
    head_kf_->P_prior = head_kf_->P.clone();
    head_kf_->x_post = head_kf_->x.clone();
    head_kf_->P_post = head_kf_->P.clone();
    save_head_snapshot();
    record_numerical_event(std::string("RESET head action=restore_snapshot stage=")
                           + (stage ? stage : "unknown"));
    return true;
}

bool KalmanBoxTracker::reinitialize_body_filter(const cv::Mat& bbox,
                                                const char* reason) {
    std::unique_ptr<KalmanFilterNew> rebuilt = build_box_kf(bbox, false);
    if (!rebuilt) return false;
    kf = std::move(rebuilt);
    long_coast_ = false;
    velocity = cv::Mat::zeros(1, 2, CV_32F);
    speed = 0.f;
    save_body_snapshot();
    record_numerical_event(std::string("RESET body action=reinitialize reason=")
                           + (reason ? reason : "unknown"));
    return true;
}

cv::Mat KalmanBoxTracker::predict() {
    if (!body_filter_healthy()) {
        record_numerical_event("NAN body stage=pre_predict");
        if (!restore_body_snapshot("pre_predict")) {
            if (!valid_bbox_xyxy(last_observation)
                || !reinitialize_body_filter(last_observation,
                                             "pre_predict_last_observation")) {
                return cv::Mat();
            }
        }
    }

    // 长时间完全失联后，速度外推已没有物理意义。冻结速度但仍推进 age/tsu，
    // 让全画面 ReID/人脸重捕逻辑继续工作。
    if (long_coast_ && time_since_update > 0 && kf->x.rows >= 6) {
        kf->x.at<float>(4, 0) = 0.f;
        kf->x.at<float>(5, 0) = 0.f;
    }

    if (!kf->predict() || !stabilize_covariance(*kf)
        || !body_filter_healthy()) {
        record_numerical_event("NAN body stage=predict");
        if (!restore_body_snapshot("predict")) return cv::Mat();
    } else {
        save_body_snapshot();
    }

    age += 1;

    if (time_since_update > 0) {
        hit_streak = 0;
    }

    time_since_update += 1;
    time_since_update_emb += 1;

    cv::Rect2f vec_out = Utils::convert_z_to_bbox(kf->x);
    cv::Mat rectMat = (cv::Mat_<float>(1, 4) << vec_out.x, vec_out.y, vec_out.width, vec_out.height);
    return valid_bbox_xyxy(rectMat) ? rectMat : cv::Mat();
}

void KalmanBoxTracker::update(std::optional<cv::Mat> bbox_) {
    if (bbox_.has_value()) {
        cv::Mat bbox = bbox_.value();
        if (!valid_bbox_xyxy(bbox)) {
            record_numerical_event("REJECT body stage=update reason=invalid_bbox");
            return;
        }

        const bool rebuilt_after_long_coast = long_coast_;
        bool state_ok = false;
        if (rebuilt_after_long_coast) {
            // 长盲后的首个已提交人体检测是强观测，直接重建全部状态（含宽高/P），
            // 不让旧位置或旧协方差拖慢重捕。
            state_ok = reinitialize_body_filter(bbox, "strong_body_after_long_coast");
        } else {
            if (!body_filter_healthy()) {
                record_numerical_event("NAN body stage=pre_update");
                (void)restore_body_snapshot("pre_update");
            }
            if (body_filter_healthy()) {
                configure_measurement_noise(*kf, bbox, false);
                cv::Mat z = Utils::convert_bbox_to_z(bbox);
                state_ok = kf->update(z)
                    && stabilize_covariance(*kf) && body_filter_healthy();
            }
            if (!state_ok) {
                record_numerical_event("NAN body stage=update");
                state_ok = reinitialize_body_filter(bbox, "strong_body_update_failed");
            }
        }
        if (!state_ok) return;
        save_body_snapshot();

        if (rebuilt_after_long_coast) {
            // OC-SORT 的独立方向历史也不能跨越长失联窗口，否则“旧位置→全局重捕位置”
            // 会制造一个巨大的伪速度并影响下一帧方向门。
            velocity = cv::Mat::zeros(1, 2, CV_32F);
            speed = 0.f;
            observations.clear();
        } else if (cv::sum(last_observation)[0] >= 0) {
            cv::Mat previous_box_tmp;
            bool found = false;

            for (int dt = delta_t; dt > 0; --dt) {
                auto it = observations.find(age - dt);
                if (it != observations.end()) {
                    previous_box_tmp = it->second.clone();
                    found = true;
                    break;
                }
            }

            if (!found || previous_box_tmp.empty()) {
                previous_box_tmp = last_observation.clone();  
            }
            const int maxSize = 10;
            if (observations.size() > maxSize) {
                observations.erase(observations.begin());
            }

            auto speeds = Utils::speed_direction(previous_box_tmp, bbox);
            velocity = speeds.first;
            speed = speeds.second;
        }

        last_observation = bbox.clone();
        observations[age] = bbox.clone();

        time_since_update = 0;
        hits += 1;
        hit_streak += 1;
    }
    else {
        if (!kf->update(cv::Mat())) {
            record_numerical_event("NAN body stage=empty_update");
            (void)restore_body_snapshot("empty_update");
        }
    }
}


// KalmanBoxTracker.h / .cpp 中修改：
void KalmanBoxTracker::update_emb(const cv::Mat& new_emb, float alpha) {
    // A12：new_emb 为空（compute_embedding 推理失败）时，下方 EMA 矩阵表达式
    // 对空操作数抛 cv::Exception（LightTracker::update_trackers_unified 的
    // 回退推理路径未挡空）。空样本 = 本帧无更新，模板保持原值。
    if (new_emb.empty()) return;
    if (emb.empty()) {
        emb = new_emb.clone();
        return;
    }
    
    // EMA: emb = (1 - alpha) * old + alpha * new
    emb = (1.0f - alpha) * emb + alpha * new_emb;
    
    // 重新归一化（余弦相似度需要）
    float norm = cv::norm(emb);
    if (norm > 1e-6f)
        emb /= norm;
}

// ═══════════════════════════════════════════════════════════
// 多样本锚点画廊
//   anchor_sim = 对画廊取最大余弦。用 max 会"放宽"身份门，因此向画廊
//   添加样本必须保守（高置信 + 多样性 + 质量淘汰），见 add_anchor_sample。
// ═══════════════════════════════════════════════════════════
namespace {
    constexpr int   kAnchorGalleryMax = 6;      // 画廊容量（Q1：3→6，容纳每可见度带各若干样本）
    constexpr int   kAnchorPerBand    = 2;      // Q1：每可见度带样本上限（2×3 带 = 6）
    constexpr float kAnchorDupThresh  = 0.92f;  // 与既有样本相似度 ≥ 此值视为"同一视角"
    constexpr float kAnchorRegQuality = 0.5f;   // 注册样本基线质量（可被更高质量样本淘汰）

    // Q1：可见度分带（与 VisibilityState 量化阈值对齐）。HIGH=全身/大半身，MID=半身，LOW=上身/仅头。
    inline int anchor_band_of(float vis) {
        if (vis >= 0.60f) return 0;   // HIGH
        if (vis >= 0.35f) return 1;   // MID
        return 2;                     // LOW
    }
}

void KalmanBoxTracker::set_anchor_emb(const cv::Mat& e) {
    anchor_gallery_.clear();
    if (e.empty()) return;
    cv::Mat s = e.clone();
    float n = (float)cv::norm(s);
    if (n > 1e-6f) s /= n;
    anchor_gallery_.push_back({ s, kAnchorRegQuality, 1.f });   // 注册假定全身可见（vis=1.0）
}

cv::Mat KalmanBoxTracker::get_anchor_emb() const {
    // 返回最高质量样本（danger 回退 / alert 回滚 需要单一代表）
    const AnchorSample* best = nullptr;
    for (const auto& s : anchor_gallery_)
        if (!best || s.quality > best->quality) best = &s;
    return best ? best->emb.clone() : cv::Mat();
}

float KalmanBoxTracker::anchor_sim_max(const cv::Mat& feat) const {
    if (anchor_gallery_.empty() || feat.empty()) return -1.f;
    float m = -1.f;
    for (const auto& s : anchor_gallery_) {
        if (s.emb.empty()) continue;
        float sim = (float)feat.dot(s.emb);   // 双方均归一化 → 余弦
        if (sim > m) m = sim;
    }
    return m;
}

// Q1：仅比对与当前可见度同带的样本（半身查询↔半身参考），本带无样本则回退全画廊。
float KalmanBoxTracker::anchor_sim_vis(const cv::Mat& feat, float cur_vis) const {
    if (anchor_gallery_.empty() || feat.empty()) return -1.f;
    int band = anchor_band_of(cur_vis);
    float m = -1.f;
    for (const auto& s : anchor_gallery_) {
        if (s.emb.empty()) continue;
        if (anchor_band_of(s.vis) != band) continue;   // 仅同可见度带
        float sim = (float)feat.dot(s.emb);
        if (sim > m) m = sim;
    }
    if (m < 0.f) return anchor_sim_max(feat);   // 本带尚无样本 → 回退全画廊（原行为，无回归）
    return m;
}

void KalmanBoxTracker::add_anchor_sample(const cv::Mat& e, float quality, float vis) {
    if (e.empty()) return;
    cv::Mat s = e.clone();
    float n = (float)cv::norm(s);
    if (n > 1e-6f) s /= n;
    int band = anchor_band_of(vis);

    if (anchor_gallery_.empty()) {
        anchor_gallery_.push_back({ s, quality, vis });
        return;
    }

    // Q1 分带维护：统计每带计数 + 每带最低质量槽 + 同带去重
    int   band_count[3]    = { 0, 0, 0 };
    int   band_minq_idx[3] = { -1, -1, -1 };
    float band_minq[3]     = { FLT_MAX, FLT_MAX, FLT_MAX };
    int   sim_idx = -1;  float sim_max = -1.f;   // 同带最相似样本（去重）
    for (int i = 0; i < (int)anchor_gallery_.size(); ++i) {
        int bi = anchor_band_of(anchor_gallery_[i].vis);
        band_count[bi]++;
        if (anchor_gallery_[i].quality < band_minq[bi]) {
            band_minq[bi] = anchor_gallery_[i].quality; band_minq_idx[bi] = i;
        }
        if (bi == band) {   // 仅同带做去重比较
            float sim = (float)s.dot(anchor_gallery_[i].emb);
            if (sim > sim_max) { sim_max = sim; sim_idx = i; }
        }
    }

    // 同带近重复视角 → 仅质量更高时刷新该槽，避免冗余
    if (sim_idx >= 0 && sim_max >= kAnchorDupThresh) {
        if (quality > anchor_gallery_[sim_idx].quality)
            anchor_gallery_[sim_idx] = { s, quality, vis };
        return;
    }

    // 未满 → 直接加入（增加多样性 / 带覆盖）
    if ((int)anchor_gallery_.size() < kAnchorGalleryMax) {
        anchor_gallery_.push_back({ s, quality, vis });
        return;
    }

    // 已满且本带已达上限 → 淘汰"同带最低质量"（质量自愈，不跨带，防同带无限膨胀）。
    if (band_count[band] >= kAnchorPerBand) {
        if (band_minq_idx[band] >= 0 && quality > band_minq[band])
            anchor_gallery_[band_minq_idx[band]] = { s, quality, vis };
        return;
    }

    // 已满但本带欠代表（低于上限）→ 从"超额带"淘汰其最低质量样本为本带腾位。
    //   关键：带覆盖优先于边际质量 —— 低可见度样本仅来自人脸确认（身份确定），即便余弦
    //   质量偏低也值得保留一个同带参考（否则常见的"先全身后遮挡"会让 LOW 带永远空、Q1 空转）。
    //   故此处无条件替换（不设 quality 门）。
    int victim = -1; float victim_q = FLT_MAX;
    for (int b = 0; b < 3; ++b) {
        if (b == band) continue;
        if (band_count[b] > kAnchorPerBand && band_minq[b] < victim_q) {
            victim_q = band_minq[b]; victim = band_minq_idx[b];
        }
    }
    if (victim >= 0) {
        anchor_gallery_[victim] = { s, quality, vis };
        return;
    }
    // 兜底（满且本带欠代表时理论上必有超额带；防御性）：质量门下淘汰全局最低。
    int gmin_idx = 0; float gmin_q = anchor_gallery_[0].quality;
    for (int i = 1; i < (int)anchor_gallery_.size(); ++i)
        if (anchor_gallery_[i].quality < gmin_q) { gmin_q = anchor_gallery_[i].quality; gmin_idx = i; }
    if (quality > gmin_q) anchor_gallery_[gmin_idx] = { s, quality, vis };
}


cv::Mat KalmanBoxTracker::get_state() {
    const cv::Mat* state = nullptr;
    if (body_filter_healthy()) state = &kf->x;
    else if (!body_good_x_.empty()
             && cv::checkRange(body_good_x_, true, nullptr, -FLT_MAX, FLT_MAX))
        state = &body_good_x_;
    if (!state) return cv::Mat::zeros(1, 4, CV_32F);

    cv::Rect2f vec_out = Utils::convert_z_to_bbox(*state);
    cv::Mat rectMat = (cv::Mat_<float>(1, 4) << vec_out.x, vec_out.y, vec_out.width, vec_out.height);
    return valid_bbox_xyxy(rectMat) ? rectMat : cv::Mat::zeros(1, 4, CV_32F);
}

float KalmanBoxTracker::get_kf_speed() const {
    if (!body_filter_healthy()) return 0.f;
    float vx = kf->x.at<float>(4, 0);
    float vy = kf->x.at<float>(5, 0);
    const float v = std::sqrt(vx * vx + vy * vy);
    return std::isfinite(v) ? v : 0.f;
}

cv::Vec2f KalmanBoxTracker::get_kf_velocity() const {
    if (!body_filter_healthy()) return cv::Vec2f(0.f, 0.f);
    return cv::Vec2f(kf->x.at<float>(4, 0), kf->x.at<float>(5, 0));
}

void KalmanBoxTracker::correct_body_from_part(const cv::Mat& bbox_, bool from_face,
                                              bool relocate_center) {
    if (!valid_bbox_xyxy(bbox_) || !kf) {
        record_numerical_event("REJECT body stage=part_update reason=invalid_bbox");
        return;
    }

    // 已确认头/脸观测能够恢复一个已经污染的完整人体 KF；只改变数值状态，
    // 不改变真实人体观测的 time_since_update/hits/last_observation 语义。
    if (long_coast_ || !body_filter_healthy()) {
        if (!body_filter_healthy())
            record_numerical_event("NAN body stage=pre_part_update");
        if (!reinitialize_body_filter(
                bbox_, from_face ? "confirmed_face_part" : "accepted_head_part")) {
            return;
        }
        const float floors[6] = {36.f, 36.f, 64.f, 100.f, 36.f, 36.f};
        for (int i = 0; i < 6; ++i)
            kf->P.at<float>(i, i) = std::max(kf->P.at<float>(i, i), floors[i]);
        (void)stabilize_covariance(*kf);
        save_body_snapshot();
        return;
    }

    configure_part_measurement_noise(*kf, bbox_, from_face);
    cv::Mat z = Utils::convert_bbox_to_z(bbox_);

    // 全画面人脸已经给出强身份直证时，大位移不是普通检测噪声。如果仍只走高 R 的
    // 弱校正，人体 KF 会在旧位置滞留数帧，下一帧找脸又会围绕旧中心搜索。这里仅把
    // 中心重定位到人脸/头部重构位置，宽高仍由高噪声量测缓慢校正；同时清零旧速度，
    // 防止遮挡前速度在新位置产生反向过冲。仍不重置 time_since_update/hits/last_obs，
    // 因此该伪框不会冒充真实人体观测。
    if (relocate_center && z.rows >= 4 && z.cols == 1
        && kf->x.rows >= 6 && kf->x.cols == 1) {
        kf->x.at<float>(0, 0) = z.at<float>(0, 0);
        kf->x.at<float>(1, 0) = z.at<float>(1, 0);
        kf->x.at<float>(4, 0) = 0.f;
        kf->x.at<float>(5, 0) = 0.f;
    }
    if (!kf->update(z)) {
        record_numerical_event("NAN body stage=part_update");
        (void)reinitialize_body_filter(
            bbox_, from_face ? "confirmed_face_update_failed"
                             : "accepted_head_update_failed");
    }

    // 伪观测不能让协方差无限收缩；保留足够位置、尺寸和速度不确定性，使下一次
    // 真实人体检测仍能迅速拉回，而不会被长期部件重建框“锁死”。
    const float floors[6] = {36.f, 36.f, 64.f, 100.f, 36.f, 36.f};
    for (int i = 0; i < 6; ++i)
        kf->P.at<float>(i, i) = std::max(kf->P.at<float>(i, i), floors[i]);
    if (!stabilize_covariance(*kf) || !body_filter_healthy()) {
        record_numerical_event("NAN body stage=part_post");
        (void)reinitialize_body_filter(
            bbox_, from_face ? "confirmed_face_post_failed"
                             : "accepted_head_post_failed");
    }
    save_body_snapshot();
}

float KalmanBoxTracker::get_kf_speed_per_sec() const {
    return get_kf_speed() / kNominalFrameDtSec;
}

cv::Vec2f KalmanBoxTracker::get_kf_velocity_per_sec() const {
    cv::Vec2f v = get_kf_velocity();
    return cv::Vec2f(v[0] / kNominalFrameDtSec, v[1] / kNominalFrameDtSec);
}

void KalmanBoxTracker::set_frame_interval(float dt_sec) {
    if (!std::isfinite(dt_sec) || dt_sec <= 0.f) dt_sec = kNominalFrameDtSec;
    const float dt_scale = clampf(dt_sec / kNominalFrameDtSec, kMinDtScale, kMaxDtScale);
    if (kf) configure_dynamics(*kf, dt_scale, false);
    if (head_valid_ && head_kf_) configure_dynamics(*head_kf_, dt_scale, true);
}

// ═══════════════════════════════════════════════════════════
// 头部轨迹 Kalman 滤波器（独立于人体 KF）
//   状态/构造与人体 KF 完全一致（6 维 CV 模型 [cx,cy,w,h,vx,vy]），
//   仅用于主目标，懒初始化（首次拿到头部观测时建立）。
// ═══════════════════════════════════════════════════════════
namespace {
// 头部 KF：与人体 KF 结构相同（6 维 [cx,cy,w,h,vx,vy]），但 **动力学参数不同**。
//   理由：PTZ 跟随时人体被补偿到近乎静止（极小过程噪声 + 强速度衰减），
//   但头部相对画面运动更大、更"灵活"（转头/点头/行走头部晃动），且头框很小
//   → 同样的像素位移对头部 IoU 的影响被放大（人体 IoU 0.5 时头部可能已 0 重叠）。
//   若沿用人体的"近静止"刚性参数，头部预测会滞后 → head_match 误判 → 误否决真目标。
//   故：放大过程噪声（位置/速度）+ 放宽速度衰减，让头部预测跟得上真实运动。
//   （以下为器件调参起点，建议在真机上按头部检测帧间位移分布微调。）
std::unique_ptr<KalmanFilterNew> build_head_kf(const cv::Mat& bbox) {
    return build_box_kf(bbox, true);
}
} // namespace

cv::Mat KalmanBoxTracker::predict_head() {
    if (!head_valid_ || !head_kf_) {
        head_pred_box_ = cv::Mat();
        return cv::Mat();
    }

    if (!head_filter_healthy()) {
        record_numerical_event("NAN head stage=pre_predict");
        if (!restore_head_snapshot("pre_predict")) {
            if (valid_bbox_xyxy(head_last_obs_)) {
                rebase_head(head_last_obs_);
                record_numerical_event(
                    "RESET head action=reinitialize reason=last_observation");
            } else {
                head_valid_ = false;
                head_kf_.reset();
                head_pred_box_ = cv::Mat();
                record_numerical_event(
                    "RESET head action=invalidate reason=no_finite_fallback");
                return cv::Mat();
            }
        }
    }

    if (long_coast_ && head_time_since_update_ > 0 && head_kf_->x.rows >= 6) {
        head_kf_->x.at<float>(4, 0) = 0.f;
        head_kf_->x.at<float>(5, 0) = 0.f;
    }

    if (!head_kf_->predict() || !stabilize_covariance(*head_kf_)
        || !head_filter_healthy()) {
        record_numerical_event("NAN head stage=predict");
        if (!restore_head_snapshot("predict")) {
            head_pred_box_ = cv::Mat();
            return cv::Mat();
        }
    } else {
        save_head_snapshot();
    }
    head_time_since_update_ += 1;

    cv::Rect2f vec_out = Utils::convert_z_to_bbox(head_kf_->x);
    head_pred_box_ = (cv::Mat_<float>(1, 4) <<
        vec_out.x, vec_out.y, vec_out.width, vec_out.height);
    if (!valid_bbox_xyxy(head_pred_box_)) {
        record_numerical_event("NAN head stage=decode");
        head_pred_box_ = cv::Mat();
        return cv::Mat();
    }
    return head_pred_box_.clone();
}

void KalmanBoxTracker::update_head(std::optional<cv::Mat> head_bbox) {
    if (head_bbox.has_value() && !head_bbox.value().empty()) {
        cv::Mat hb = head_bbox.value();
        if (!valid_bbox_xyxy(hb)) {
            record_numerical_event("REJECT head stage=update reason=invalid_bbox");
            return;
        }
        if (!head_valid_ || !head_kf_) {
            // 首次观测：懒初始化头部轨迹
            head_kf_ = build_head_kf(hb);
            if (!head_kf_) return;
            head_valid_ = true;
            head_last_obs_ = hb.clone();
            head_time_since_update_ = 0;
            head_pred_box_ = hb.clone();
            save_head_snapshot();
            return;
        }

        bool updated = false;
        if (long_coast_) {
            rebase_head(hb);
            updated = head_filter_healthy();
            if (updated) {
                record_numerical_event(
                    "RESET head action=reinitialize reason=strong_head_after_long_coast");
            }
        } else if (!head_filter_healthy()) {
            record_numerical_event("NAN head stage=pre_update");
        } else {
            configure_measurement_noise(*head_kf_, hb, true);
            cv::Mat z = Utils::convert_bbox_to_z(hb);
            updated = head_kf_->update(z)
                && stabilize_covariance(*head_kf_) && head_filter_healthy();
        }
        if (!updated) {
            // 真实头检测比旧预测更可信，直接清除旧中心/速度/P。
            rebase_head(hb);
            record_numerical_event(
                "RESET head action=reinitialize reason=strong_head_update_failed");
        }
        head_last_obs_ = hb.clone();
        head_time_since_update_ = 0;
        head_pred_box_ = hb.clone();
        save_head_snapshot();
    } else {
        // 无观测：纯预测推进（仅在已建立轨迹时）
        if (head_valid_ && head_kf_ && !head_kf_->update(cv::Mat())) {
            record_numerical_event("NAN head stage=empty_update");
            (void)restore_head_snapshot("empty_update");
        }
    }
}

void KalmanBoxTracker::rebase_head(const cv::Mat& head_bbox) {
    if (!valid_bbox_xyxy(head_bbox)) return;

    head_kf_ = build_head_kf(head_bbox);
    if (!head_kf_) return;
    head_valid_ = true;
    head_last_obs_ = head_bbox.clone();
    head_pred_box_ = head_bbox.clone();
    head_time_since_update_ = 0;
    save_head_snapshot();
}

void KalmanBoxTracker::clear_head_track() {
    head_kf_.reset();
    head_last_obs_ = cv::Mat();
    head_pred_box_ = cv::Mat();
    head_time_since_update_ = 0;
    head_valid_ = false;
    head_good_x_ = cv::Mat();
    head_good_P_ = cv::Mat();
}

cv::Mat KalmanBoxTracker::get_head_state() {
    if (!head_filter_healthy()) return cv::Mat();
    cv::Rect2f vec_out = Utils::convert_z_to_bbox(head_kf_->x);
    cv::Mat out = (cv::Mat_<float>(1, 4) <<
        vec_out.x, vec_out.y, vec_out.width, vec_out.height);
    return valid_bbox_xyxy(out) ? out : cv::Mat();
}

// GMC 状态级补偿（见 .h 注释）。状态布局 [cx,cy,w,h,vx,vy]，列向量。
//   位置按完整仿射变换；尺寸按相似变换的尺度分量缩放；速度只受线性部分
//   作用（平移不改变速度）。协方差也必须变换，否则 GMC 后量测增益仍停在旧坐标系。
void KalmanBoxTracker::apply_camera_motion(const cv::Mat& M) {
    if (M.empty()) return;
    if (M.rows < 2 || M.cols < 3
        || (M.type() != CV_32F && M.type() != CV_64F)
        || !cv::checkRange(M, true, nullptr, -DBL_MAX, DBL_MAX)) {
        record_numerical_event("REJECT gmc reason=invalid_matrix");
        return;
    }
    auto m = [&](int r, int c) -> double {
        return M.type() == CV_64F ? M.at<double>(r, c) : (double)M.at<float>(r, c);
    };
    const double a = m(0, 0), b = m(0, 1), tx = m(0, 2);
    const double c = m(1, 0), d = m(1, 1), ty = m(1, 2);
    const float  s = (float)std::sqrt(a * a + b * b);   // 相似变换的尺度分量
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)
        || !std::isfinite(d) || !std::isfinite(tx) || !std::isfinite(ty)
        || !std::isfinite(s) || s < 1.0e-4f || s > 10.f) {
        record_numerical_event("REJECT gmc reason=nonfinite_or_degenerate");
        return;
    }

    if (body_filter_healthy()) save_body_snapshot();
    if (head_filter_healthy()) save_head_snapshot();

    auto warp_state = [&](cv::Mat& X) {
        if (X.empty() || X.rows < 4 || X.cols < 1) return;
        float cx = X.at<float>(0, 0), cy = X.at<float>(1, 0);
        X.at<float>(0, 0) = (float)(a * cx + b * cy + tx);
        X.at<float>(1, 0) = (float)(c * cx + d * cy + ty);
        X.at<float>(2, 0) *= s;     // w
        X.at<float>(3, 0) *= s;     // h
        if (X.rows >= 6) {          // 速度分量：仅线性部分（旋转/缩放）
            float vx = X.at<float>(4, 0), vy = X.at<float>(5, 0);
            X.at<float>(4, 0) = (float)(a * vx + b * vy);
            X.at<float>(5, 0) = (float)(c * vx + d * vy);
        }
    };

    auto warp_covariance = [&](cv::Mat& P) {
        if (P.empty() || P.rows < 6 || P.cols < 6) return;
        cv::Mat T = cv::Mat::eye(6, 6, CV_32F);
        T.at<float>(0, 0) = (float)a; T.at<float>(0, 1) = (float)b;
        T.at<float>(1, 0) = (float)c; T.at<float>(1, 1) = (float)d;
        T.at<float>(2, 2) = s;        T.at<float>(3, 3) = s;
        T.at<float>(4, 4) = (float)a; T.at<float>(4, 5) = (float)b;
        T.at<float>(5, 4) = (float)c; T.at<float>(5, 5) = (float)d;
        P = T * P * T.t();
        P = 0.5f * (P + P.t());
    };

    if (kf) {
        warp_state(kf->x);
        warp_covariance(kf->P);
        if (!stabilize_covariance(*kf) || !body_filter_healthy()) {
            record_numerical_event("NAN body stage=gmc");
            (void)restore_body_snapshot("gmc");
        } else {
            save_body_snapshot();
        }
    }
    if (head_valid_ && head_kf_) {
        warp_state(head_kf_->x);
        warp_covariance(head_kf_->P);
        if (!stabilize_covariance(*head_kf_) || !head_filter_healthy()) {
            record_numerical_event("NAN head stage=gmc");
            if (!restore_head_snapshot("gmc"))
                head_pred_box_ = cv::Mat();
        } else {
            save_head_snapshot();
        }
    }
}
