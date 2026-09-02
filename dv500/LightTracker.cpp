#include "LightTracker.h"
#include "ModelProfiler.h"
#include "Detector.h"
#include "PersonReID.h"
#include "Facekps.h"
#include "FaceRecognition.h"
#include "utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <chrono>
#include <optional>
#include <ostream>
#include <cstdlib>
#include <thread>

#ifndef FX_TRACKER_LIFECYCLE_TEST_HOOKS
#define FX_TRACKER_LIFECYCLE_TEST_HOOKS 0
#endif

// ── 关闭 LightTracker 的所有控制台输出（null_sink / printf → 无操作 sink）──
//   嵌入式上 UART/终端 I/O 是实打实的耗时。[N9] verbose 改 static constexpr，
//   `if (verbose)` 变编译期 `if (false)` → 整段被优化掉，兑现零运行时开销。
namespace {

bool fail_model_init_for_test(const char* stage)
{
#if FX_TRACKER_LIFECYCLE_TEST_HOOKS
    const char* requested = std::getenv("FX_TRACKER_TEST_FAIL_INIT");
    return requested != nullptr && stage != nullptr
        && std::strcmp(requested, stage) == 0;
#else
    (void)stage;
    return false;
#endif
}

struct NullSink {
    static constexpr bool verbose = false;   // [N9] 编译期常量 → 分支被消除，真正零开销

    template <class T>
    NullSink& operator<<(const T& val) {
        if (verbose)
            std::cout << val;
        return *this;
    }

    NullSink& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (verbose)
            manip(std::cout);
        return *this;
    }
};

static NullSink null_sink;

// ── [MATCH] 聚焦决策摘要开关（live-on-device 定位用）──
//   [N9] 不再每帧直接 std::cout（115200 波特阻塞 ≈ 15~17ms/帧，FPS 预算第一梯队）。
//   改为每帧写入 64KB 缓冲文件；事件或心跳才 fflush，完整保留静默错锁前后的
//   CAND/MATCH/OUTPUT。生产默认关闭；诊断构建加 -DFX_TRACKER_MATCH_TRACE=1。
#ifndef FX_TRACKER_MATCH_TRACE
#define FX_TRACKER_MATCH_TRACE 0
#endif
static constexpr bool kMatchTrace = FX_TRACKER_MATCH_TRACE != 0;
// DV500 的 SVP ACL runtime 初始化完成后给驱动预留稳定时间。
// 该平台原实现值为 3000 ms；SS927 不需要这段平台延时。
static constexpr int64_t kInitDelayMs = 3000;

// trace 默认写入 /tmp（通常是 tmpfs），避免 UART 阻塞和持久闪存写放大。
// 运行时可用环境变量 FX_TRACKER_MATCH_TRACE_FILE 改路径；首次写入会截断旧文件。
#ifndef FX_TRACKER_MATCH_TRACE_FILE
#define FX_TRACKER_MATCH_TRACE_FILE "/tmp/fx_tracker_match_trace.log"
#endif
#ifndef FX_TRACKER_MATCH_TRACE_MAX_BYTES
#define FX_TRACKER_MATCH_TRACE_MAX_BYTES (16u * 1024u * 1024u)
#endif

// kMatchTrace 的文件 sink：
//   - 文件只打开一次；每条 trace 仅写入 64KB stdio 用户态缓冲；
//   - trace_flush 才 fflush（事件帧或每 25 帧心跳），不调用 fsync；
//   - 达到 16MB 后停止记录，防止长时间诊断耗尽 /tmp；
//   - 打开失败不回退 UART（只报一次 stderr），避免“为了诊断反而制造卡顿”。
class MatchTraceFileSink {
public:
    ~MatchTraceFileSink() {
        if (fp_) {
            std::fflush(fp_);
            std::fclose(fp_);
        }
    }

    void append_line(const char* line) {
        if (!line || disabled_ || !ensure_open()) return;
        const std::size_t len = std::strlen(line);
        if (!reserve(len + 1)) return;
        if (std::fwrite(line, 1, len, fp_) != len || std::fputc('\n', fp_) == EOF) {
            disable_io("write failed");
            return;
        }
        bytes_ += len + 1;
    }

    void flush() {
        if (!fp_ || disabled_) return;
        if (std::fflush(fp_) != 0) disable_io("flush failed");
    }

private:
    bool ensure_open() {
        if (fp_) return true;
        if (open_attempted_ || disabled_) return false;
        open_attempted_ = true;

        const char* env_path = std::getenv("FX_TRACKER_MATCH_TRACE_FILE");
        path_ = (env_path && env_path[0] != '\0')
              ? env_path : FX_TRACKER_MATCH_TRACE_FILE;
        fp_ = std::fopen(path_.c_str(), "w");
        if (!fp_) {
            disabled_ = true;
            std::fprintf(stderr,
                "[MATCH_TRACE] cannot open %s; file trace disabled\n", path_.c_str());
            return false;
        }

        // 大缓冲把逐行 fwrite 变成内存复制；真正系统写入只在缓冲满或 trace_flush 时发生。
        (void)std::setvbuf(fp_, nullptr, _IOFBF, 64u * 1024u);
        char header[256];
        std::snprintf(header, sizeof(header),
            "==== [TRACE FILE START path=%s max=%u] ====",
            path_.c_str(), (unsigned)FX_TRACKER_MATCH_TRACE_MAX_BYTES);
        append_line(header);
        return !disabled_;
    }

    bool reserve(std::size_t need) {
        const std::size_t cap = (std::size_t)FX_TRACKER_MATCH_TRACE_MAX_BYTES;
        if (bytes_ + need <= cap) return true;

        static constexpr const char* marker =
            "==== [TRACE FILE STOP: size limit reached] ====";
        const std::size_t marker_len = std::strlen(marker);
        if (fp_ && bytes_ + marker_len + 1 <= cap) {
            (void)std::fwrite(marker, 1, marker_len, fp_);
            (void)std::fputc('\n', fp_);
            bytes_ += marker_len + 1;
        }
        // 即使剩余空间容不下 stop marker，也要把此前缓冲内容落到内核页缓存。
        if (fp_) (void)std::fflush(fp_);
        disabled_ = true;
        std::fprintf(stderr,
            "[MATCH_TRACE] %s reached %u bytes; trace stopped\n",
            path_.c_str(), (unsigned)FX_TRACKER_MATCH_TRACE_MAX_BYTES);
        return false;
    }

    void disable_io(const char* why) {
        disabled_ = true;
        std::fprintf(stderr, "[MATCH_TRACE] %s: %s; trace stopped\n",
                     path_.c_str(), why);
    }

    FILE* fp_ = nullptr;
    std::string path_;
    std::size_t bytes_ = 0;
    bool open_attempted_ = false;
    bool disabled_ = false;
};

MatchTraceFileSink& match_trace_file_sink() {
    static MatchTraceFileSink sink;
    return sink;
}

} // namespace
#define printf(...) ((void)0)

const int INF = 999;

LightTracker::LightTracker(const LightTrackerConfig& config)
    : det_thresh(config.det_thresh),
    appearance_thresh(config.appearance_thresh),
    max_age(config.max_age),
    min_hits(config.min_hits),
    iou_threshold(config.iou_threshold),
    delta_t(config.delta_t),
    inertia(config.inertia),
    gate(config.gate),
    frame_count(0),
    main_track_unmatched_time(0),
    img_h(0),
    img_w(0)
{
}

LightTracker::~LightTracker()
{
    // 析构函数体执行时 ACL runtime 仍有效；成员随后析构时 release() 为幂等 no-op。
    release_models();
}

void LightTracker::release_models()
{
    initialized_ = false;
    // DV500 实际初始化顺序：Detector -> ReID -> FaceKps -> FaceReco -> Pose。
    // 这里严格逆序释放；各 wrapper 的 destroy/release 均为幂等。
    pose_estimator.destroy();
    face_recognizer.release();
    detector_fk.destroy();
    emb_model.destroy();
    detector.destroy();
}

int LightTracker::init() {

    if (initialized_) return 0;
    if (svp_runtime_.init(0) != 0) {
        ERROR_LOG("LightTracker SVP ACL runtime initialization failed");
        return -1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kInitDelayMs));

    std::string model_dir = "/oem/model";
    std::string reid_model_path = model_dir + "/mobilev2_dsr_reid_v4.om";
    std::string face_rec_model_path = model_dir + "/faceReco.om";
    std::string det_model_path = model_dir + "/yolox_416_face_body_head.om";
    std::string fkps_model_path = model_dir + "/facekps.om";
    std::string pose_est_model_path = model_dir + "/rtmpose-t.om";


    /**
    * Person and Face detector init
    */
    DetectorConfig config_det;
    config_det.model_path = det_model_path;
    // YOLOX OM/AIPP 契约与 SS927 一致：OpenCV BGR、HWC U8、416x416。
    config_det.bgr2rgb = false;
    config_det.num_anchor = 3549;
    config_det.dim_anchor = 8;
    int retry = 0;
    while (retry < 10) {
        if(detector.init(config_det) != 0) {
            null_sink << "Failed to initialize detector" << std::endl;
            retry++;
            sleep(1);  
        }
        else{
            break; 
        }
    }
    if(retry >= 10) {
        release_models();
        svp_runtime_.release();
        return -1;
    }
    /**
    * Reid model init
    */
    retry = 0;
    PCBReIDConfig config_reid_pcb;
    config_reid_pcb.model_path = reid_model_path;
    while (retry < 10) {
        if(emb_model.init(config_reid_pcb) != 0) {
            null_sink << "Failed to initialize reid model" << std::endl;
            retry++;
            sleep(1);  // 等 1 秒再试
        }
        else{
            break;  // 成功初始化，跳出循环
        }
    }
    if(retry >= 10) {
        release_models();
        svp_runtime_.release();
        return -1;
    }

    /**
    * Face keypoint predict init
    */
    FaceKeypointConfig config_fk;
    config_fk.model_path = fkps_model_path;
    retry = 0;
    while (retry < 10) {
        if(detector_fk.init(config_fk) != 0) {
            null_sink << "Failed to initialize face keypoint model" << std::endl;
            retry++;
            sleep(1);  
        }
        else{
            break;  
        }
    }
    if(retry >= 10) {
        release_models();
        svp_runtime_.release();
        return -1;
    }

    /**
    * Face recognition init
    */
    FaceRecogConfigDV500 config_fr;
    config_fr.model_path = face_rec_model_path;
    retry = 0;
    while (retry < 10) {
        if(face_recognizer.init(config_fr, 0.50f, 0.40f) != 0) {
            null_sink << "Failed to initialize face recognition model" << std::endl;
            retry++;
            sleep(1);
        }
        else {
            break; 
        }
    }
    if(retry >= 10) {
        release_models();
        svp_runtime_.release();
        return -1;
    }


    PoseEstimatorConfig config_pose;
    config_pose.model_path = pose_est_model_path;
    config_pose.kp_conf_thresh = 0.40f;
    retry = 0;
    while (retry < 10) {
        if(pose_estimator.init(config_pose) != 0) {
            null_sink << "Failed to initialize pose estimation model" << std::endl;
            retry++;
            sleep(1);  
        }
        else {
            break;  
        }
    }
    if(retry >= 10) {
        release_models();
        svp_runtime_.release();
        return -1;
    }

    initialized_ = true;
    INFO_LOG("[LightTracker] all models initialized");
    return 0;
}


std::pair<cv::Mat, int> LightTracker::setMainTarget(const cv::Mat& img, const cv::Rect& mainBox) {
    printf("=========[f 20] in setMainTarget ======== \n");

    reset();

    // App 指定框本身就是本帧的真实 BODY 观测。reset 后显式恢复输出语义，避免
    // “空闲 reset 状态”和“刚完成注册状态”共用同一个默认控制权重。
    frame_output_source_ = OutputSource::BODY;
    coast_weight_ = 1.0f;

    cv::Mat box = (cv::Mat_<float>(1, 5) << mainBox.x, mainBox.y, mainBox.width, mainBox.height, 1);
    // 人脸-only 重构依赖最近真实体框尺寸。注册帧本身就是 App 指定的真实主目标框，
    // 必须在 reset() 后立即建立尺寸基准；否则刚注册后身体即丢检、只露脸时无法重构。
    last_main_bw_ = (float)(mainBox.width  - mainBox.x);
    last_main_bh_ = (float)(mainBox.height - mainBox.y);
    if (last_main_bw_ <= 1.f || last_main_bh_ <= 1.f) {
        last_main_bw_ = -1.f;
        last_main_bh_ = -1.f;
    }
    last_real_obs_ms_ = now_ms();
    last_body_observation_ms_ = last_real_obs_ms_;
    update_lead_center(box.colRange(0, 4));
    update_body_reid_search_anchor(box.colRange(0, 4));
    update_ptz_blind_anchor(box.colRange(0, 4), last_body_observation_ms_);
    note_returned_box(box.colRange(0, 4));

    cv::Mat main_embs = compute_embedding(img, box);
    auto main_trk = std::make_shared<KalmanBoxTracker>(box, delta_t, main_embs, true);
    main_trk->set_anchor_emb(main_embs);
    main_trk->set_confirmed_emb(main_embs);
    trackers.push_back(main_trk);
    main_id = main_trk->get_id();
    frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
    record_motion_observation(box.colRange(0, 4),
                              MotionObservationSource::BODY,
                              current_frame_timestamp_ms_ >= 0
                                  ? current_frame_timestamp_ms_
                                  : last_body_observation_ms_);

    // 注册框已由上层确认；Pose 只刷新 committed 状态，不参与本次身份选择。
    {
        cv::Mat bbox_for_match = box.colRange(0, 4);
        PoseCacheEntry* entry = request_pose(
            img, bbox_for_match, -1, -1, PoseReason::TARGET_REFRESH);
        if (entry && entry->status == PoseRequestStatus::AVAILABLE) {
            commit_pose(entry->pose, bbox_for_match, img, "registration");
            BodyProportionDescriptor bp = PoseMatch::extract_body_proportions(
                entry->pose.keypoints, 0.4f);
            if (bp.valid) {
                main_trk->set_body_proportions(bp);
                main_trk->set_anchor_body_proportions(bp);
                null_sink << "[POSE] anchor body proportions registered" << std::endl;
            }
        }
    }

    std::vector<ObjDetInfo> det_result;
    cv::Rect face_box;
    std::vector<float> kps;

    cv::Rect bbox_Region(mainBox.x, mainBox.y, mainBox.width - mainBox.x, mainBox.height - mainBox.y);
    cv::Rect region = bbox_Region & cv::Rect(0, 0, img.cols, img.rows);
    cv::Mat rect_img = img(region).clone();

    if (detector.run(rect_img, 0.3, det_result) != 0) {
        reset();
        throw TrackerRuntimeError("Detector failed while registering main target");
    }
    // A3 修复：多脸场景不再"最后一个赢"。头取最高分；脸须与头/上部几何一致，一致者取最高分；
    // 存在第二张可比(≥0.8×最高)的一致脸 → 歧义，放弃注册（延迟注册通道有身份安全闸补上）。
    int faceCNT = 0;
    cv::Rect head_box_crop;
    int headCNT = 0;
    {
        float head_best = -1.f;
        std::vector<cv::Rect> cf;  std::vector<float> cfs;
        for (int i = 0; i < det_result.size(); i++) {
            if (det_result[i].label == LABEL_FACE) {
                cf.push_back(det_result[i].box);
                cfs.push_back(det_result[i].score);
            } else if (det_result[i].label == LABEL_HEAD) {
                if (det_result[i].score > head_best) {
                    head_best = det_result[i].score;
                    head_box_crop = det_result[i].box;
                }
                headCNT++;
            }
        }
        int   best_i = -1, second_i = -1;
        float best_s = -1.f, second_s = -1.f;
        for (int i = 0; i < (int)cf.size(); ++i) {
            float fcx = (cf[i].x + cf[i].width)  * 0.5f;
            float fcy = (cf[i].y + cf[i].height) * 0.5f;
            bool ok;
            if (headCNT >= 1) {
                ok = fcx >= head_box_crop.x && fcx <= head_box_crop.width
                  && fcy >= head_box_crop.y && fcy <= head_box_crop.height;
            } else {
                ok = fcy <= 0.45f * rect_img.rows;
            }
            if (!ok) continue;
            if (cfs[i] > best_s)        { second_s = best_s; second_i = best_i;
                                          best_s = cfs[i];   best_i = i; }
            else if (cfs[i] > second_s) { second_s = cfs[i]; second_i = i; }
        }
        (void)second_i;
        if (best_i >= 0 && (second_s < 0.f || second_s < 0.8f * best_s)) {
            face_box = cf[best_i];
            faceCNT  = 1;
        } else {
            faceCNT = 0;
            if (best_i >= 0)
                null_sink << "[FACE_REG] ambiguous faces at designation ("
                             << cf.size() << " faces), defer registration" << std::endl;
        }
    }

    // ── 初始化主目标头部轨迹（坐标从裁剪图还原到全图，xyxy 语义）──
    if (headCNT >= 1) {
        cv::Mat head_full = (cv::Mat_<float>(1, 4) <<
            (float)(head_box_crop.x      + region.x),
            (float)(head_box_crop.y      + region.y),
            (float)(head_box_crop.width  + region.x),
            (float)(head_box_crop.height + region.y));
        main_trk->update_head(head_full);
        // 指定框是用户确认的主目标可见框；立即建立头→体几何，避免注册后下一帧
        // 人体漏检时虽有头轨迹却无法重构。后续只允许干净完整帧慢速修正该比例。
        learn_head_body_geom(box.colRange(0, 4), head_full);
        if (!head_body_geom_valid_) {
            // 指定框若本身只含肩膀/头部，比例护栏会拒绝把它当完整人体；此时启用
            // reset 后的保守默认完整比例，随后照常做画面求交，仍不会拉出越界返回框。
            head_body_geom_valid_ = true;
        }
        null_sink << "[HEAD] anchor head track initialized at register" << std::endl;
    } else {
        // 指定主目标框本身是可信真值。若注册帧的头检测偶发漏检，仍建立一个保守的
        // 上部头先验并学习其与身体的几何关系；否则目标下一帧躲到椅子后只露头时，
        // head-only 路径会因“没有 head KF/几何先验”完全不可用。
        float bx1 = box.at<float>(0, 0), by1 = box.at<float>(0, 1);
        float bx2 = box.at<float>(0, 2), by2 = box.at<float>(0, 3);
        float bw = bx2 - bx1, bh = by2 - by1;
        if (bw > 4.f && bh > 8.f) {
            float hcx = (bx1 + bx2) * 0.5f;
            float hcy = by1 + bh * 0.14f;
            float hw = bw * 0.40f, hh = bh * 0.20f;
            cv::Mat head_prior = (cv::Mat_<float>(1, 4) <<
                hcx - hw * 0.5f, hcy - hh * 0.5f,
                hcx + hw * 0.5f, hcy + hh * 0.5f);
            main_trk->update_head(head_prior);
            learn_head_body_geom(box.colRange(0, 4), head_prior);
            null_sink << "[HEAD] register head miss: initialized geometric head prior" << std::endl;
        }
    }

    // 指定帧若人体完整落在画面内且唯一脸归属明确，立即建立脸→完整人体比例。
    // 这样注册后马上只露脸时也能按当前脸尺度重构；触边人体不学习，避免近距裁剪污染。
    if (faceCNT == 1
        && mainBox.x > 3 && mainBox.y > 3
        && mainBox.width < img.cols - 4 && mainBox.height < img.rows - 4) {
        cv::Mat face_full = (cv::Mat_<float>(1, 4) <<
            (float)(face_box.x + region.x),
            (float)(face_box.y + region.y),
            (float)(face_box.width + region.x),
            (float)(face_box.height + region.y));
        const float face_h = face_full.at<float>(0, 3)
                           - face_full.at<float>(0, 1);
        const float body_w = (float)(mainBox.width - mainBox.x);
        const float body_h = (float)(mainBox.height - mainBox.y);
        // 注册框可能本身就是椅子后的半身框。只有比例看起来像完整人体时才把
        // 它写成“脸→完整人体”尺度；否则等待后续干净帧学习，避免先验变短。
        if (face_h > 1.f && body_w > 1.f
            && body_h / face_h >= 4.5f && body_h / body_w >= 1.45f) {
            learn_face_body_geom(box.colRange(0, 4), face_full);
        }
    }

    if (faceCNT == 0) {
        return {box, 1};
    }

    if (!take_face_model_slot())
        return { box, 1 };
    FaceKeypointResult fk_result = detector_fk.run(rect_img, face_box);
    std::vector<float> kps_10 = get_kps10(fk_result.points);
    if (kps_10.size() < 10)   // A10：关键点失败 → 不注册（延迟注册通道稍后补上）
        return { box, 1 };

    // [N6] 初始注册质量门：复用延迟注册的质量评分（正脸/尺寸/完整性）+ 最小脸尺寸。
    //   侧脸/小脸初注册会拖垮冷启动期识别率（恰是最需要人脸兜底的阶段）→ 质量不达标
    //   本次不注册，交由延迟注册通道（try_deferred_face_register）在稳定跟踪后补上。
    //   face_box 为 xyxy 塞 Rect：脸高 = height(=y2) − y。
    float init_face_h = (float)(face_box.height - face_box.y);
    float init_q = evaluate_face_quality(kps_10, face_box, rect_img.cols, rect_img.rows);
    if (init_face_h < kFaceRegisterMinFacePx || init_q < kFaceInitRegisterMinQ) {
        null_sink << "[FACE_REG] initial defer: face_h=" << init_face_h
                     << " q=" << init_q << " (thr px=" << kFaceRegisterMinFacePx
                     << " q=" << kFaceInitRegisterMinQ << ")" << std::endl;
        return { box, 1 };
    }

    std::string name = face_recognizer.register_face(rect_img, kps_10, "bro");
    if (name != "success" || !face_recognizer.has_face_template()) {
        face_registered_ = face_recognizer.has_face_template();
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_REG] f=%d result=reject mode=initial reason=%s template=%d",
                frame_count, name.c_str(), face_registered_ ? 1 : 0);
            trace_push(line);
            trace_event_pending_ = true;
        }
        return { box, 1 };
    }
    face_registered_          = true;
    face_template_quality_    = init_q;
    last_face_register_ms_    = now_ms();
    last_confirmed_face_box_ = cv::Rect(
        face_box.x + region.x, face_box.y + region.y,
        face_box.width + region.x, face_box.height + region.y);
    last_confirmed_face_ms_ = last_face_register_ms_;
    last_confirmed_face_frame_ = frame_count;
    // 用户指定框本身是真值；该框内完成的初始注册等价于 BODY+FACE 身份确认，
    // 避免刚开始健康跟踪时因 body-face 时间戳为空而额外逐帧全局扫脸。
    last_face_identity_ms_ = last_face_register_ms_;
    last_body_face_identity_ms_ = last_face_register_ms_;
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_REG] f=%d result=accept mode=initial q=%.2f",
            frame_count, init_q);
        trace_push(line);
    }
    null_sink << "[FACE_REG] initial registered at designation (q=" << init_q << ")" << std::endl;
    return { box, 1};
}

int64_t LightTracker::now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// [N1] const 版（供 get_blind_ms 等 const 成员），实现同 now_ms。
int64_t LightTracker::now_ms_const() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

cv::Mat LightTracker::build_main_row(const cv::Mat& box4)
{
    cv::Mat out(1, 5, CV_32F);
    for (int c = 0; c < 4; c++)
        out.at<float>(0, c) = box4.at<float>(0, c);
    out.at<float>(0, 4) = 1.f;
    return out;
}

cv::Mat LightTracker::clip_reconstructed_body_to_frame(const cv::Mat& box4,
                                                        const char* source_tag)
{
    if (box4.empty() || box4.cols < 4 || img_w < 2 || img_h < 2)
        return cv::Mat();

    const float ox1 = box4.at<float>(0, 0);
    const float oy1 = box4.at<float>(0, 1);
    const float ox2 = box4.at<float>(0, 2);
    const float oy2 = box4.at<float>(0, 3);
    if (!std::isfinite(ox1) || !std::isfinite(oy1)
        || !std::isfinite(ox2) || !std::isfinite(oy2)
        || ox2 <= ox1 + 1.f || oy2 <= oy1 + 1.f) {
        return cv::Mat();
    }

    // 真正做“重构框 ∩ 画面”。不能逐坐标强夹出一个 1px 假框：若重构框
    // 已完全跑出画面，应该拒绝该观测，而不是让 KF/云台吸向画面边缘。
    const float frame_x2 = (float)img_w - 1.f;
    const float frame_y2 = (float)img_h - 1.f;
    if (ox2 <= 0.f || oy2 <= 0.f || ox1 >= frame_x2 || oy1 >= frame_y2)
        return cv::Mat();
    const float x1 = std::max(0.f, ox1);
    const float y1 = std::max(0.f, oy1);
    const float x2 = std::min(frame_x2, ox2);
    const float y2 = std::min(frame_y2, oy2);
    if (x2 <= x1 + 1.f || y2 <= y1 + 1.f)
        return cv::Mat();

    cv::Mat clipped = (cv::Mat_<float>(1, 4) << x1, y1, x2, y2);
    bool changed = std::fabs(x1 - ox1) > 0.5f || std::fabs(y1 - oy1) > 0.5f
                || std::fabs(x2 - ox2) > 0.5f || std::fabs(y2 - oy2) > 0.5f;
    if (changed && kMatchTrace) {
        const int clip_mask = (ox1 < 0.f ? 1 : 0)
                            | (oy1 < 0.f ? 2 : 0)
                            | (ox2 > (float)img_w - 1.f ? 4 : 0)
                            | (oy2 > (float)img_h - 1.f ? 8 : 0);
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[PART_RECON] f=%d src=%s clip=1 mask=0x%X raw=(%.0f,%.0f,%.0f,%.0f)"
            " clipped=(%.0f,%.0f,%.0f,%.0f)",
            frame_count, source_tag ? source_tag : "?",
            clip_mask, ox1, oy1, ox2, oy2, x1, y1, x2, y2);
        trace_push(line);
    }
    return clipped;
}

cv::Mat LightTracker::stabilize_returned_box(const cv::Mat& raw_return_box,
                                              OutputSource source,
                                              bool force_snap)
{
    if (raw_return_box.empty() || raw_return_box.cols < 4)
        return cv::Mat();

    const float obs_x1 = raw_return_box.at<float>(0, 0);
    const float obs_y1 = raw_return_box.at<float>(0, 1);
    const float obs_x2 = raw_return_box.at<float>(0, 2);
    const float obs_y2 = raw_return_box.at<float>(0, 3);
    float x1 = obs_x1, y1 = obs_y1, x2 = obs_x2, y2 = obs_y2;
    float raw_cx = (x1 + x2) * 0.5f;
    float raw_cy = (y1 + y2) * 0.5f;
    float raw_w = x2 - x1, raw_h = y2 - y1;
    if (!std::isfinite(raw_cx) || !std::isfinite(raw_cy)
        || !std::isfinite(raw_w) || !std::isfinite(raw_h)
        || raw_w <= 1.f || raw_h <= 1.f) return cv::Mat();

    const int64_t now = now_ms();
    const bool is_part = source == OutputSource::HEAD
                      || source == OutputSource::FACE;
    const bool recent_output = last_returned_box_ms_ >= 0
                            && now - last_returned_box_ms_
                               <= kPartBoxBridgeMaxGapMs;
    const bool stale = part_output_update_ms_ < 0
                    || now - part_output_update_ms_ > kPartBoxBridgeMaxGapMs;

    // ── 近距离底部截断框归一化 ──
    // 近场只露肩膀以上时，BODY/HEAD/FACE 的检测尺寸会随正脸↔侧脸明显变化。
    // 这种尺寸变化不是人体纵向运动：以新鲜头中心作为纵向锚点，框底统一到画面
    // 底边。转头只改变部件形状，不再把整框中心先向下拉、随后又向上抬。
    bool head_fresh = false;
    int head_tsu = -1;
    float head_cy = -1.f, head_h = 0.f;
    for (const auto& t : trackers) {
        if (!t->get_is_main() || !t->has_head_track()) continue;
        head_tsu = t->get_head_time_since_update();
        if (head_tsu > kCloseupControlHeadMaxTsu) break;
        cv::Mat hp = t->get_head_pred_box();
        if (hp.empty() || hp.cols < 4) break;
        const float hy1 = hp.at<float>(0, 1);
        const float hy2 = hp.at<float>(0, 3);
        if (std::isfinite(hy1) && std::isfinite(hy2) && hy2 > hy1 + 1.f) {
            head_cy = (hy1 + hy2) * 0.5f;
            head_h = hy2 - hy1;
            head_fresh = true;
        }
        break;
    }

    // 没有真实头框时，已确认人脸仍可给出保守头顶：人脸检测框通常不包含完整
    // 额头/头发，因此从 face_y1 再向上补约 1/4 脸高。它只作为 head 缺失回退。
    bool face_fresh = false;
    float face_cy = -1.f, face_h = 0.f, face_head_top = -1.f;
    const int face_frame_age = last_confirmed_face_frame_ >= 0
                             ? frame_count - last_confirmed_face_frame_ : -1;
    if (last_confirmed_face_frame_ == frame_count
        && last_confirmed_face_ms_ >= 0
        && last_confirmed_face_box_.width > last_confirmed_face_box_.x
        && last_confirmed_face_box_.height > last_confirmed_face_box_.y) {
        const float fy1 = (float)last_confirmed_face_box_.y;
        const float fy2 = (float)last_confirmed_face_box_.height;
        face_h = fy2 - fy1;
        if (face_h > 1.f) {
            face_cy = (fy1 + fy2) * 0.5f;
            face_head_top = fy1 - kFaceToHeadTopFrac * face_h;
            face_fresh = std::isfinite(face_cy) && std::isfinite(face_head_top);
        }
    }

    const float frame_bottom = (float)img_h - 1.f;
    const float frame_diag = std::hypot((float)img_w, (float)img_h);
    const float enter_bottom_margin = std::max(4.f,
        kCloseupEnterBottomFrac * (float)img_h);
    const float exit_bottom_margin = std::max(6.f,
        kCloseupExitBottomFrac * (float)img_h);
    const bool near_bottom_enter = y2 >= frame_bottom - enter_bottom_margin;
    const bool near_bottom_keep = y2 >= frame_bottom - exit_bottom_margin;
    const bool close_scale = (head_fresh
                           && head_h >= kCloseupMinHeadHeightFrac * (float)img_h)
                          || raw_w >= kCloseupMinBodyWidthFrac * (float)img_w;
    const bool tall_enough = raw_h >= kCloseupMinVisibleHeightFrac * (float)img_h;
    bool closeup = near_bottom_enter && close_scale && tall_enough;
    if (!closeup && closeup_output_active_ && near_bottom_keep && tall_enough)
        closeup = true;  // 朝向变化导致头/宽度短暂缩水时保持状态，不来回切换

    bool strong_spatial_relocate = false;
    if (recent_output && last_returned_x2_ > last_returned_x1_
        && last_returned_y2_ > last_returned_y1_) {
        const float last_cx = (last_returned_x1_ + last_returned_x2_) * 0.5f;
        const float last_cy = (last_returned_y1_ + last_returned_y2_) * 0.5f;
        strong_spatial_relocate = std::hypot(raw_cx - last_cx, raw_cy - last_cy)
                               > kCloseupRelocateDiagFrac * frame_diag;
    }

    if (closeup) {
        const bool state_stale = closeup_update_ms_ < 0
                              || now - closeup_update_ms_ > kPartBoxBridgeMaxGapMs;
        const bool reinitialize = !closeup_output_active_ || state_stale
                               || (force_snap && strong_spatial_relocate);
        if (reinitialize) {
            const bool bridge_previous = recent_output && !strong_spatial_relocate
                && last_returned_y2_ >= frame_bottom - exit_bottom_margin
                && last_returned_y2_ > last_returned_y1_;
            if (head_fresh) {
                // 近场 y1 的定义就是稳定头顶，不能继承旧人体框的任意顶部。
                closeup_top_y_ = head_cy - 0.5f * head_h;
                closeup_last_head_cy_ = head_cy;
                closeup_last_head_ms_ = now;
                closeup_last_face_cy_ = -1.f;
                closeup_last_face_ms_ = -1;
                closeup_anchor_kind_ = 1;
            } else if (face_fresh) {
                closeup_top_y_ = face_head_top;
                closeup_last_face_cy_ = face_cy;
                closeup_last_face_ms_ = now;
                closeup_last_head_cy_ = -1.f;
                closeup_last_head_ms_ = -1;
                closeup_anchor_kind_ = 2;
            } else {
                closeup_top_y_ = bridge_previous ? last_returned_y1_ : y1;
                closeup_last_head_cy_ = -1.f;
                closeup_last_head_ms_ = -1;
                closeup_last_face_cy_ = -1.f;
                closeup_last_face_ms_ = -1;
                closeup_anchor_kind_ = 0;
            }
            closeup_output_active_ = true;
        } else if (head_fresh) {
            // 以进入近场时的头中心为锚。正/侧脸切换造成的小范围检测中心变化
            // 留在姿态死区内；真实的累计纵向移动超过死区后才整体平移返回框。
            const float head_jitter_db = std::max(4.f,
                std::min(24.f, kCloseupHeadJitterFrac * head_h));
            if (closeup_anchor_kind_ != 1 || closeup_last_head_cy_ < 0.f) {
                // 首次从 BODY/FACE 回到真实头框时，重新校准到头顶；最终输出仍经过
                // 整框限速，不会瞬时跳变。
                closeup_top_y_ = head_cy - 0.5f * head_h;
                closeup_last_head_cy_ = head_cy;
            } else {
                const float head_dy = head_cy - closeup_last_head_cy_;
                if (std::fabs(head_dy) > head_jitter_db) {
                    // 只消费超过姿态死区的位移，留下固定残差。避免越过阈值时一次
                    // 吃掉全部位移，形成“蓄积—跳动—再蓄积”的锯齿运动。
                    const float consumed = head_dy
                        - std::copysign(head_jitter_db, head_dy);
                    closeup_top_y_ += consumed;
                    closeup_last_head_cy_ += consumed;
                }
            }
            closeup_last_head_ms_ = now;
            closeup_last_face_cy_ = -1.f;
            closeup_last_face_ms_ = -1;
            closeup_anchor_kind_ = 1;
        } else if (face_fresh) {
            const float face_jitter_db = std::max(4.f,
                std::min(24.f, kCloseupHeadJitterFrac * face_h));
            const bool recent_head = closeup_last_head_ms_ >= 0
                && now - closeup_last_head_ms_ <= kCloseupHeadHoldMs;
            if (recent_head && closeup_anchor_kind_ == 1) {
                // 当前帧虽有人脸，但刚刚仍有真实头观测：只缓存脸中心，不立即
                // HEAD→FACE 换锚。这样周期性人脸推理不会与逐帧头框轮流拉动 y1。
                closeup_last_face_cy_ = face_cy;
                closeup_last_face_ms_ = now;
            } else if (closeup_anchor_kind_ != 2 || closeup_last_face_cy_ < 0.f) {
                // 已有稳定头/脸顶部时，切换运动锚但保持顶部连续；只有完全没有
                // 可信锚点时才由当前人脸估算头顶。
                if (closeup_anchor_kind_ == 0) closeup_top_y_ = face_head_top;
                closeup_last_face_cy_ = face_cy;
                closeup_last_face_ms_ = now;
                closeup_anchor_kind_ = 2;
            } else {
                const float face_dy = face_cy - closeup_last_face_cy_;
                if (std::fabs(face_dy) > face_jitter_db) {
                    const float consumed = face_dy
                        - std::copysign(face_jitter_db, face_dy);
                    closeup_top_y_ += consumed;
                    closeup_last_face_cy_ += consumed;
                }
                closeup_last_face_ms_ = now;
                closeup_anchor_kind_ = 2;
            }
        } else if ((closeup_last_head_ms_ < 0
                    || now - closeup_last_head_ms_ > kCloseupHeadHoldMs)
                   && (closeup_last_face_ms_ < 0
                       || now - closeup_last_face_ms_ > kCloseupHeadHoldMs)) {
            // 长时间没有头观测时不能永久冻结；把 BODY/重构框顶部交给后续限速器
            // 缓慢跟随。短暂侧脸漏头则保持，跨过检测空档。
            closeup_top_y_ = y1;
            closeup_anchor_kind_ = 0;
            closeup_last_head_cy_ = -1.f;
            closeup_last_face_cy_ = -1.f;
        }
        closeup_top_y_ = std::max(0.f,
            std::min(closeup_top_y_, frame_bottom - 2.f));
        y1 = closeup_top_y_;
        y2 = frame_bottom;
        raw_cx = (x1 + x2) * 0.5f;
        raw_cy = (y1 + y2) * 0.5f;
        raw_w = x2 - x1;
        raw_h = y2 - y1;
        closeup_update_ms_ = now;
    } else {
        closeup_output_active_ = false;
        closeup_top_y_ = -1.f;
        closeup_last_head_cy_ = -1.f;
        closeup_update_ms_ = -1;
        closeup_last_head_ms_ = -1;
        closeup_last_face_cy_ = -1.f;
        closeup_last_face_ms_ = -1;
        closeup_anchor_kind_ = 0;
    }

    // 近距离本地人脸确认不能因为姿态引起的重构尺寸变化触发 snap；真正跨画面
    // 找回仍由 strong_spatial_relocate 保留强制跳转。
    const bool effective_force_snap = force_snap
                                   && (!closeup || strong_spatial_relocate);
    const bool needs_stability = is_part || closeup;
    // 用裁剪后框高的 1/3 作为控制尺度，使完整人体与近距可见部分都获得随景深
    // 自适应的死区/速度；这里只影响输出平滑，不参与身份判断。
    const float scale = std::max(4.f, raw_h / 3.f);
    // 近场顶部已经由头/脸锚做过姿态死区，不再叠加一个大位置死区。
    const float pos_db = closeup ? kPartBoxDeadbandMinPx
        : std::max(kPartBoxDeadbandMinPx,
            std::min(kPartBoxDeadbandMaxPx, kPartBoxPosDeadbandFrac * scale));
    const float w_db = std::max(kPartBoxDeadbandMinPx,
        std::min(kPartBoxDeadbandMaxPx, kPartBoxSizeDeadbandFrac * raw_w));
    const float h_db = std::max(kPartBoxDeadbandMinPx,
        std::min(kPartBoxDeadbandMaxPx, kPartBoxSizeDeadbandFrac * raw_h));
    const char* action = "raw";

    auto set_state = [&](float cx, float cy, float w, float h) {
        part_output_cx_ = cx; part_output_cy_ = cy;
        part_output_w_ = w;   part_output_h_ = h;
    };
    auto close_enough = [&](float cx, float cy, float w, float h,
                            float mul) {
        return std::fabs(cx - raw_cx) <= mul * pos_db
            && std::fabs(cy - raw_cy) <= mul * pos_db
            && std::fabs(w  - raw_w)  <= mul * w_db
            && std::fabs(h  - raw_h)  <= mul * h_db;
    };

    if (!needs_stability && (!part_output_box_valid_ || !recent_output)) {
        part_output_box_valid_ = false;
        part_output_pending_since_ms_ = -1;
    } else if (effective_force_snap || stale || !part_output_box_valid_) {
        const bool bridge_previous = needs_stability && !effective_force_snap
                                  && recent_output
                                  && last_returned_x2_ > last_returned_x1_
                                  && last_returned_y2_ > last_returned_y1_;
        if (bridge_previous) {
            const float prev_cx = (last_returned_x1_ + last_returned_x2_) * 0.5f;
            const float prev_w = last_returned_x2_ - last_returned_x1_;
            if (closeup) {
                // 近场所有来源共享“顶部 + 画面底边”的同一可见框语义。
                const float prev_top = std::max(0.f,
                    std::min(last_returned_y1_, frame_bottom - 2.f));
                set_state(prev_cx, (prev_top + frame_bottom) * 0.5f,
                          prev_w, frame_bottom - prev_top);
                action = "closeup_bridge";
            } else {
                set_state(prev_cx,
                          (last_returned_y1_ + last_returned_y2_) * 0.5f,
                          prev_w, last_returned_y2_ - last_returned_y1_);
                action = "body_to_part_bridge";
            }
        } else {
            set_state(raw_cx, raw_cy, raw_w, raw_h);
            action = effective_force_snap ? "face_relocate_snap"
                   : closeup ? "closeup_init" : "part_init";
        }
        part_output_box_valid_ = needs_stability;
        part_output_pending_since_ms_ = -1;
    } else {
        const float dt0 = part_output_update_ms_ >= 0
                        ? (float)(now - part_output_update_ms_) / 1000.f : 0.04f;
        const float dt = std::max(0.01f, std::min(dt0, 0.20f));
        const float stable_top = part_output_cy_ - part_output_h_ * 0.5f;
        bool within = closeup
            ? std::fabs(part_output_cx_ - raw_cx) <= pos_db
              && std::fabs(stable_top - y1) <= pos_db
              && std::fabs(part_output_w_ - raw_w) <= w_db
            : close_enough(part_output_cx_, part_output_cy_,
                           part_output_w_, part_output_h_, 1.f);

        if (needs_stability && within) {
            part_output_pending_since_ms_ = -1;
            action = closeup ? "closeup_deadband_hold" : "box_deadband_hold";
        } else {
            // 候选只要持续落在稳定框死区外就开始跟随；不要求它停在同一像素位置，
            // 否则真实目标/云台连续运动时 pending 会被每帧重置，表现为“框冻结”。
            if (part_output_pending_since_ms_ < 0) {
                part_output_pending_since_ms_ = now;
            }

            bool may_follow = !needs_stability || closeup
                           || (part_output_pending_since_ms_ >= 0
                               && now - part_output_pending_since_ms_
                                  >= kPartBoxDwellMs);
            if (may_follow) {
                const float speed_mul = needs_stability ? 1.f : kPartBoxBodyReturnSpeedMul;
                const float pos_rate = closeup ? kCloseupPosSpeedScaleSec
                                               : kPartBoxPosSpeedScaleSec * speed_mul;
                const float size_rate = closeup ? kCloseupSizeSpeedScaleSec
                                                : kPartBoxSizeSpeedScaleSec * speed_mul;
                const float pos_step = std::max(2.f,
                    pos_rate * scale * dt);
                const float size_step = std::max(2.f,
                    size_rate * scale * dt);
                auto approach = [](float cur, float dst, float lim) {
                    return cur + std::max(-lim, std::min(dst - cur, lim));
                };
                part_output_cx_ = approach(part_output_cx_, raw_cx, pos_step);
                part_output_w_  = approach(part_output_w_,  raw_w,  size_step);
                if (closeup) {
                    // 底边固定的框只平滑一个自由度：顶部。不能分别平滑 cy/h，
                    // 否则两者速度不同会凭空制造 y1 漂移。
                    const float cur_top = part_output_cy_
                                        - part_output_h_ * 0.5f;
                    const float next_top = approach(cur_top, y1, pos_step);
                    part_output_h_ = frame_bottom - next_top;
                    part_output_cy_ = (next_top + frame_bottom) * 0.5f;
                } else {
                    part_output_cy_ = approach(part_output_cy_, raw_cy, pos_step);
                    part_output_h_  = approach(part_output_h_,  raw_h,  size_step);
                }
                action = closeup ? "closeup_follow"
                       : is_part ? "box_follow" : "body_bridge";
            } else {
                action = "box_dwell_hold";
            }

            if (!needs_stability
                && close_enough(part_output_cx_, part_output_cy_,
                                part_output_w_, part_output_h_, 1.f)) {
                set_state(raw_cx, raw_cy, raw_w, raw_h);
                part_output_box_valid_ = false;
                part_output_pending_since_ms_ = -1;
                action = "body_done";
            }
        }
    }

    part_output_update_ms_ = now;
    if (closeup && part_output_box_valid_) {
        // cx/cy/w/h 的位置、尺寸限速必须服从“底边固定”这个几何约束；否则 cy 与 h
        // 以不同速率靠近时，输出底边会短暂上浮再落下，重新制造一次上下摆动。
        const float stable_top = std::max(0.f, std::min(
            part_output_cy_ - part_output_h_ * 0.5f, frame_bottom - 2.f));
        part_output_h_ = frame_bottom - stable_top;
        part_output_cy_ = (stable_top + frame_bottom) * 0.5f;
    }
    cv::Mat out = (cv::Mat_<float>(1, 4) << x1, y1, x2, y2);
    if (part_output_box_valid_ || std::strcmp(action, "raw") != 0) {
        const float sw = std::max(2.f, part_output_w_);
        const float sh = std::max(2.f, part_output_h_);
        out = (cv::Mat_<float>(1, 4) <<
            part_output_cx_ - sw * 0.5f, part_output_cy_ - sh * 0.5f,
            part_output_cx_ + sw * 0.5f, part_output_cy_ + sh * 0.5f);
        cv::Mat clipped = clip_reconstructed_body_to_frame(out, "OUTPUT_VISIBLE");
        if (!clipped.empty()) out = clipped;
    }

    if (kMatchTrace && (needs_stability || std::strcmp(action, "raw") != 0)) {
        const char* src = source == OutputSource::BODY ? "BODY"
                        : source == OutputSource::HEAD ? "HEAD"
                        : source == OutputSource::FACE ? "FACE" : "NONE";
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[OUTPUT_STABLE] f=%d src=%s action=%s close=%d head=%d/tsu%d face=%d/age%d anc=%d"
            " htop=%.0f ftop=%.0f"
            " obs=(%.0f,%.0f,%.0f,%.0f) norm=(%.0f,%.0f,%.0f,%.0f)"
            " out=(%.0f,%.0f,%.0f,%.0f) snap=%d/%d",
            frame_count, src, action, closeup ? 1 : 0,
            head_fresh ? 1 : 0, head_tsu, face_fresh ? 1 : 0,
            face_frame_age, closeup_anchor_kind_,
            head_fresh ? head_cy - 0.5f * head_h : -1.f,
            face_fresh ? face_head_top : -1.f,
            obs_x1, obs_y1, obs_x2, obs_y2, x1, y1, x2, y2,
            out.at<float>(0, 0), out.at<float>(0, 1),
            out.at<float>(0, 2), out.at<float>(0, 3),
            force_snap ? 1 : 0, effective_force_snap ? 1 : 0);
        trace_push(line);
    }
    return out;
}

void LightTracker::note_returned_box(const cv::Mat& box4)
{
    if (box4.empty() || box4.cols < 4) return;
    const float y1 = box4.at<float>(0, 1);
    const float x1 = box4.at<float>(0, 0);
    const float x2 = box4.at<float>(0, 2);
    const float y2 = box4.at<float>(0, 3);
    if (!std::isfinite(x1) || !std::isfinite(y1)
        || !std::isfinite(x2) || !std::isfinite(y2)) return;
    last_returned_x1_ = x1;
    last_returned_y1_ = y1;
    last_returned_x2_ = x2;
    last_returned_y2_ = y2;
    last_returned_box_ms_ = now_ms();
    last_returned_source_ = frame_output_source_;
}

int LightTracker::motion_source_priority(MotionObservationSource source)
{
    return static_cast<int>(source);
}

const char* LightTracker::motion_source_name(MotionObservationSource source)
{
    switch (source) {
    case MotionObservationSource::BODY: return "BODY";
    case MotionObservationSource::FACE_IDENTITY: return "FACE_IDENTITY";
    case MotionObservationSource::FACE_TRACK: return "FACE_TRACK";
    case MotionObservationSource::HEAD: return "HEAD";
    }
    return "UNKNOWN";
}

const char* LightTracker::motion_quality_name(MotionQuality quality)
{
    switch (quality) {
    case MotionQuality::HIGH: return "HIGH";
    case MotionQuality::MEDIUM: return "MEDIUM";
    case MotionQuality::LOW: return "LOW";
    }
    return "LOW";
}

const char* LightTracker::prediction_mode_name(PredictionMode mode)
{
    switch (mode) {
    case PredictionMode::MOVE_HIGH: return "PRED_MOVE_HIGH";
    case PredictionMode::MOVE_MEDIUM: return "PRED_MOVE_MEDIUM";
    case PredictionMode::HOLD_LOW: return "PRED_HOLD_LOW";
    }
    return "PRED_HOLD_LOW";
}

const char* LightTracker::identity_evidence_name(IdentityEvidence evidence)
{
    switch (evidence) {
    case IdentityEvidence::POSITIVE: return "POSITIVE";
    case IdentityEvidence::NEGATIVE: return "NEGATIVE";
    case IdentityEvidence::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char* LightTracker::person_identity_risk_reason_name(uint32_t reasons)
{
    if (reasons & kPersonRiskOverlap) return "overlap";
    if (reasons & kPersonRiskMergeOnset) return "merge_onset";
    if (reasons & kPersonRiskOwnerCompetition) return "owner_competition";
    if (reasons & kPersonRiskKnownOther) return "known_other";
    if (reasons & kPersonRiskAlertCompetition) return "alert_competition";
    return "none";
}

void LightTracker::note_person_identity_ambiguity(
    uint32_t reasons, int close_count, bool direct_competition)
{
    if (reasons == kPersonRiskNone) return;
    const int64_t now = now_ms();
    if (person_identity_context_.last_evidence_frame == frame_count)
        person_identity_context_.reasons |= reasons;
    else
        person_identity_context_.reasons = reasons;
    person_identity_context_.last_evidence_ms = now;
    person_identity_context_.last_evidence_frame = frame_count;
    person_identity_context_.close_count = close_count;
    if (direct_competition)
        person_identity_context_.last_direct_competition_frame = frame_count;

    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[IDENTITY_CTX] f=%d action=refresh active=1 reason=%s mask=%u"
            " close=%d age=0ms hold=%lldms",
            frame_count,
            person_identity_risk_reason_name(person_identity_context_.reasons),
            (unsigned)person_identity_context_.reasons, close_count,
            (long long)kPersonIdentityRiskHoldMs);
        trace_push(line);
        trace_event_pending_ = true;
    }
}

bool LightTracker::person_identity_ambiguity_active(int64_t now) const
{
    return person_identity_context_.last_evidence_ms >= 0
        && now >= person_identity_context_.last_evidence_ms
        && (now - person_identity_context_.last_evidence_ms)
           <= kPersonIdentityRiskHoldMs;
}

void LightTracker::expire_person_identity_ambiguity(int64_t now)
{
    if (person_identity_context_.last_evidence_ms < 0) return;
    if (person_identity_ambiguity_active(now)) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[IDENTITY_CTX] f=%d action=hold active=1 reason=%s mask=%u"
                " close=%d age=%lldms hold=%lldms",
                frame_count,
                person_identity_risk_reason_name(person_identity_context_.reasons),
                (unsigned)person_identity_context_.reasons,
                person_identity_context_.close_count,
                (long long)(now - person_identity_context_.last_evidence_ms),
                (long long)kPersonIdentityRiskHoldMs);
            trace_push(line);
        }
        return;
    }
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[IDENTITY_CTX] f=%d action=expire active=0 reason=%s mask=%u"
            " close=%d age=%lldms hold=%lldms",
            frame_count,
            person_identity_risk_reason_name(person_identity_context_.reasons),
            (unsigned)person_identity_context_.reasons,
            person_identity_context_.close_count,
            (long long)(now - person_identity_context_.last_evidence_ms),
            (long long)kPersonIdentityRiskHoldMs);
        trace_push(line);
        trace_event_pending_ = true;
    }
    person_identity_context_ = PersonIdentityAmbiguityContext{};
}

void LightTracker::mark_body_identity_evidence(
    const cv::Mat& body_box, IdentityEvidence evidence, const char* reason)
{
    if (body_box.empty() || body_box.cols < 4) return;
    int best = -1;
    float best_iou = 0.f;
    for (int i = 0; i < (int)recovery_body_boxes_.size(); ++i) {
        const cv::Rect& r = recovery_body_boxes_[i];
        cv::Mat raw = (cv::Mat_<float>(1, 4) <<
            (float)r.x, (float)r.y, (float)r.width, (float)r.height);
        float iou = Utils::iou_single(body_box, raw);
        if (iou > best_iou) {
            best_iou = iou;
            best = i;
        }
    }
    if (best < 0 || best_iou < 0.98f
        || best >= (int)recovery_body_identity_evidence_.size()) return;

    IdentityEvidence& old = recovery_body_identity_evidence_[best];
    // 当前帧强正身份优先；负证据只覆盖 UNKNOWN，不能反向污染已确认 BODY。
    // UNKNOWN 只补充“为什么没有身份证据”的诊断原因，不改变三态等级。
    if (evidence == IdentityEvidence::POSITIVE
        || (evidence == IdentityEvidence::NEGATIVE
            && old == IdentityEvidence::UNKNOWN)
        || (evidence == IdentityEvidence::UNKNOWN
            && old == IdentityEvidence::UNKNOWN)) {
        old = evidence;
        recovery_body_identity_reason_[best] = reason ? reason : "unknown";
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[BODY_EVIDENCE] f=%d body=%d status=%s reason=%s iou=%.2f",
                frame_count, best, identity_evidence_name(evidence),
                recovery_body_identity_reason_[best].c_str(), best_iou);
            trace_push(line);
        }
    }
}

LightTracker::IdentityEvidence LightTracker::face_identity_evidence_for_box(
    const cv::Rect& face_box, std::string* reason) const
{
    for (const auto& cached : face_inference_cache_) {
        if (cached.box.x != face_box.x || cached.box.y != face_box.y
            || cached.box.width != face_box.width
            || cached.box.height != face_box.height) continue;
        if (reason) *reason = cached.reason;
        return cached.evidence;
    }
    if (reason) *reason = "not_evaluated";
    return IdentityEvidence::UNKNOWN;
}

LightTracker::IdentityEvidence LightTracker::body_identity_evidence_for_owner(
    int owner_person, std::string* reason) const
{
    if (owner_person < 0
        || owner_person >= (int)recovery_body_identity_evidence_.size()) {
        if (reason) *reason = "no_owner";
        return IdentityEvidence::UNKNOWN;
    }

    IdentityEvidence evidence = recovery_body_identity_evidence_[owner_person];
    std::string why = owner_person < (int)recovery_body_identity_reason_.size()
                    ? recovery_body_identity_reason_[owner_person] : "unknown";
    if (evidence != IdentityEvidence::POSITIVE) {
        for (int i = 0; i < (int)recovery_faces_.size(); ++i) {
            if (i >= (int)recovery_face_owner_person_.size()
                || recovery_face_owner_person_[i] != owner_person
                || (i < (int)recovery_face_owner_ambiguous_.size()
                    && recovery_face_owner_ambiguous_[i])) continue;
            std::string face_reason;
            IdentityEvidence face_evidence =
                face_identity_evidence_for_box(recovery_faces_[i], &face_reason);
            if (face_evidence == IdentityEvidence::NEGATIVE) {
                evidence = IdentityEvidence::NEGATIVE;
                why = "associated_face_mismatch";
                break;
            }
        }
    }
    if (reason) *reason = why;
    return evidence;
}

LightTracker::IdentityEvidence LightTracker::body_identity_evidence_for_box(
    const cv::Mat& body_box, std::string* reason, int* owner_person) const
{
    int best = -1;
    float best_iou = 0.f;
    if (!body_box.empty() && body_box.cols >= 4) {
        for (int i = 0; i < (int)recovery_body_boxes_.size(); ++i) {
            const cv::Rect& r = recovery_body_boxes_[i];
            cv::Mat raw = (cv::Mat_<float>(1, 4) <<
                (float)r.x, (float)r.y, (float)r.width, (float)r.height);
            const float iou = Utils::iou_single(body_box, raw);
            if (iou > best_iou) {
                best_iou = iou;
                best = i;
            }
        }
    }
    if (owner_person) *owner_person = best_iou >= 0.98f ? best : -1;
    if (best < 0 || best_iou < 0.98f) {
        if (reason) *reason = "no_owner";
        return IdentityEvidence::UNKNOWN;
    }
    return body_identity_evidence_for_owner(best, reason);
}

bool LightTracker::body_provisional_geometry(
    const cv::Mat& body_box, float& center_distance,
    float& box_iou, float& size_ratio) const
{
    center_distance = FLT_MAX;
    box_iou = 0.f;
    size_ratio = 0.f;
    if (pending_body_hyp_id_ < 0 || body_box.empty() || body_box.cols < 4)
        return false;

    const float x1 = body_box.at<float>(0, 0);
    const float y1 = body_box.at<float>(0, 1);
    const float x2 = body_box.at<float>(0, 2);
    const float y2 = body_box.at<float>(0, 3);
    const float pw = std::max(1.f, pending_body_x2_ - pending_body_x1_);
    const float ph = std::max(1.f, pending_body_y2_ - pending_body_y1_);
    const float cw = std::max(1.f, x2 - x1);
    const float ch = std::max(1.f, y2 - y1);
    const float pdiag = std::hypot(pw, ph);
    const float cdiag = std::hypot(cw, ch);
    const float pcx = 0.5f * (pending_body_x1_ + pending_body_x2_);
    const float pcy = 0.5f * (pending_body_y1_ + pending_body_y2_);
    const float ccx = 0.5f * (x1 + x2);
    const float ccy = 0.5f * (y1 + y2);
    center_distance = std::hypot(ccx - pcx, ccy - pcy);
    size_ratio = cdiag / std::max(1.f, pdiag);

    const float ix1 = std::max(pending_body_x1_, x1);
    const float iy1 = std::max(pending_body_y1_, y1);
    const float ix2 = std::min(pending_body_x2_, x2);
    const float iy2 = std::min(pending_body_y2_, y2);
    const float iw = std::max(0.f, ix2 - ix1);
    const float ih = std::max(0.f, iy2 - iy1);
    const float inter = iw * ih;
    box_iou = inter / std::max(1.f, pw * ph + cw * ch - inter);

    return center_distance <= kBodyReidHypMatchDiag * std::max(pdiag, cdiag)
        && size_ratio >= 0.40f && size_ratio <= 2.50f;
}

bool LightTracker::body_provisional_scores_stable(float reid, float anchor) const
{
    return pending_body_hyp_id_ >= 0
        && reid + kProvisionalScoreEpsilon >= pending_body_reid_
        && anchor + kProvisionalScoreEpsilon >= pending_body_anchor_;
}

void LightTracker::trace_body_provisional(
    const char* action, int hyp, int prev_hyp, int source, int index,
    float reid, float anchor, float prev_reid, float prev_anchor,
    int streak, int need, float center_distance, float box_iou,
    float size_ratio, const char* reason, const char* gate,
    const char* override_reason)
{
    if (!kMatchTrace) return;
    char line[kTraceLineLen];
    std::snprintf(line, sizeof(line),
        "[PROVISIONAL] f=%d action=%s hyp=%d prevHyp=%d src=%d idx=%d"
        " r=%.3f A=%.3f prevR=%.3f prevA=%.3f streak=%d need=%d"
        " center=%.1f iou=%.3f size=%.3f reason=%s gate=%s override=%s"
        " startF=%d startTs=%lld",
        frame_count, action ? action : "unknown", hyp, prev_hyp,
        source, index, reid, anchor, prev_reid, prev_anchor,
        streak, need, center_distance, box_iou, size_ratio,
        reason ? reason : "unknown", gate ? gate : "unknown",
        override_reason ? override_reason : "none",
        pending_body_start_frame_, (long long)pending_body_start_ms_);
    trace_push(line);
    trace_event_pending_ = true;
}

void LightTracker::clear_body_provisional(
    const char* reason, const char* gate, int source, int index,
    float reid, float anchor, float center_distance,
    float box_iou, float size_ratio, int current_hyp)
{
    if (pending_body_hyp_id_ >= 0 && reason != nullptr) {
        const int previous_hyp = pending_body_hyp_id_;
        const int trace_hyp = current_hyp >= 0 ? current_hyp : previous_hyp;
        const int trace_source = source >= 0 ? source : pending_src_;
        const int trace_index = index >= 0 ? index : pending_idx_;
        const float trace_reid = reid > -998.f ? reid : pending_body_reid_;
        const float trace_anchor = anchor > -998.f
                                 ? anchor : pending_body_anchor_;
        trace_body_provisional(
            "reset", trace_hyp, previous_hyp,
            trace_source, trace_index, trace_reid,
            trace_anchor, pending_body_reid_, pending_body_anchor_,
            reacq_defer_count_, kReacqMaxDefer,
            center_distance, box_iou, size_ratio,
            reason, gate);
    }
    pending_body_hyp_id_ = -1;
    pending_body_x1_ = pending_body_y1_ = -1.f;
    pending_body_x2_ = pending_body_y2_ = -1.f;
    pending_body_reid_ = pending_body_anchor_ = -1.f;
    pending_body_start_frame_ = -1;
    pending_body_start_ms_ = -1;
}

void LightTracker::trace_continuity_gate(
    const char* type, const char* action, bool scene_risk,
    IdentityEvidence part_evidence, IdentityEvidence owner_evidence,
    int owner, const char* reason)
{
    if (!kMatchTrace) return;
    char line[kTraceLineLen];
    std::snprintf(line, sizeof(line),
        "[CONTINUITY_GATE] f=%d type=%s action=%s sceneRisk=%d"
        " partEvidence=%s ownerEvidence=%s owner=%d reason=%s",
        frame_count, type ? type : "UNKNOWN", action ? action : "block",
        scene_risk ? 1 : 0, identity_evidence_name(part_evidence),
        identity_evidence_name(owner_evidence), owner,
        reason ? reason : "unknown");
    trace_push(line);
    if (std::strcmp(action ? action : "block", "block") == 0)
        trace_event_pending_ = true;
}

int LightTracker::box_clip_mask(const cv::Mat& body_box, int width, int height)
{
    if (body_box.empty() || body_box.cols < 4 || width < 2 || height < 2)
        return 0;
    const float x1 = body_box.at<float>(0, 0);
    const float y1 = body_box.at<float>(0, 1);
    const float x2 = body_box.at<float>(0, 2);
    const float y2 = body_box.at<float>(0, 3);
    return (x1 <= 0.5f ? 1 : 0)
         | (y1 <= 0.5f ? 2 : 0)
         | (x2 >= (float)width - 1.5f ? 4 : 0)
         | (y2 >= (float)height - 1.5f ? 8 : 0);
}

void LightTracker::prune_motion_history(int64_t now)
{
    int pruned = 0;
    while (!motion_history_.empty()
           && (now - motion_history_.front().timestamp_ms) > kMotionHistoryMaxAgeMs) {
        motion_history_.pop_front();
        ++pruned;
    }
    while (motion_history_.size() > kMotionHistoryMaxCount) {
        motion_history_.pop_front();
        ++pruned;
    }
    if (pruned > 0 && kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[MOTION_HISTORY] f=%d action=prune count=%d remain=%zu max_age=%lldms",
            frame_count, pruned, motion_history_.size(),
            (long long)kMotionHistoryMaxAgeMs);
        trace_push(line);
    }
}

void LightTracker::clear_motion_history()
{
    motion_history_.clear();
}

void LightTracker::record_motion_observation(
    const cv::Mat& body_box, MotionObservationSource source, int64_t timestamp_ms)
{
    auto reject = [&](const char* reason) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[MOTION_OBS] f=%d action=reject reason=%s src=%s ts=%lld",
                frame_count, reason, motion_source_name(source),
                (long long)timestamp_ms);
            trace_push(line);
        }
    };
    if (body_box.empty() || body_box.type() != CV_32F
        || body_box.rows < 1 || body_box.cols < 4
        || !cv::checkRange(body_box.row(0).colRange(0, 4),
                           true, nullptr, -FLT_MAX, FLT_MAX)) {
        reject("invalid_box");
        return;
    }
    const float x1 = body_box.at<float>(0, 0);
    const float y1 = body_box.at<float>(0, 1);
    const float x2 = body_box.at<float>(0, 2);
    const float y2 = body_box.at<float>(0, 3);
    if (timestamp_ms < 0) {
        reject("invalid_timestamp");
        return;
    }
    if (x2 <= x1 + 1.f || y2 <= y1 + 1.f) {
        reject("degenerate_box");
        return;
    }

    prune_motion_history(timestamp_ms);
    MotionObservation obs;
    obs.frame_id = frame_count;
    obs.timestamp_ms = timestamp_ms;
    obs.x1 = x1; obs.y1 = y1; obs.x2 = x2; obs.y2 = y2;
    obs.source = source;
    obs.clip_mask = box_clip_mask(body_box, img_w, img_h);

    const char* action = "append";
    const char* old_source = "NONE";
    if (!motion_history_.empty() && motion_history_.back().frame_id == frame_count) {
        MotionObservation& old = motion_history_.back();
        old_source = motion_source_name(old.source);
        if (motion_source_priority(source) > motion_source_priority(old.source)
            || source == old.source) {
            old = obs;
            action = "replace";
        } else {
            action = "keep_higher_priority";
        }
    } else {
        if (!motion_history_.empty()
            && timestamp_ms <= motion_history_.back().timestamp_ms) {
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[MOTION_OBS] f=%d action=reject reason=non_monotonic"
                    " src=%s ts=%lld last=%lld",
                    frame_count, motion_source_name(source),
                    (long long)timestamp_ms,
                    (long long)motion_history_.back().timestamp_ms);
                trace_push(line);
            }
            return;
        }
        motion_history_.push_back(obs);
        while (motion_history_.size() > kMotionHistoryMaxCount)
            motion_history_.pop_front();
    }

    if (kMatchTrace) {
        const MotionObservation& kept = motion_history_.back();
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[MOTION_OBS] f=%d action=%s old=%s new=%s kept=%s"
            " priority=%d center=(%.1f,%.1f) clip=0x%X history=%zu ts=%lld",
            frame_count, action, old_source, motion_source_name(source),
            motion_source_name(kept.source), motion_source_priority(kept.source),
            0.5f * (kept.x1 + kept.x2), 0.5f * (kept.y1 + kept.y2),
            kept.clip_mask, motion_history_.size(),
            (long long)kept.timestamp_ms);
        trace_push(line);
    }
}

const LightTracker::SecondaryFrameObservation*
LightTracker::secondary_frame_observation(int tracker_id) const
{
    auto it = secondary_frame_observations_.find(tracker_id);
    if (it == secondary_frame_observations_.end()) return nullptr;
    const SecondaryFrameObservation& obs = it->second;
    if (obs.frame_id != frame_count
        || obs.timestamp_ms != current_frame_timestamp_ms_
        || obs.box.empty() || obs.box.cols < 4) {
        return nullptr;
    }
    return &obs;
}

void LightTracker::prune_relative_motion_history(int64_t now)
{
    for (auto it = relative_motion_history_.begin();
         it != relative_motion_history_.end();) {
        auto& history = it->second;
        while (!history.empty()
               && now >= history.front().timestamp_ms
               && now - history.front().timestamp_ms > kMotionHistoryMaxAgeMs) {
            history.pop_front();
        }
        while (history.size() > kRelativeMotionHistoryMaxCount)
            history.pop_front();
        if (history.empty()) it = relative_motion_history_.erase(it);
        else ++it;
    }
}

void LightTracker::record_relative_motion_observation(
    int tracker_id, const cv::Mat& main_box, const cv::Mat& secondary_box)
{
    auto reject = [&](const char* reason) {
        if (!kMatchTrace) return;
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[RELATIVE_OBS] f=%d other=%d action=reject reason=%s"
            " mainReal=1 secondaryReal=1 mainTs=%lld secondaryTs=%lld",
            frame_count, tracker_id, reason,
            (long long)current_frame_timestamp_ms_,
            (long long)current_frame_timestamp_ms_);
        trace_push(line);
    };

    const SecondaryFrameObservation* secondary_obs =
        secondary_frame_observation(tracker_id);
    if (secondary_obs == nullptr) {
        reject("secondary_not_current_corrected");
        return;
    }
    if (current_frame_timestamp_ms_ < 0) {
        reject("invalid_timestamp");
        return;
    }
    auto valid_box = [](const cv::Mat& box) {
        return !box.empty() && box.type() == CV_32F
            && box.rows >= 1 && box.cols >= 4
            && cv::checkRange(box.row(0).colRange(0, 4),
                              true, nullptr, -FLT_MAX, FLT_MAX)
            && box.at<float>(0, 2) > box.at<float>(0, 0) + 1.f
            && box.at<float>(0, 3) > box.at<float>(0, 1) + 1.f;
    };
    if (!valid_box(main_box) || !valid_box(secondary_box)
        || !valid_box(secondary_obs->box)) {
        reject("invalid_real_box");
        return;
    }
    // 调用方传入的 secondary raw detection 必须就是帧级 association 保存的 observation。
    if (Utils::iou_single(secondary_box, secondary_obs->box) < 0.999f) {
        reject("secondary_observation_mismatch");
        return;
    }

    prune_relative_motion_history(current_frame_timestamp_ms_);
    RelativeCenterObservation obs;
    obs.frame_id = frame_count;
    obs.timestamp_ms = current_frame_timestamp_ms_;
    obs.main_cx = 0.5f * (main_box.at<float>(0, 0) + main_box.at<float>(0, 2));
    obs.main_cy = 0.5f * (main_box.at<float>(0, 1) + main_box.at<float>(0, 3));
    obs.secondary_cx = 0.5f * (secondary_box.at<float>(0, 0)
                              + secondary_box.at<float>(0, 2));
    obs.secondary_cy = 0.5f * (secondary_box.at<float>(0, 1)
                              + secondary_box.at<float>(0, 3));
    obs.relative_x = obs.main_cx - obs.secondary_cx;
    obs.relative_y = obs.main_cy - obs.secondary_cy;
    obs.main_width = std::max(1.f,
        main_box.at<float>(0, 2) - main_box.at<float>(0, 0));
    obs.main_height = std::max(1.f,
        main_box.at<float>(0, 3) - main_box.at<float>(0, 1));
    obs.secondary_width = std::max(1.f,
        secondary_box.at<float>(0, 2) - secondary_box.at<float>(0, 0));
    obs.secondary_height = std::max(1.f,
        secondary_box.at<float>(0, 3) - secondary_box.at<float>(0, 1));
    // validator 尚未启用；先记录 A-relative 归一化几何分布。符号采用
    // secondary-main，与 relative_x/main-secondary 相反，便于直接解释“B 在 A 哪侧”。
    obs.normalized_x = (obs.secondary_cx - obs.main_cx) / obs.main_width;
    obs.normalized_y = (obs.secondary_cy - obs.main_cy) / obs.main_height;
    obs.width_ratio = obs.secondary_width / obs.main_width;
    obs.height_ratio = obs.secondary_height / obs.main_height;
    obs.secondary_aspect = obs.secondary_width / obs.secondary_height;
    const float main_diag = std::hypot(obs.main_width, obs.main_height);
    const float secondary_diag = std::hypot(
        obs.secondary_width, obs.secondary_height);
    obs.reference_diag = std::max(1.f, 0.5f * (main_diag + secondary_diag));

    auto& history = relative_motion_history_[tracker_id];
    const bool has_previous = !history.empty();
    const RelativeCenterObservation previous = has_previous
        ? history.back() : RelativeCenterObservation{};
    if (!history.empty() && history.back().frame_id == frame_count) {
        history.back() = obs;
    } else if (history.empty()
               || obs.timestamp_ms > history.back().timestamp_ms) {
        history.push_back(obs);
    } else {
        reject("non_monotonic");
        return;
    }
    while (history.size() > kRelativeMotionHistoryMaxCount) history.pop_front();

    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[RELATIVE_OBS] f=%d other=%d action=accept"
            " mainReal=1 secondaryReal=1 mainFrame=%d secondaryFrame=%d"
            " mainTs=%lld secondaryTs=%lld rel=(%.1f,%.1f) history=%zu",
            frame_count, tracker_id, frame_count, secondary_obs->frame_id,
            (long long)obs.timestamp_ms,
            (long long)secondary_obs->timestamp_ms,
            obs.relative_x, obs.relative_y, history.size());
        trace_push(line);
        const int64_t previous_dt_ms = has_previous
            ? obs.timestamp_ms - previous.timestamp_ms : -1;
        std::snprintf(line, sizeof(line),
            "[RELATIVE_GEOM] f=%d other=%d rx=%.3f ry=%.3f"
            " drx=%.3f dry=%.3f wr=%.3f hr=%.3f ar=%.3f"
            " dwr=%.3f dhr=%.3f dar=%.3f dt=%lldms diagnostic_only=1",
            frame_count, tracker_id, obs.normalized_x, obs.normalized_y,
            has_previous ? obs.normalized_x - previous.normalized_x : 0.f,
            has_previous ? obs.normalized_y - previous.normalized_y : 0.f,
            obs.width_ratio, obs.height_ratio, obs.secondary_aspect,
            has_previous ? obs.width_ratio - previous.width_ratio : 0.f,
            has_previous ? obs.height_ratio - previous.height_ratio : 0.f,
            has_previous ? obs.secondary_aspect - previous.secondary_aspect : 0.f,
            (long long)previous_dt_ms);
        trace_push(line);
    }
}

bool LightTracker::estimate_relative_emergence_direction(
    int tracker_id, float& dir_x, float& dir_y,
    int* sample_count, float* direction_consistency)
{
    dir_x = dir_y = 0.f;
    if (sample_count) *sample_count = 0;
    if (direction_consistency) *direction_consistency = 0.f;
    auto trace_estimate = [&](const char* result, const char* reason,
                              size_t observations, size_t pairs,
                              float speed, float consistency,
                              float residual_ratio) {
        if (!kMatchTrace) return;
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[RELATIVE_ESTIMATE] f=%d other=%d result=%s reason=%s"
            " obs=%zu pairs=%zu speed=%.4f consistency=%.3f residual=%.3f",
            frame_count, tracker_id, result, reason, observations, pairs,
            speed, consistency, residual_ratio);
        trace_push(line);
    };
    prune_relative_motion_history(current_frame_timestamp_ms_ >= 0
                                ? current_frame_timestamp_ms_ : now_ms());
    auto it = relative_motion_history_.find(tracker_id);
    if (it == relative_motion_history_.end()) {
        trace_estimate("invalid", "no_history", 0, 0, 0.f, 0.f, 0.f);
        return false;
    }
    const auto& history = it->second;

    struct Sample { float vx, vy; };
    std::vector<Sample> samples;
    for (size_t i = 1; i < history.size(); ++i) {
        const auto& a = history[i - 1];
        const auto& b = history[i];
        const int64_t dt_ms = b.timestamp_ms - a.timestamp_ms;
        if (b.frame_id <= a.frame_id || dt_ms <= 0
            || dt_ms > kMotionAdjacentMaxGapMs) continue;
        samples.push_back({(b.relative_x - a.relative_x) / (float)dt_ms,
                           (b.relative_y - a.relative_y) / (float)dt_ms});
    }
    if (sample_count) *sample_count = (int)samples.size();
    if (samples.size() < 2) {
        trace_estimate("invalid", "insufficient_pairs", history.size(),
                       samples.size(), 0.f, 0.f, 0.f);
        return false;
    }

    auto median = [](std::vector<float> values) {
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        float value = values[mid];
        if ((values.size() & 1U) == 0U) {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
            value = 0.5f * (value + values[mid - 1]);
        }
        return value;
    };
    std::vector<float> xs, ys;
    xs.reserve(samples.size()); ys.reserve(samples.size());
    for (const auto& sample : samples) {
        xs.push_back(sample.vx); ys.push_back(sample.vy);
    }
    const float vx = median(xs);
    const float vy = median(ys);
    const float speed = std::hypot(vx, vy);
    const float reference_diag = std::max(1.f, history.back().reference_diag);
    const float stationary_speed = kMotionStationaryDiagRatio * reference_diag
                                 / kShortPredictionNominalDtMs;
    if (!std::isfinite(speed) || speed <= stationary_speed) {
        trace_estimate("invalid", "stationary_or_nonfinite", history.size(),
                       samples.size(), speed, 0.f, 0.f);
        return false;
    }

    int agree = 0;
    std::vector<float> deviations;
    deviations.reserve(samples.size());
    for (const auto& sample : samples) {
        if (sample.vx * vx + sample.vy * vy > 0.f) ++agree;
        deviations.push_back(std::hypot(sample.vx - vx, sample.vy - vy));
    }
    const float consistency = (float)agree / (float)samples.size();
    const float residual_ratio = median(deviations)
                               * kShortPredictionNominalDtMs / reference_diag;
    if (direction_consistency) *direction_consistency = consistency;
    if (consistency < kMotionDirectionMediumRatio
        || residual_ratio > kMotionResidualMediumRatio) {
        trace_estimate("invalid",
                       consistency < kMotionDirectionMediumRatio
                           ? "direction_inconsistent" : "residual_too_large",
                       history.size(), samples.size(), speed,
                       consistency, residual_ratio);
        return false;
    }
    dir_x = vx / speed;
    dir_y = vy / speed;
    const bool finite = std::isfinite(dir_x) && std::isfinite(dir_y);
    trace_estimate(finite ? "valid" : "invalid",
                   finite ? "passed" : "direction_nonfinite",
                   history.size(), samples.size(), speed,
                   consistency, residual_ratio);
    return finite;
}

bool LightTracker::recovery_search_center(float& cx, float& cy) const
{
    const bool emergence_fresh = emergence_valid_
        && current_frame_timestamp_ms_ >= emergence_update_ms_
        && current_frame_timestamp_ms_ - emergence_update_ms_
           <= kEmergenceMaxAgeMs;
    if (emergence_fresh) {
        cx = emergence_cx_;
        cy = emergence_cy_;
        return true;
    }
    if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
        cx = lead_cx_;
        cy = lead_cy_;
        return true;
    }
    return false;
}

LightTracker::MotionEstimate LightTracker::estimate_motion(int64_t now) const
{
    MotionEstimate estimate;
    estimate.observation_count = (int)motion_history_.size();
    if (motion_history_.empty()) return estimate;
    estimate.valid = true;

    struct VelocitySample { float vx, vy; };
    std::vector<VelocitySample> samples;
    samples.reserve(motion_history_.size());
    bool has_body_or_identity_face = false;
    for (const auto& obs : motion_history_) {
        estimate.clip_risk = estimate.clip_risk || obs.clip_mask != 0;
        has_body_or_identity_face = has_body_or_identity_face
            || obs.source == MotionObservationSource::BODY
            || obs.source == MotionObservationSource::FACE_IDENTITY;
    }
    for (size_t i = 1; i < motion_history_.size(); ++i) {
        const MotionObservation& a = motion_history_[i - 1];
        const MotionObservation& b = motion_history_[i];
        estimate.source_transition_risk = estimate.source_transition_risk
                                       || a.source != b.source;
        const int64_t dt_ms = b.timestamp_ms - a.timestamp_ms;
        if (dt_ms <= 0 || dt_ms > kMotionAdjacentMaxGapMs
            || a.clip_mask != b.clip_mask) {
            continue;
        }
        const float acx = 0.5f * (a.x1 + a.x2);
        const float acy = 0.5f * (a.y1 + a.y2);
        const float bcx = 0.5f * (b.x1 + b.x2);
        const float bcy = 0.5f * (b.y1 + b.y2);
        samples.push_back({(bcx - acx) / (float)dt_ms,
                           (bcy - acy) / (float)dt_ms});
    }
    estimate.valid_pair_count = (int)samples.size();
    if (samples.empty()) return estimate;

    auto median = [](std::vector<float> values) {
        const size_t n = values.size();
        const size_t mid = n / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        float value = values[mid];
        if ((n & 1U) == 0U) {
            std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
            value = 0.5f * (value + values[mid - 1]);
        }
        return value;
    };
    std::vector<float> xs, ys;
    xs.reserve(samples.size()); ys.reserve(samples.size());
    for (const auto& sample : samples) {
        xs.push_back(sample.vx); ys.push_back(sample.vy);
    }
    estimate.history_vx = median(xs);
    estimate.history_vy = median(ys);

    const MotionObservation& latest = motion_history_.back();
    const float body_diag = std::max(1.f,
        std::hypot(latest.x2 - latest.x1, latest.y2 - latest.y1));
    const float history_speed = std::hypot(estimate.history_vx,
                                           estimate.history_vy);
    const float stationary_speed = kMotionStationaryDiagRatio * body_diag
                                 / kShortPredictionNominalDtMs;
    std::vector<float> deviations;
    int direction_total = 0, direction_agree = 0;
    for (const auto& sample : samples) {
        deviations.push_back(std::hypot(sample.vx - estimate.history_vx,
                                        sample.vy - estimate.history_vy));
        const float sample_speed = std::hypot(sample.vx, sample.vy);
        if (history_speed > stationary_speed && sample_speed > stationary_speed) {
            ++direction_total;
            if (sample.vx * estimate.history_vx
                + sample.vy * estimate.history_vy > 0.f) {
                ++direction_agree;
            }
        }
    }
    estimate.residual_ratio = median(deviations)
                            * kShortPredictionNominalDtMs / body_diag;
    estimate.direction_consistency = direction_total > 0
        ? (float)direction_agree / (float)direction_total : 1.f;
    const bool stationary = history_speed <= stationary_speed;

    if (estimate.observation_count >= 4 && estimate.valid_pair_count >= 3
        && estimate.residual_ratio <= kMotionResidualHighRatio
        && (stationary
            || estimate.direction_consistency >= kMotionDirectionHighRatio)) {
        estimate.history_quality = MotionQuality::HIGH;
    } else if (estimate.observation_count >= 3 && estimate.valid_pair_count >= 2
               && estimate.residual_ratio <= kMotionResidualMediumRatio
               && (stationary
                   || estimate.direction_consistency >= kMotionDirectionMediumRatio)) {
        estimate.history_quality = MotionQuality::MEDIUM;
    }
    // 三种来源已统一为人体框中心。单纯 source 切换只记录，不自动降级；
    // 若重构偏差真实存在，会由 residual/direction 自然体现。边缘 clip 或全部为弱 HEAD/FACE_TRACK
    // 历史才将 HIGH 上限收到 MEDIUM。
    if ((estimate.clip_risk || !has_body_or_identity_face)
        && estimate.history_quality == MotionQuality::HIGH) {
        estimate.history_quality = MotionQuality::MEDIUM;
    }
    estimate.effective_quality = estimate.history_quality;

    for (const auto& trk : trackers) {
        if (!trk || !trk->get_is_main()) continue;
        const cv::Vec2f kf_velocity = trk->get_kf_velocity();
        estimate.kf_vx = kf_velocity[0] / kShortPredictionNominalDtMs;
        estimate.kf_vy = kf_velocity[1] / kShortPredictionNominalDtMs;
        estimate.kf_valid = std::isfinite(estimate.kf_vx)
                         && std::isfinite(estimate.kf_vy);
        break;
    }
    const float kf_speed = std::hypot(estimate.kf_vx, estimate.kf_vy);
    if (estimate.kf_valid && history_speed > stationary_speed
        && kf_speed > stationary_speed) {
        estimate.kf_direction_cos =
            (estimate.history_vx * estimate.kf_vx
             + estimate.history_vy * estimate.kf_vy)
            / (history_speed * kf_speed);
        estimate.kf_strong_conflict =
            estimate.kf_direction_cos <= kMotionKfReverseCosThreshold;
        if (estimate.kf_strong_conflict) {
            estimate.effective_quality = MotionQuality::LOW;
        } else if (estimate.kf_direction_cos < 0.f
                   && estimate.effective_quality == MotionQuality::HIGH) {
            estimate.effective_quality = MotionQuality::MEDIUM;
        }
    }
    (void)now;
    return estimate;
}

void LightTracker::clear_short_prediction(const char* reason)
{
    if (frozen_prediction_.lifecycle != PredictionLifecycle::IDLE && kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[SHORT_PRED_STATE] f=%d action=clear reason=%s old=%s anchor_f=%d",
            frame_count, reason ? reason : "unknown",
            frozen_prediction_.lifecycle == PredictionLifecycle::ACTIVE
                ? "ACTIVE" : "EXHAUSTED",
            frozen_prediction_.anchor_frame_id);
        trace_push(line);
        trace_event_pending_ = true;
    }
    frozen_prediction_ = FrozenPredictionState{};
}

void LightTracker::exhaust_short_prediction(const char* reason)
{
    if (frozen_prediction_.lifecycle != PredictionLifecycle::ACTIVE) return;
    const int64_t now = now_ms();
    const int64_t prediction_elapsed = std::max<int64_t>(
        0, now - frozen_prediction_.prediction_start_ms);
    const int64_t motion_age = std::max<int64_t>(
        0, now - frozen_prediction_.anchor_timestamp_ms);
    const float displacement = std::hypot(
        frozen_prediction_.last_output_cx - frozen_prediction_.anchor_cx,
        frozen_prediction_.last_output_cy - frozen_prediction_.anchor_cy);
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[SHORT_PRED_STATE] f=%d action=exhaust reason=%s"
            " motion_age=%lldms pred_elapsed=%lldms anchor_f=%d mode=%s"
            " disp=%.1f/%.1f",
            frame_count, reason ? reason : "unknown",
            (long long)motion_age, (long long)prediction_elapsed,
            frozen_prediction_.anchor_frame_id,
            prediction_mode_name(frozen_prediction_.mode),
            displacement, frozen_prediction_.max_displacement);
        trace_push(line);
        trace_event_pending_ = true;
    }
    frozen_prediction_.lifecycle = PredictionLifecycle::EXHAUSTED;
    coast_weight_ = 0.f;
}

bool LightTracker::try_short_prediction(cv::Mat& out_box, const char* context)
{
    out_box.release();
    const int64_t now = now_ms();

    auto reject = [&](const char* reason, bool exhaust_active) {
        if (exhaust_active) exhaust_short_prediction(reason);
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[SHORT_PRED] f=%d result=reject ctx=%s reason=%s lifecycle=%s blind=%lldms",
                frame_count, context ? context : "unknown", reason,
                frozen_prediction_.lifecycle == PredictionLifecycle::ACTIVE ? "ACTIVE"
              : frozen_prediction_.lifecycle == PredictionLifecycle::EXHAUSTED ? "EXHAUSTED"
                                                                               : "IDLE",
                (long long)get_blind_ms());
            trace_push(line);
        }
        return false;
    };

    std::shared_ptr<KalmanBoxTracker> main_trk;
    for (const auto& trk : trackers) {
        if (trk && trk->get_is_main()) {
            main_trk = trk;
            break;
        }
    }
    if (!main_trk) return reject("no_main_tracker", true);

    const bool person_occlusion_risk = person_identity_ambiguity_active(now);
    const bool identity_risk = id_switch_alert_ || pending_active_
                            || pending_from_sweep_ || face_global_pending_
                            || person_occlusion_risk;
    if (frame_dt_sec_ > kOverloadDtHi) return reject("abnormal_frame_gap", true);
    if (frozen_prediction_.lifecycle == PredictionLifecycle::EXHAUSTED)
        return reject("exhausted_anchor", false);

    if (frozen_prediction_.lifecycle == PredictionLifecycle::IDLE) {
        if (main_trk->get_time_since_update() < 1)
            return reject("body_still_measured", false);
        prune_motion_history(now);
        if (motion_history_.empty()) return reject("no_motion_history", false);
        const MotionObservation& anchor = motion_history_.back();
        const int64_t observation_age = std::max<int64_t>(
            0, now - anchor.timestamp_ms);
        if (observation_age > kMotionStartFreshnessMaxAgeMs)
            return reject("motion_observation_stale", false);

        const float x1 = anchor.x1, y1 = anchor.y1;
        const float x2 = anchor.x2, y2 = anchor.y2;
        const bool anchor_valid = std::isfinite(x1) && std::isfinite(y1)
            && std::isfinite(x2) && std::isfinite(y2)
            && x2 > x1 + 1.f && y2 > y1 + 1.f;
        if (!anchor_valid) return reject("invalid_motion_anchor", false);

        MotionEstimate estimate = estimate_motion(now);
        const bool low_visibility = visibility_state_ == VisibilityState::UPPER
                                 || visibility_state_ == VisibilityState::HEAD_ONLY;
        const bool uncertain_candidate = frame_measurement_reliability_
                                      == MeasurementReliability::UNCERTAIN;
        const bool crowded = recovery_body_boxes_.size() > 1;
        // identity/scene risk 继续约束当前候选能否接管，但 short prediction
        // 只使用遮挡前已安全提交的真实 Motion History。这些当帧风险只作
        // 诊断，不再把已经稳定的非零历史速度降成 HOLD_LOW。
        const bool strong_context_risk = identity_risk || low_visibility
                                      || occlusion_state_ != OcclusionState::CLEAR;
        const bool ordinary_context_risk = uncertain_candidate || crowded;
        const bool context_risk = strong_context_risk || ordinary_context_risk;
        const MotionQuality pre_context_quality = estimate.effective_quality;
        const char* context_action = context_risk ? "diagnostic_only" : "none";

        const float body_diag = std::hypot(x2 - x1, y2 - y1);
        const float stationary_speed = kMotionStationaryDiagRatio * body_diag
                                     / kShortPredictionNominalDtMs;
        const float history_speed = std::hypot(estimate.history_vx,
                                               estimate.history_vy);
        const int64_t history_span_ms = motion_history_.size() >= 2
            ? std::max<int64_t>(0, motion_history_.back().timestamp_ms
                                  - motion_history_.front().timestamp_ms)
            : 0;
        FrozenPredictionState frozen;
        frozen.lifecycle = PredictionLifecycle::ACTIVE;
        frozen.motion_quality = estimate.effective_quality;
        frozen.anchor_x1 = x1; frozen.anchor_y1 = y1;
        frozen.anchor_x2 = x2; frozen.anchor_y2 = y2;
        frozen.anchor_cx = 0.5f * (x1 + x2);
        frozen.anchor_cy = 0.5f * (y1 + y2);
        frozen.anchor_timestamp_ms = anchor.timestamp_ms;
        frozen.anchor_frame_id = anchor.frame_id;
        frozen.anchor_source = anchor.source;
        frozen.prediction_start_ms = now;
        frozen.last_output_cx = frozen.anchor_cx;
        frozen.last_output_cy = frozen.anchor_cy;
        if (estimate.effective_quality == MotionQuality::HIGH
            && history_speed > stationary_speed) {
            frozen.mode = PredictionMode::MOVE_HIGH;
            frozen.frozen_vx = estimate.history_vx;
            frozen.frozen_vy = estimate.history_vy;
        } else if (estimate.effective_quality == MotionQuality::MEDIUM
                   && history_speed > stationary_speed) {
            frozen.mode = PredictionMode::MOVE_MEDIUM;
            frozen.frozen_vx = kMotionMediumVelocityScale * estimate.history_vx;
            frozen.frozen_vy = kMotionMediumVelocityScale * estimate.history_vy;
        } else {
            frozen.mode = PredictionMode::HOLD_LOW;
            frozen.frozen_vx = frozen.frozen_vy = 0.f;
        }
        const float frame_diag = std::hypot((float)img_w, (float)img_h);
        frozen.max_displacement = std::min(
            kShortPredictionMaxBodyDiag * body_diag,
            kShortPredictionMaxFrameDiag * frame_diag);
        if (!std::isfinite(frozen.max_displacement)
            || frozen.max_displacement <= 1.f) {
            return reject("invalid_displacement_limit", false);
        }
        frozen_prediction_ = frozen;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[MOTION_ESTIMATE] f=%d obs=%d pairs=%d span=%lldms"
                " hv=(%.3f,%.3f)pxms speed=%.3f stationary=%.3f"
                " residual=%.3f dir=%.2f history_q=%s kf=(%.3f,%.3f)"
                " kf_valid=%d cos=%.2f conflict=%d effective_q=%s risk=%d"
                " strongRisk=%d ordinaryRisk=%d"
                " clip=%d source_transition=%d",
                frame_count, estimate.observation_count, estimate.valid_pair_count,
                (long long)history_span_ms,
                estimate.history_vx, estimate.history_vy,
                history_speed, stationary_speed,
                estimate.residual_ratio, estimate.direction_consistency,
                motion_quality_name(estimate.history_quality),
                estimate.kf_vx, estimate.kf_vy, estimate.kf_valid ? 1 : 0,
                estimate.kf_direction_cos, estimate.kf_strong_conflict ? 1 : 0,
                motion_quality_name(estimate.effective_quality),
                context_risk ? 1 : 0, strong_context_risk ? 1 : 0,
                ordinary_context_risk ? 1 : 0, estimate.clip_risk ? 1 : 0,
                estimate.source_transition_risk ? 1 : 0);
            trace_push(line);
            std::snprintf(line, sizeof(line),
                "[MOTION_QUALITY] f=%d alert_pending=%d occ=%d low_vis=%d"
                " uncertain=%d crowded=%d strong=%d ordinary=%d"
                " pre=%s post=%s action=keep context_action=%s",
                frame_count, identity_risk ? 1 : 0,
                occlusion_state_ != OcclusionState::CLEAR ? 1 : 0,
                low_visibility ? 1 : 0, uncertain_candidate ? 1 : 0,
                crowded ? 1 : 0, strong_context_risk ? 1 : 0,
                ordinary_context_risk ? 1 : 0,
                motion_quality_name(pre_context_quality),
                motion_quality_name(estimate.effective_quality),
                context_action);
            trace_push(line);
            std::snprintf(line, sizeof(line),
                "[SHORT_PRED_STATE] f=%d action=start ctx=%s anchor_f=%d src=%s"
                " anchor_ts=%lld start_ts=%lld initial_gap=%lldms mode=%s q=%s"
                " frozen_v=(%.3f,%.3f)pxms max_ms=%lld max_disp=%.1f"
                " anchor=(%.1f,%.1f) size=(%.1f,%.1f)",
                frame_count, context ? context : "unknown",
                frozen.anchor_frame_id, motion_source_name(frozen.anchor_source),
                (long long)frozen.anchor_timestamp_ms,
                (long long)frozen.prediction_start_ms,
                (long long)observation_age,
                prediction_mode_name(frozen.mode),
                motion_quality_name(frozen.motion_quality),
                frozen.frozen_vx, frozen.frozen_vy,
                (long long)kShortPredictionMaxDurationMs,
                frozen.max_displacement, frozen.anchor_cx, frozen.anchor_cy,
                x2 - x1, y2 - y1);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }

    const int64_t prediction_elapsed_ms = std::max<int64_t>(
        0, now - frozen_prediction_.prediction_start_ms);
    const int64_t motion_extrapolation_age_ms = std::max<int64_t>(
        0, now - frozen_prediction_.anchor_timestamp_ms);
    if (prediction_elapsed_ms >= kShortPredictionMaxDurationMs)
        return reject("duration", true);
    if (frozen_prediction_.mode == PredictionMode::HOLD_LOW
        && prediction_elapsed_ms >= kShortPredictionLowHoldMs) {
        return reject("low_hold_duration", true);
    }

    const float total_dx = frozen_prediction_.frozen_vx
                         * (float)motion_extrapolation_age_ms;
    const float total_dy = frozen_prediction_.frozen_vy
                         * (float)motion_extrapolation_age_ms;
    const float next_cx = frozen_prediction_.anchor_cx + total_dx;
    const float next_cy = frozen_prediction_.anchor_cy + total_dy;
    const float displacement = std::hypot(next_cx - frozen_prediction_.anchor_cx,
                                          next_cy - frozen_prediction_.anchor_cy);
    if (!std::isfinite(displacement)
        || displacement >= frozen_prediction_.max_displacement) {
        return reject("max_displacement", true);
    }

    // 只预测中心，宽高固定；接近边界时整框平移回画面内，不压扁 box。
    const float box_w = frozen_prediction_.anchor_x2 - frozen_prediction_.anchor_x1;
    const float box_h = frozen_prediction_.anchor_y2 - frozen_prediction_.anchor_y1;
    const float frame_x2 = (float)img_w - 1.f;
    const float frame_y2 = (float)img_h - 1.f;
    if (frame_x2 <= 1.f || frame_y2 <= 1.f
        || box_w > frame_x2 || box_h > frame_y2) {
        return reject("invalid_frame_or_size", true);
    }
    float x1 = next_cx - 0.5f * box_w;
    float y1 = next_cy - 0.5f * box_h;
    float x2 = x1 + box_w;
    float y2 = y1 + box_h;
    if (x1 < 0.f) { x2 -= x1; x1 = 0.f; }
    if (y1 < 0.f) { y2 -= y1; y1 = 0.f; }
    if (x2 > frame_x2) { x1 -= x2 - frame_x2; x2 = frame_x2; }
    if (y2 > frame_y2) { y1 -= y2 - frame_y2; y2 = frame_y2; }
    if (x1 < 0.f || y1 < 0.f || x2 <= x1 + 1.f || y2 <= y1 + 1.f)
        return reject("box_out_of_frame", true);

    const float time_conf = 1.f - (float)prediction_elapsed_ms
                                  / (float)kShortPredictionMaxDurationMs;
    const float dist_conf = 1.f - displacement / frozen_prediction_.max_displacement;
    const float weight = std::max(0.f, std::min(time_conf, dist_conf));
    if (!std::isfinite(weight) || weight <= 0.f)
        return reject("confidence", true);

    frozen_prediction_.last_output_cx = next_cx;
    frozen_prediction_.last_output_cy = next_cy;
    frozen_prediction_.last_output_ms = now;
    out_box = (cv::Mat_<float>(1, 4) << x1, y1, x2, y2);
    frame_output_source_ = OutputSource::PREDICTED;
    coast_weight_ = weight;

    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[SHORT_PRED] f=%d result=accept ctx=%s mode=%s"
            " anchor_ts=%lld start_ts=%lld now=%lld initial_gap=%lldms"
            " motion_age=%lldms pred_elapsed=%lldms disp=%.1f/%.1f weight=%.2f"
            " feature_update=0 reason=predicted_not_measurement"
            " box=(%.0f,%.0f,%.0f,%.0f)",
            frame_count, context ? context : "unknown",
            prediction_mode_name(frozen_prediction_.mode),
            (long long)frozen_prediction_.anchor_timestamp_ms,
            (long long)frozen_prediction_.prediction_start_ms, (long long)now,
            (long long)(frozen_prediction_.prediction_start_ms
                        - frozen_prediction_.anchor_timestamp_ms),
            (long long)motion_extrapolation_age_ms,
            (long long)prediction_elapsed_ms,
            displacement, frozen_prediction_.max_displacement, weight,
            x1, y1, x2, y2);
        trace_push(line);
        std::snprintf(line, sizeof(line),
            "[SHORT_PRED_MOTION] f=%d frozen_v=(%.3f,%.3f)pxms"
            " total_delta=(%.1f,%.1f)"
            " center=(%.1f,%.1f)",
            frame_count, frozen_prediction_.frozen_vx,
            frozen_prediction_.frozen_vy,
            total_dx, total_dy, next_cx, next_cy);
        trace_push(line);
    }
    return true;
}

void LightTracker::trace_prediction_coverage(bool has_main_target)
{
    if (!has_main_target) return;
    const bool real_observation = frame_output_source_ == OutputSource::BODY
                               || frame_output_source_ == OutputSource::HEAD
                               || frame_output_source_ == OutputSource::FACE;
    if (real_observation) return;

    ++pred_missing_real_frames_;
    const char* category = "OUTPUT_NONE";
    if (frame_output_source_ == OutputSource::PREDICTED
        && frozen_prediction_.lifecycle == PredictionLifecycle::ACTIVE) {
        if (frozen_prediction_.mode == PredictionMode::MOVE_HIGH) {
            ++pred_move_high_frames_;
            category = "PRED_MOVE_HIGH";
        } else if (frozen_prediction_.mode == PredictionMode::MOVE_MEDIUM) {
            ++pred_move_medium_frames_;
            category = "PRED_MOVE_MEDIUM";
        } else {
            ++pred_hold_low_frames_;
            category = "PRED_HOLD_LOW";
        }
    } else {
        ++pred_output_none_frames_;
    }

    if (kMatchTrace) {
        const double moving_coverage = pred_missing_real_frames_ > 0
            ? (double)(pred_move_high_frames_ + pred_move_medium_frames_)
              / (double)pred_missing_real_frames_ : 0.0;
        const double safe_hold_coverage = pred_missing_real_frames_ > 0
            ? (double)pred_hold_low_frames_ / (double)pred_missing_real_frames_ : 0.0;
        const double none_rate = pred_missing_real_frames_ > 0
            ? (double)pred_output_none_frames_ / (double)pred_missing_real_frames_ : 0.0;
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[PRED_COVERAGE] f=%d category=%s missing=%llu high=%llu medium=%llu"
            " low_hold=%llu none=%llu moving=%.3f safe_hold=%.3f none_rate=%.3f",
            frame_count, category,
            (unsigned long long)pred_missing_real_frames_,
            (unsigned long long)pred_move_high_frames_,
            (unsigned long long)pred_move_medium_frames_,
            (unsigned long long)pred_hold_low_frames_,
            (unsigned long long)pred_output_none_frames_,
            moving_coverage, safe_hold_coverage, none_rate);
        trace_push(line);
    }
}

void LightTracker::update_lead_center(const cv::Mat& box) {
    if (box.empty() || box.cols < 4) return;
    lead_cx_ = (box.at<float>(0, 0) + box.at<float>(0, 2)) * 0.5f;
    lead_cy_ = (box.at<float>(0, 1) + box.at<float>(0, 3)) * 0.5f;
}

void LightTracker::update_body_reid_search_anchor(const cv::Mat& box) {
    if (box.empty() || box.cols < 4) return;
    const float x1 = box.at<float>(0, 0), y1 = box.at<float>(0, 1);
    const float x2 = box.at<float>(0, 2), y2 = box.at<float>(0, 3);
    if (!std::isfinite(x1) || !std::isfinite(y1)
        || !std::isfinite(x2) || !std::isfinite(y2)
        || x2 <= x1 || y2 <= y1) return;
    body_reid_anchor_cx_ = (x1 + x2) * 0.5f;
    body_reid_anchor_cy_ = (y1 + y2) * 0.5f;
    const float obs_diag = std::hypot(x2 - x1, y2 - y1);
    const float frame_diag = (img_w > 0 && img_h > 0)
                           ? std::hypot((float)img_w, (float)img_h) : obs_diag;
    body_reid_anchor_diag_ = std::max(8.f, std::min(obs_diag,
                                                   std::max(8.f, frame_diag)));
    body_reid_anchor_ms_ = now_ms();
}

void LightTracker::set_ptz_blind_phase(int phase) {
    const PtzBlindPhase next = phase == (int)PtzBlindPhase::SLIDING
        ? PtzBlindPhase::SLIDING
        : phase == (int)PtzBlindPhase::STOPPED
            ? PtzBlindPhase::STOPPED : PtzBlindPhase::IDLE;
    if (next == ptz_blind_phase_) return;
    ptz_blind_phase_ = next;
    if (next != PtzBlindPhase::IDLE) {
        spatial_prior_invalid_ = true;
        clear_body_provisional("ptz_slide", "phase_transition");
        pending_active_ = false;
        pending_src_ = pending_idx_ = -1;
        pending_from_sweep_ = false;
        reacq_defer_count_ = 0;
        face_recovery_hypotheses_.clear();
        face_global_pending_ = false;
        face_global_pending_streak_ = 0;
        face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
        face_global_pending_ms_ = -1;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[PTZ_RECOVERY] f=%d phase=%s spatial_prior_invalid=1 anchor=%d",
                frame_count, next == PtzBlindPhase::SLIDING ? "SLIDING" : "STOPPED",
                ptz_blind_anchor_box_.empty() ? 0 : 1);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }
}

void LightTracker::update_ptz_blind_anchor(const cv::Mat& box, int64_t timestamp_ms) {
    if (box.empty() || box.cols < 4) return;
    const float x1 = box.at<float>(0, 0), y1 = box.at<float>(0, 1);
    const float x2 = box.at<float>(0, 2), y2 = box.at<float>(0, 3);
    if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2)
        || !std::isfinite(y2) || x2 <= x1 || y2 <= y1) return;
    ptz_blind_anchor_box_ = box.colRange(0, 4).clone();
    ptz_blind_anchor_ms_ = timestamp_ms;
}

void LightTracker::finish_ptz_blind_reacquisition(int main_trk_idx,
                                                   const cv::Mat& body_box) {
    if (!spatial_prior_invalid_) return;
    if (main_trk_idx >= 0 && main_trk_idx < (int)trackers.size())
        trackers[main_trk_idx]->set_long_coast(true);
    spatial_prior_invalid_ = false;
    ptz_reacq_body_streak_ = 1;
    lead_cx_ = lead_cy_ = -1.f;
    emergence_valid_ = false;
    emergence_cx_ = emergence_cy_ = -1.f;
    body_reid_hypotheses_.clear();
    secondary_frame_observations_.clear();
    relative_motion_history_.clear();
    occluder_tracker_id_ = -1;
    face_recovery_hypotheses_.clear();
    face_global_pending_ = false;
    face_global_pending_streak_ = 0;
    pending_active_ = false;
    pending_src_ = pending_idx_ = -1;
    clear_body_provisional("ptz_reacq_rebase", "rebase");
    clear_coast_search_hint("ptz_reacq_rebase");
    update_ptz_blind_anchor(body_box, now_ms());
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[REACQ_REBASE] f=%d action=body_rebase feature_freeze=1",
            frame_count);
        trace_push(line);
        trace_event_pending_ = true;
    }
}

// ════════════════════════════════════════════════════════════════════
// [N9] 决策 trace：完整缓冲文件 + 固定内存环（取代每帧 UART 直出）
// ════════════════════════════════════════════════════════════════════
void LightTracker::trace_push(const char* line) {
    std::strncpy(trace_ring_[trace_head_], line, kTraceLineLen - 1);
    trace_ring_[trace_head_][kTraceLineLen - 1] = '\0';
    trace_head_ = (trace_head_ + 1) % kTraceRingN;
    if (trace_count_ < kTraceRingN) ++trace_count_;

    // 文件采用大缓冲逐行追加，保留完整时间线（包括算法自身未触发 event 的静默错锁）。
    // 此处通常只有一次内存复制；系统写入与 flush 均在 trace_flush 中节流。
    if (kMatchTrace) match_trace_file_sink().append_line(line);
}

void LightTracker::trace_flush(const char* reason, bool full) {
    if (!kMatchTrace) return;
    if (full) {
        // 文件已经按帧连续写入，不重复回放整环；只追加事件边界，便于快速定位。
        char marker[128];
        std::snprintf(marker, sizeof(marker),
                      "==== [TRACE %s @f%d] ====", reason, frame_count);
        match_trace_file_sink().append_line(marker);
        trace_count_ = 0;
    }
    // 事件立即落到内核页缓存；平稳期每 kTraceHeartbeatFrames 帧一次。
    // 未调用 fsync，不强制等待物理介质。
    match_trace_file_sink().flush();
    trace_last_dump_frame_ = frame_count;
}

// ════════════════════════════════════════════════════════════════════
// [N10] 自运动前馈：无 GMC 的替代先验（只动软参考点，绝不碰 KF 状态）
// ════════════════════════════════════════════════════════════════════
void LightTracker::apply_ego_feedforward() {
    ego_sx_ = ego_sy_ = 0.f; ego_shift_mag_ = 0.f;
    if (!ego_enabled_ || !ego_active_) return;
    // 最近未输出主框（云台已停）→ 前馈失效
    if (now_ms() - ego_out_ms_ > kEgoTimeoutMs) { ego_active_ = false; return; }

    // 本帧闭合比例 f = clamp(β·dt)；预测表观位移 = -f·e（目标向画面中心移动）
    float f = ego_beta_sec_ * frame_dt_sec_;
    if (f < 0.f) f = 0.f;
    if (f > kEgoMaxFracPerFrame) f = kEgoMaxFracPerFrame;
    ego_sx_ = -f * ego_ex_;
    ego_sy_ = -f * ego_ey_;
    ego_shift_mag_ = std::sqrt(ego_sx_ * ego_sx_ + ego_sy_ * ego_sy_);

    // 头部 KF 不直接注入估计相机运动（估错会污染长期状态），但 head-only/face-only
    // 匹配必须使用当前图像坐标。故累计到一个仅供“预测框→本帧候选”比较的偏移量，
    // 一旦获得真实头观测便清零并由 KF 正式校正。
    if (!gmc_enabled_) {
        head_ego_dx_ += ego_sx_;
        head_ego_dy_ += ego_sy_;
    }

    // 平移软参考点（等价于对它们做穷人版 GMC）。最近确认脸和全局待确认脸也必须
    // 跟随云台表观位移，否则 FACE 输出刚驱动云台，下一帧参考点反而停在旧像素位置。
    if (lead_cx_ >= 0.f && lead_cy_ >= 0.f)      { lead_cx_ += ego_sx_;   lead_cy_ += ego_sy_; }
    if (smooth_cx_ > 0.f && smooth_cy_ > 0.f)    { smooth_cx_ += ego_sx_; smooth_cy_ += ego_sy_; }
    if (emergence_valid_)                        { emergence_cx_ += ego_sx_; emergence_cy_ += ego_sy_; }
    if (last_confirmed_face_ms_ >= 0) {
        int dx = (int)std::lround(ego_sx_);
        int dy = (int)std::lround(ego_sy_);
        last_confirmed_face_box_.x += dx;
        last_confirmed_face_box_.width += dx;
        last_confirmed_face_box_.y += dy;
        last_confirmed_face_box_.height += dy;
    }
    if (face_global_pending_) {
        face_global_pending_cx_ += ego_sx_;
        face_global_pending_cy_ += ego_sy_;
    }

    // 消费掉已闭合的误差分量（多帧累计闭合 → 表观位移渐近逼近整段误差）
    ego_ex_ *= (1.f - f);
    ego_ey_ *= (1.f - f);
}

void LightTracker::note_output_for_ego(bool out_valid, float out_cx, float out_cy, bool matched_real) {
    const int64_t now = now_ms();
    const float scx = img_w * 0.5f, scy = img_h * 0.5f;
    const float sdiag = std::sqrt((float)(img_w * img_w + img_h * img_h));
    const float deadband = kEgoDeadbandFrac * sdiag;

    if (!out_valid) {
        // 本帧不输出 → 云台将停 → 前馈失效，并断开 β 采样链（避免跨 coast 空档算位移）
        ego_active_ = false;
        prev_match_valid_ = false;
        return;
    }

    float ex = out_cx - scx, ey = out_cy - scy;
    float emag = std::sqrt(ex * ex + ey * ey);

    // ── β 在线估计（仅连续两个真实命中帧；apparent Δ 才有意义）──
    //   Δ = center_t − center_{t-1}；相机项 s_cam ≈ -β·dt·e_{t-1}；投影到 ê_{t-1}：
    //   β ≈ -(Δ·e_{t-1}) / (dt·|e_{t-1}|²)。仅 |e_{t-1}| 足够大（相机项主导目标运动噪声）时采样。
    if (matched_real && prev_match_valid_ && prev_match_ms_ > 0) {
        float dt = (float)(now - prev_match_ms_) / 1000.f;
        float prev_emag = std::sqrt(prev_match_ex_ * prev_match_ex_ + prev_match_ey_ * prev_match_ey_);
        if (dt > 1e-3f && (now - prev_match_ms_) <= kEgoObsMaxGapMs
            && prev_emag >= 2.f * deadband) {
            float dx = out_cx - prev_match_cx_, dy = out_cy - prev_match_cy_;
            float dot = dx * prev_match_ex_ + dy * prev_match_ey_;
            float beta_s = -dot / (dt * prev_emag * prev_emag);
            if (beta_s > kEgoBetaSecMin && beta_s < kEgoBetaSecMax) {   // 丢弃退化/爆估样本
                ego_beta_sec_ = (1.f - kEgoBetaEma) * ego_beta_sec_ + kEgoBetaEma * beta_s;
            }
        }
    }

    // 未闭合误差（供下帧前馈）+ 死区（|e| 太小 → 云台不动 → 无前馈）
    ego_ex_ = ex; ego_ey_ = ey;
    ego_active_ = (emag >= deadband);
    ego_out_ms_ = now;

    // β 采样锚点（只在真实命中帧串联）
    prev_match_cx_ = out_cx; prev_match_cy_ = out_cy;
    prev_match_ex_ = ex;     prev_match_ey_ = ey;
    prev_match_ms_ = now;
    prev_match_valid_ = matched_real;
}

// ════════════════════════════════════════════════════════════════════
// [N11] 头锚定瞄准点：App 云台应把该点（而非框中心）驱动到画面中心
// ════════════════════════════════════════════════════════════════════
void LightTracker::compute_aim_point(const cv::Mat& main_box4) {
    aim_valid_ = false;
    if (main_box4.empty() || main_box4.cols < 4) return;
    float x1 = main_box4.at<float>(0, 0), y1 = main_box4.at<float>(0, 1);
    float x2 = main_box4.at<float>(0, 2), y2 = main_box4.at<float>(0, 3);
    float bcx = (x1 + x2) * 0.5f;
    float box_h = std::max(1.f, y2 - y1);
    float ref_h = (main_h_hold_ > 1.f) ? main_h_hold_ : box_h;   // 全身高度优先（框半身时更稳）

    // 头部 KF 新鲜 → 头中心下方 ~上胸口；否则框顶 + 比例（消除半身框中心上移的 tilt 抖动）
    KalmanBoxTracker* main_trk = nullptr;
    for (auto& t : trackers) if (t->get_is_main()) { main_trk = t.get(); break; }
    bool head_ok = false;
    if (main_trk && main_trk->has_head_track()
        && main_trk->get_head_time_since_update() <= kHeadPredMaxAge) {
        cv::Mat hp = main_trk->get_head_pred_box();
        if (!hp.empty() && hp.cols >= 4) {
            float hcx = (hp.at<float>(0, 0) + hp.at<float>(0, 2)) * 0.5f;
            float hcy = (hp.at<float>(0, 1) + hp.at<float>(0, 3)) * 0.5f;
            aim_x_ = hcx;
            aim_y_ = hcy + kAimChestFrac * ref_h;
            head_ok = true;
        }
    }
    if (!head_ok) {
        aim_x_ = bcx;
        aim_y_ = y1 + kAimTopFrac * ref_h;
    }
    // 钳到画面内
    aim_x_ = std::min(std::max(aim_x_, 0.f), (float)img_w - 1.f);
    aim_y_ = std::min(std::max(aim_y_, 0.f), (float)img_h - 1.f);
    aim_valid_ = true;
}

// ════════════════════════════════════════════════════════════════════
// [N6] 人脸质量评分（正脸程度+尺寸+完整性）∈[0,1]；退化返回 -1。
//   初始注册（setMainTarget）与延迟注册（try_deferred_face_register）共用。
//   kps_10（ArcFace 5 点）：[左眼,右眼,鼻尖,左嘴角,右嘴角]×(x,y)。
//   face_box_xyxy 为 xyxy 塞 Rect（工程约定：width 存 x2、height 存 y2）。
// ════════════════════════════════════════════════════════════════════
float LightTracker::evaluate_face_quality(const std::vector<float>& k,
                                          const cv::Rect& fb,
                                          int cw, int chh) const {
    if (k.size() < 10) return -1.f;
    float lex = k[0], ley = k[1];
    float rex = k[2], rey = k[3];
    float nx  = k[4], ny  = k[5];
    float mlx = k[6], mly = k[7];
    float mrx = k[8], mry = k[9];

    float eye_d = std::sqrt((rex - lex) * (rex - lex) + (rey - ley) * (rey - ley));
    if (eye_d < 1.f) return -1.f;

    float dnl = std::sqrt((nx - lex) * (nx - lex) + (ny - ley) * (ny - ley));
    float dnr = std::sqrt((nx - rex) * (nx - rex) + (ny - rey) * (ny - rey));
    float dml = std::sqrt((nx - mlx) * (nx - mlx) + (ny - mly) * (ny - mly));
    float dmr = std::sqrt((nx - mrx) * (nx - mrx) + (ny - mry) * (ny - mry));
    float sym = std::min(1.f, 1.5f * (std::fabs(dnl - dnr) / eye_d
                                    + std::fabs(dml - dmr) / eye_d));

    float size_score = std::min(1.f, eye_d / 40.f);
    bool complete = fb.x >= 3 && fb.y >= 3
                 && fb.width  <= cw  - 3
                 && fb.height <= chh - 3;

    float q = (1.f - sym) * 0.55f + size_score * 0.30f + (complete ? 0.15f : 0.f);
    return std::max(0.f, std::min(1.f, q));
}

// ── 头部连续性（A）辅助实现 ──────────────────────────────────────

// 学习"头→体"几何（EMA）。仅在 body 与 head 同时可见时调用。
// 带合理性护栏，避免异常检测污染几何先验。
void LightTracker::learn_head_body_geom(const cv::Mat& body_box,
                                        const cv::Mat& head_box) {
    if (body_box.empty() || head_box.empty()) return;
    if (body_box.cols < 4 || head_box.cols < 4) return;

    float bx1 = body_box.at<float>(0, 0), by1 = body_box.at<float>(0, 1);
    float bx2 = body_box.at<float>(0, 2), by2 = body_box.at<float>(0, 3);
    float hx1 = head_box.at<float>(0, 0), hy1 = head_box.at<float>(0, 1);
    float hx2 = head_box.at<float>(0, 2), hy2 = head_box.at<float>(0, 3);

    float bw = bx2 - bx1, bh = by2 - by1;
    float hw = hx2 - hx1, hh = hy2 - hy1;
    if (bw <= 1.f || bh <= 1.f || hw <= 1.f || hh <= 1.f) return;

    float bcx = (bx1 + bx2) * 0.5f;
    float hcx = (hx1 + hx2) * 0.5f;

    float r_h  = bh / hh;
    float r_w  = bw / hw;
    float r_dx = (bcx - hcx) / hw;

    // 护栏：身高约为头高的 3~12 倍、体宽约为头宽的 1.2~6 倍，超出视为异常
    if (r_h < 3.0f || r_h > 12.0f) return;
    if (r_w < 1.2f || r_w > 6.0f)  return;

    if (!head_body_geom_valid_) {
        hb_h_ratio_  = r_h;  hb_w_ratio_  = r_w;
        hb_dx_ratio_ = r_dx;
        head_body_geom_valid_ = true;
    } else {
        const float a = kHbGeomAlpha;
        hb_h_ratio_  = (1.f - a) * hb_h_ratio_  + a * r_h;
        hb_w_ratio_  = (1.f - a) * hb_w_ratio_  + a * r_w;
        hb_dx_ratio_ = (1.f - a) * hb_dx_ratio_ + a * r_dx;
    }
}

// 学习“脸→完整人体”几何。只由调用方在 CLEAR、单人、完整身体且身份可靠时调用；
// 函数内再做上部归属和比例护栏，避免把邻人的脸或裁剪人体写入尺度先验。
void LightTracker::learn_face_body_geom(const cv::Mat& body_box,
                                        const cv::Mat& face_box) {
    if (body_box.empty() || face_box.empty()
        || body_box.cols < 4 || face_box.cols < 4) return;

    const float bx1 = body_box.at<float>(0, 0), by1 = body_box.at<float>(0, 1);
    const float bx2 = body_box.at<float>(0, 2), by2 = body_box.at<float>(0, 3);
    const float fx1 = face_box.at<float>(0, 0), fy1 = face_box.at<float>(0, 1);
    const float fx2 = face_box.at<float>(0, 2), fy2 = face_box.at<float>(0, 3);
    const float bw = bx2 - bx1, bh = by2 - by1;
    const float fw = fx2 - fx1, fh = fy2 - fy1;
    if (bw <= 1.f || bh <= 1.f || fw <= 1.f || fh <= 1.f) return;

    const float bcx = (bx1 + bx2) * 0.5f;
    const float fcx = (fx1 + fx2) * 0.5f, fcy = (fy1 + fy2) * 0.5f;
    if (fcx < bx1 || fcx > bx2
        || fcy < by1 - 0.05f * bh || fcy > by1 + 0.45f * bh) return;

    const float r_h = bh / fh;
    const float r_w = bw / fw;
    const float r_dx = (bcx - fcx) / fw;
    if (r_h < 4.5f || r_h > 18.f || r_w < 1.3f || r_w > 8.f
        || std::fabs(r_dx) > 2.f) return;

    if (!face_body_geom_valid_) {
        fb_h_ratio_ = r_h; fb_w_ratio_ = r_w;
        fb_dx_ratio_ = r_dx;
        face_body_geom_valid_ = true;
    } else {
        const float a = kFbGeomAlpha;
        fb_h_ratio_  = (1.f - a) * fb_h_ratio_  + a * r_h;
        fb_w_ratio_  = (1.f - a) * fb_w_ratio_  + a * r_w;
        fb_dx_ratio_ = (1.f - a) * fb_dx_ratio_ + a * r_dx;
    }
}

// 由头部框按学习到的几何重建身体框。几何无效 → 返回空（调用方转盲 coast）。
cv::Mat LightTracker::reconstruct_body_from_head(const cv::Mat& head_box) const {
    if (!head_body_geom_valid_) return cv::Mat();
    if (head_box.empty() || head_box.cols < 4) return cv::Mat();

    float hx1 = head_box.at<float>(0, 0), hy1 = head_box.at<float>(0, 1);
    float hx2 = head_box.at<float>(0, 2), hy2 = head_box.at<float>(0, 3);
    float hcx = (hx1 + hx2) * 0.5f;
    float hw  = std::max(1.f, hx2 - hx1), hh = std::max(1.f, hy2 - hy1);

    // §2 防几何漂移：EMA 学习的头↔体比例若被异常帧污染，会把重建体框甩到远处空区 →
    //   云台追向空区。用时钳到人体合理范围（默认高比 7.0、宽比 2.5、横偏移 0.0
    //   均落在内；纵向顶部不再学习偏移，而是直接使用 head_y1）。
    auto clampf = [](float v, float lo, float hi){ return v < lo ? lo : (v > hi ? hi : v); };
    float bw  = hw * clampf(hb_w_ratio_,  1.5f,  5.0f);
    float bh  = hh * clampf(hb_h_ratio_,  3.0f, 11.0f);
    // 历史真实体尺寸只作极宽的异常护栏，不再用旧 ±40% 窄带限制景深变化。
    // 当前头变大时允许重构完整人体越过画面，再由统一画面求交得到真实可见部分；
    // 头轨迹尺寸门 + 输出整框稳定器负责抑制单帧头检测抖动。
    if (last_main_bw_ > 0.f && last_main_bh_ > 0.f) {
        bw = clampf(bw, last_main_bw_ * kHeadReconSizeLo, last_main_bw_ * kHeadReconSizeHi);
        bh = clampf(bh, last_main_bh_ * kHeadReconSizeLo, last_main_bh_ * kHeadReconSizeHi);
    }
    float bcx = hcx + clampf(hb_dx_ratio_, -1.5f, 1.5f) * hw;

    // 人体框顶部必须保持“头顶”语义，供 App 的大/中/小顶部距离直接使用。
    // 旧公式由 body_cy/body_h 反推 y1；默认 7.0/3.5 会得到 y1=head_cy，
    // 近距离时比真实头顶低约半个头高，控制器因此会把头顶推出画面。
    // 纵向改为直接从当前头框顶部向下构建；横向与总高度仍使用学习几何。
    const float body_y1 = hy1;

    return (cv::Mat_<float>(1, 4) <<
        bcx - bw * 0.5f, body_y1,
        bcx + bw * 0.5f, body_y1 + bh);
}

// 由独立人脸框预测完整人体框（人脸连续性 B 专用）。无论是否已学到目标个体
// 几何，都按当前脸尺寸缩放：未学习时 fb_* 是保守默认值，学习后是目标自身比例。
// 历史人体尺寸只作极宽异常护栏，不能阻止近距离大脸重构出越界完整框。
cv::Mat LightTracker::reconstruct_body_from_face(const cv::Rect& face_box) const {
    // face_box 为 xyxy 塞 Rect：x=x1, y=y1, width=x2, height=y2
    float fx1 = (float)face_box.x,     fy1 = (float)face_box.y;
    float fx2 = (float)face_box.width, fy2 = (float)face_box.height;
    if (fx2 - fx1 < 1.f || fy2 - fy1 < 1.f) return cv::Mat();
    float fcx = (fx1 + fx2) * 0.5f;

    const float fw = fx2 - fx1, fh = fy2 - fy1;
    auto clampf = [](float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    };
    float bw = fw * clampf(fb_w_ratio_, 1.3f, 8.f);
    float bh = fh * clampf(fb_h_ratio_, 4.5f, 18.f);
    const float bcx = fcx + clampf(fb_dx_ratio_, -2.f, 2.f) * fw;
    if (last_main_bw_ > 0.f && last_main_bh_ > 0.f) {
        bw = clampf(bw, last_main_bw_ * kHeadReconSizeLo,
                        last_main_bw_ * kHeadReconSizeHi);
        bh = clampf(bh, last_main_bh_ * kHeadReconSizeLo,
                        last_main_bh_ * kHeadReconSizeHi);
    }

    // 人脸框一般不含完整额头/头发，向上补 1/4 脸高作为保守头顶；从该 y1
    // 向下构建完整人体，确保仅脸找回时返回框顶部仍可直接用于构图距离控制。
    const float body_y1 = fy1 - kFaceToHeadTopFrac * fh;
    return (cv::Mat_<float>(1, 4) <<
        bcx - bw * 0.5f, body_y1,
        bcx + bw * 0.5f, body_y1 + bh);
}

bool LightTracker::face_body_geometry_consistent(const cv::Rect& face_box,
                                                 const cv::Mat& body_box,
                                                 float* out_cost) const {
    if (out_cost) *out_cost = FLT_MAX;
    if (body_box.empty() || body_box.cols < 4) return false;

    const float bx1 = body_box.at<float>(0, 0);
    const float by1 = body_box.at<float>(0, 1);
    const float bx2 = body_box.at<float>(0, 2);
    const float by2 = body_box.at<float>(0, 3);
    const float bw = bx2 - bx1;
    const float bh = by2 - by1;
    const float fx1 = (float)face_box.x;
    const float fy1 = (float)face_box.y;
    const float fx2 = (float)face_box.width;
    const float fy2 = (float)face_box.height;
    const float fw = fx2 - fx1;
    const float fh = fy2 - fy1;
    if (!std::isfinite(bx1) || !std::isfinite(by1)
        || !std::isfinite(bx2) || !std::isfinite(by2)
        || bw <= 1.f || bh <= 1.f || fw <= 1.f || fh <= 1.f)
        return false;

    const float fcx = (fx1 + fx2) * 0.5f;
    const float fcy = (fy1 + fy2) * 0.5f;
    const float bcx = (bx1 + bx2) * 0.5f;
    const float rel_x = std::fabs(fcx - bcx) / bw;
    const float rel_y = (fcy - by1) / bh;
    const float face_w_ratio = fw / bw;
    const float face_h_ratio = fh / bh;

    // 人脸允许少量越过人体框顶，但必须位于人体上部并靠近水平中线。尺寸闸只
    // 排除明显“一个大人体框吞进远处小脸”等异常，不限制正常全身/半身变化。
    bool basic = fcx >= bx1 - 0.05f * bw
              && fcx <= bx2 + 0.05f * bw
              && fcy >= by1 - 0.10f * bh
              && fcy <= by1 + 0.55f * bh
              && rel_x <= 0.32f
              && rel_y >= -0.10f && rel_y <= 0.55f
              && face_w_ratio >= 0.06f && face_w_ratio <= 0.80f
              && face_h_ratio >= 0.04f && face_h_ratio <= 0.60f;
    if (!basic) return false;

    float recon_x_cost = 0.f;
    cv::Mat recon = reconstruct_body_from_face(face_box);
    if (!recon.empty() && recon.cols >= 4) {
        const float rx1 = recon.at<float>(0, 0);
        const float rx2 = recon.at<float>(0, 2);
        const float rw = std::max(1.f, rx2 - rx1);
        const float rcx = (rx1 + rx2) * 0.5f;
        recon_x_cost = std::fabs(bcx - rcx) / std::max(bw, rw);
        if (recon_x_cost > 0.30f) return false;
    }

    if (out_cost) {
        *out_cost = 1.5f * rel_x
                  + std::fabs(rel_y - 0.20f)
                  + 0.75f * recon_x_cost;
    }
    return true;
}

// 人脸已经确认身份时，用其所处的独立头校正头部 KF。这样“头+脸”共存的帧不会
// 白白浪费头观测，下一帧脸消失时可立即切到 head-only，而不是拿陈旧预测去找头。
void LightTracker::sync_head_track_from_confirmed_face(const cv::Rect& face_box,
                                                       bool relocate) {
    const float fcx = ((float)face_box.x + (float)face_box.width) * 0.5f;
    const float fcy = ((float)face_box.y + (float)face_box.height) * 0.5f;
    int best = -1;
    int best_area = std::numeric_limits<int>::max();
    for (int i = 0; i < (int)recovery_heads_.size(); ++i) {
        const cv::Rect& h = recovery_heads_[i];  // xyxy 语义
        if (fcx < h.x || fcx > h.width || fcy < h.y || fcy > h.height) continue;
        int area = std::max(1, h.width - h.x) * std::max(1, h.height - h.y);
        if (area < best_area) { best_area = area; best = i; }
    }
    if (best < 0) {
        // 全局人脸已经把主目标搬到新位置时，旧头 KF 仍可能在遮挡前位置保持最多
        // kHeadPredMaxAge 帧。compute_aim_point 会优先使用它，造成“FACE 框已到右侧，
        // 瞄准点仍在左侧”。没有当前真实头可重建时，宁可清掉旧头，让瞄准点回退到
        // 已确认的人脸重构框；局部连续脸不清，仍保留短时 head-only 退路。
        if (relocate) {
            for (const auto& trk : trackers) {
                if (!trk->get_is_main()) continue;
                trk->clear_head_track();
                head_ego_dx_ = head_ego_dy_ = 0.f;
                null_sink << "[HEAD_FACE_SYNC] global face invalidated stale head KF"
                             << std::endl;
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[HEAD_FACE_SYNC] f=%d action=clear_stale_head reason=global_face_no_head",
                        frame_count);
                    trace_push(line);
                }
                break;
            }
        }
        return;
    }

    for (const auto& trk : trackers) {
        if (!trk->get_is_main()) continue;
        const cv::Rect& h = recovery_heads_[best];
        cv::Mat hb = (cv::Mat_<float>(1, 4) <<
            (float)h.x, (float)h.y, (float)h.width, (float)h.height);
        if (relocate) trk->rebase_head(hb);
        else          trk->update_head(hb);
        head_ego_dx_ = head_ego_dy_ = 0.f;
        null_sink << "[HEAD_FACE_SYNC] confirmed face corrected head KF" << std::endl;
        return;
    }
}

void LightTracker::try_register_face_from_confirmed_head(const cv::Mat& img) {
    if (face_recognizer.has_face_template()) {
        face_registered_ = true;
        return;
    }
    face_registered_ = false;
    if (img.empty()) return;
    // 身体丢失时不能把任意脸当模板；仅允许“已经被 head-only 连续性确认”的
    // 单头单脸场景，且脸中心必须落在该头内。
    if (recovery_heads_.size() != 1 || recovery_faces_.size() != 1) return;
    const cv::Rect& h = recovery_heads_[0];
    const cv::Rect& f = recovery_faces_[0];
    float fcx = ((float)f.x + (float)f.width) * 0.5f;
    float fcy = ((float)f.y + (float)f.height) * 0.5f;
    if (fcx < h.x || fcx > h.width || fcy < h.y || fcy > h.height) return;
    if ((float)(f.height - f.y) < kFaceRegisterMinFacePx) return;
    if (!take_face_model_slot()) return;

    FaceKeypointResult fk = detector_fk.run(img, f);
    std::vector<float> kps_10 = get_kps10(fk.points);
    if (kps_10.size() < 10) return;
    float q = evaluate_face_quality(kps_10, f, img.cols, img.rows);
    if (q < kFaceInitRegisterMinQ) return;

    std::string reg_result = face_recognizer.register_face(img, kps_10, "bro");
    if (reg_result != "success" || !face_recognizer.has_face_template()) {
        face_registered_ = face_recognizer.has_face_template();
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_REG] f=%d result=reject mode=head reason=%s template=%d",
                frame_count, reg_result.c_str(), face_registered_ ? 1 : 0);
            trace_push(line);
            trace_event_pending_ = true;
        }
        return;
    }
    face_registered_ = true;
    face_template_quality_ = q;
    last_face_register_ms_ = now_ms();
    last_confirmed_face_box_ = f;
    last_confirmed_face_ms_ = last_face_register_ms_;
    last_confirmed_face_frame_ = frame_count;
    sync_head_track_from_confirmed_face(f);
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_REG] f=%d result=accept mode=head q=%.2f",
            frame_count, q);
        trace_push(line);
    }
    null_sink << "[FACE_REG] head-confirmed standalone face registered (q=" << q << ")" << std::endl;
}

// ── 主目标朝向 + 肩部连续性辅助实现 ─────────────────────────────

// 从关键点提取肩中点(像素)+肩宽；双肩置信度不足或肩宽过小返回 false。
bool LightTracker::extract_shoulder_geom(const PoseKeypoint* kps, float conf_thresh,
                                         cv::Point2f& mid, float& width) {
    if (!kps) return false;
    const PoseKeypoint& ls = kps[LEFT_SHOULDER];
    const PoseKeypoint& rs = kps[RIGHT_SHOULDER];
    if (ls.confidence < conf_thresh || rs.confidence < conf_thresh) return false;
    mid.x = (ls.x + rs.x) * 0.5f;
    mid.y = (ls.y + rs.y) * 0.5f;
    float dx = ls.x - rs.x, dy = ls.y - rs.y;
    width = std::sqrt(dx * dx + dy * dy);
    return width >= 1.f;
}

// 主目标朝向评估 + 肩部框相对几何 EMA。见 .h。
void LightTracker::update_orientation_state(const PoseResult* main_pose, const cv::Mat& body_box) {
    // 门控：无 pose / 框太小 → 朝向不可信 → frontalness_=-1（下游门控全空操作）
    if (!main_pose || body_box.empty() || body_box.cols < 4) { frontalness_ = -1.f; return; }
    float bx1 = body_box.at<float>(0, 0), by1 = body_box.at<float>(0, 1);
    float bx2 = body_box.at<float>(0, 2), by2 = body_box.at<float>(0, 3);
    float bw = bx2 - bx1, bh = by2 - by1;
    if (bh < kOrientMinTorsoPx || bw < 4.f) { frontalness_ = -1.f; return; }

    const PoseKeypoint* kps = main_pose->keypoints;
    cv::Point2f smid; float sw = 0.f;
    if (!extract_shoulder_geom(kps, 0.4f, smid, sw)) { frontalness_ = -1.f; return; }

    // ── frontalness 主判据：面部关键点可见性（背对时鼻/眼不可见 → 恰是 ReID 失效之时）──
    auto vis = [&](int i) { return kps[i].confidence >= 0.4f; };
    float face = (vis(NOSE) ? 0.5f : 0.f)
               + (vis(LEFT_EYE) ? 0.25f : 0.f)
               + (vis(RIGHT_EYE) ? 0.25f : 0.f);      // [0,1]：正面≈1，背面≈0
    // 侧身修正：肩宽相对躯干高很窄 → profile → 封顶。躯干高优先用肩→髋，退框高一半。
    float torso_h = bh * 0.5f;
    if (vis(LEFT_HIP) && vis(RIGHT_HIP)) {
        float hipx = (kps[LEFT_HIP].x + kps[RIGHT_HIP].x) * 0.5f;
        float hipy = (kps[LEFT_HIP].y + kps[RIGHT_HIP].y) * 0.5f;
        float d = std::sqrt((smid.x - hipx) * (smid.x - hipx) + (smid.y - hipy) * (smid.y - hipy));
        if (d > 1.f) torso_h = d;
    }
    float sr   = (torso_h > 1.f) ? sw / torso_h : 1.f;                 // 肩宽/躯干高
    float side = std::min(1.f, std::max(0.f, (sr - 0.25f) / 0.35f));   // 窄→0(profile)，宽→1
    float raw  = std::min(face, 0.5f + 0.5f * side);                   // 面部为主，侧身封顶；背对 face=0→raw=0

    frontalness_ = (frontalness_ < 0.f) ? raw
                 : (1.f - kFrontalnessAlpha) * frontalness_ + kFrontalnessAlpha * raw;

    // ── 肩部框相对几何 EMA（肩可见即更新；偏移≈框顶部中心，视角近似不变）──
    float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
    float dxr = (smid.x - bcx) / std::max(1.f, bw);
    float dyr = (smid.y - bcy) / std::max(1.f, bh);
    float wr  = sw / std::max(1.f, bw);
    // 护栏：肩宽应在框宽 0.15~1.2 倍、肩心在框上半区 → 拒绝异常几何污染
    if (wr > 0.15f && wr < 1.2f && dyr > -0.6f && dyr < 0.2f) {
        if (!shoulder_geom_valid_) {
            sb_dx_ratio_ = dxr; sb_dy_ratio_ = dyr; sb_w_ratio_ = wr;
            shoulder_geom_valid_ = true;
        } else {
            const float a = 0.2f;
            sb_dx_ratio_ = (1.f - a) * sb_dx_ratio_ + a * dxr;
            sb_dy_ratio_ = (1.f - a) * sb_dy_ratio_ + a * dyr;
            sb_w_ratio_  = (1.f - a) * sb_w_ratio_  + a * wr;
        }
    }
}

// 候选肩中点 vs 主目标预测肩中点(框相对偏移重建)的标准化距离分 [0,1]。见 .h。
float LightTracker::shoulder_cont_score(const cv::Point2f& cand_mid, const cv::Mat& pred_body_box) const {
    if (!shoulder_geom_valid_) return 0.f;
    if (cand_mid.x < 0.f || pred_body_box.empty() || pred_body_box.cols < 4) return 0.f;
    float bx1 = pred_body_box.at<float>(0, 0), by1 = pred_body_box.at<float>(0, 1);
    float bx2 = pred_body_box.at<float>(0, 2), by2 = pred_body_box.at<float>(0, 3);
    float bw = std::max(1.f, bx2 - bx1), bh = std::max(1.f, by2 - by1);
    float bcx = (bx1 + bx2) * 0.5f, bcy = (by1 + by2) * 0.5f;
    float pmx = bcx + sb_dx_ratio_ * bw;              // 预测肩中点
    float pmy = bcy + sb_dy_ratio_ * bh;
    float scale = std::max(4.f, sb_w_ratio_ * bw);    // 归一尺度 = 预测肩宽
    float d = std::sqrt((cand_mid.x - pmx) * (cand_mid.x - pmx)
                      + (cand_mid.y - pmy) * (cand_mid.y - pmy));
    float m = 1.f - (d / scale) / kShoulderContFalloff;
    return std::max(0.f, std::min(1.f, m));
}

// 在本帧全部恢复头候选中找最接近 head_pred 中心的头部框（标准化门控）。
// 门控半径 = kHeadReacqGateRatio × 头部预测框较长边 × gate_scale。返回索引或 -1。
// 注意：此路径没有任何外观/身份校验（无身体可提 emb），吸附错头会被几何重建
// 放大成整个身体框并喂入预测器（错误轨迹自我强化）。两道防线：
//   ① gate_scale 由调用方按预测新鲜度收缩（越陈旧半径越小）；
//   ② 门内出现第二个距离相当的头 → 歧义，拒绝吸附（宁可盲 coast 不抓错）。
int LightTracker::find_recovery_head_near(const cv::Mat& head_pred, float gate_scale,
                                          float* out_d1, float* out_d2,
                                          float* out_gate) const {
    if (head_pred.empty() || head_pred.cols < 4) return -1;
    if (recovery_heads_.empty()) return -1;

    float px1 = head_pred.at<float>(0, 0), py1 = head_pred.at<float>(0, 1);
    float px2 = head_pred.at<float>(0, 2), py2 = head_pred.at<float>(0, 3);
    float pcx = (px1 + px2) * 0.5f, pcy = (py1 + py2) * 0.5f;
    float pw  = std::max(1.f, px2 - px1), ph = std::max(1.f, py2 - py1);

    float gate = kHeadReacqGateRatio * std::max(pw, ph) * std::max(0.1f, gate_scale);
    int   best = -1;
    // 记录全部候选中的真实最近/次近距离；旧实现把 d1 初始化为 gate，门内无人时
    // trace 会误写成 d1==gate，而实际最近距离被塞进 d2，妨碍现场判断。是否进门
    // 在遍历后单独判断，匹配行为（严格 d<gate）保持不变。
    float d1 = FLT_MAX, d2 = FLT_MAX;
    for (int i = 0; i < (int)recovery_heads_.size(); ++i) {
        const cv::Rect& h = recovery_heads_[i];   // xyxy
        float hcx = (h.x + h.width)  * 0.5f;
        float hcy = (h.y + h.height) * 0.5f;
        float d = std::sqrt((hcx - pcx) * (hcx - pcx) + (hcy - pcy) * (hcy - pcy));
        if (d < d1)      { d2 = d1; d1 = d; best = i; }
        else if (d < d2) { d2 = d; }
    }
    if (out_d1) *out_d1 = d1;
    if (out_d2) *out_d2 = d2;
    if (out_gate) *out_gate = gate;
    if (best < 0 || !(d1 < gate)) return -1;

    // 歧义检查：门内还有第二个头且与最近者不可分（非决定性接近）→ 拒绝
    if (d2 < gate && d1 > kHeadReacqAmbigRatio * d2) {
        null_sink << "[HEAD] standalone ambiguous: d1=" << d1
                  << " d2=" << d2 << " gate=" << gate << " -> reject" << std::endl;
        return -1;
    }
    // 置信度门（F3）：head_match/头重捕是纯空间信号，无外观/身份证据；遮挡期低置信或
    //   半截头检测极易把连续性桥接到邻人头上 → 几何重建出邻人身体、错误指派身份。
    //   要求该头检测分 ≥ kHeadReacqMinScore（几何门/歧义门之外的第三道防线）。
    // 无竞争单头是“人体被家具遮挡”的高概率场景，可接受较低但连续的检测分；
    // 多头仍保持严格阈值，避免纯空间路径吸到旁人。
    const float min_score = recovery_heads_.size() == 1
                          ? kHeadReacqMinScoreSolo : kHeadReacqMinScore;
    if (best < (int)recovery_head_scores_.size()
        && recovery_head_scores_[best] < min_score) {
        null_sink << "[HEAD] recovery low-conf: score=" << recovery_head_scores_[best]
                  << " < " << min_score << " -> reject" << std::endl;
        return -1;
    }
    return best;
}

// 尝试头部连续维持：身体丢失时用独立头部 + 头部 KF 预测维持跟踪。
// 成功（找到匹配头 + 有几何先验）→ 用真实头校正头部 KF、重建身体框写入
// out_box、返回 true。失败 → 返回 false（调用方进入盲 coast）。
bool LightTracker::try_head_continuity(cv::Mat& out_box) {
    if (recovery_heads_.empty()) return false;

    // 调用方始终先做人脸识别。若脸因角度/质量未通过，仍允许头轨迹在最近候选
    // 明显优于次近候选时接续；歧义门和已确认他人占用否决共同防止交错吸错头。

    // 定位主 tracker（头部 KF 仅主目标维护）
    int main_idx = -1;
    for (int i = 0; i < (int)trackers.size(); ++i) {
        if (trackers[i]->get_is_main()) { main_idx = i; break; }
    }
    if (main_idx < 0) return false;

    auto& trk = trackers[main_idx];
    if (!trk->has_head_track()) return false;

    cv::Mat head_pred = trk->get_head_pred_box();   // unmatched 路径已含 GMC 补偿
    int tsu = trk->get_head_time_since_update();
    if (head_pred.empty() || tsu > kHeadPredMaxAge) {
        head_reacq_pending_streak_ = 0;
        head_reacq_pending_ms_ = -1;
        return false;
    }

    // 无 GMC 时，云台居中会让真实头在图像内移动；主 KF 不吃这个估计位移，
    // 因此此处只对用于关联的副本加累计 ego 偏移，不能再拿旧图像坐标作头匹配。
    head_pred = head_pred.clone();
    head_pred.at<float>(0, 0) += head_ego_dx_;
    head_pred.at<float>(0, 2) += head_ego_dx_;
    head_pred.at<float>(0, 1) += head_ego_dy_;
    head_pred.at<float>(0, 3) += head_ego_dy_;

    // 吸附半径随预测新鲜度收缩：tsu 越大头部预测越漂移（新调参的头部 KF 响应
    // 更灵敏，漂移也更快），固定 2.5×头径的门会把邻人的独立头吸进来 →
    // 几何重建出"邻人的身体框"，轨迹被错误指派。fresh(tsu=0)→1.0×，
    // 最陈旧(tsu=kHeadPredMaxAge)→0.4×（即 2.5→1.0 倍头径）。
    float gate_scale = 1.f - 0.6f * (float)tsu / (float)kHeadPredMaxAge;
    float head_d1 = FLT_MAX, head_d2 = FLT_MAX, head_gate = 0.f;
    int sh = find_recovery_head_near(head_pred, gate_scale,
                                     &head_d1, &head_d2, &head_gate);
    bool extended_reacq = false;
    float ext_d1 = FLT_MAX, ext_d2 = FLT_MAX, ext_gate = 0.f;

    // 严格门外恢复：只在头预测已有一定陈旧度时启用。扩展门仍执行原有的最近/次近
    // 歧义门和检测分门；命中后还必须是同一空间假设连续 N 帧，绝不单帧跳头。
    if (sh < 0 && tsu >= kHeadReacqExtendedMinTsu) {
        int ext_sh = find_recovery_head_near(
            head_pred, 1.f + kHeadReacqExtendedAgeGrow * (float)tsu,
            &ext_d1, &ext_d2, &ext_gate);
        if (ext_sh >= 0) {
            const cv::Rect& eh = recovery_heads_[ext_sh];
            float ecx = (float)(eh.x + eh.width) * 0.5f;
            float ecy = (float)(eh.y + eh.height) * 0.5f;
            float escale = std::max(4.f, (float)std::max(eh.width - eh.x,
                                                               eh.height - eh.y));
            int64_t now_head_ms = now_ms();
            bool pending_fresh = head_reacq_pending_ms_ >= 0
                              && (now_head_ms - head_reacq_pending_ms_)
                                 <= kHeadReacqExtendedMaxGapMs;
            float step = pending_fresh
                       ? std::sqrt((ecx - head_reacq_pending_cx_)
                                   * (ecx - head_reacq_pending_cx_)
                                 + (ecy - head_reacq_pending_cy_)
                                   * (ecy - head_reacq_pending_cy_))
                       : FLT_MAX;
            bool same_hyp = pending_fresh
                         && step <= kHeadReacqExtendedStepRatio * escale;
            head_reacq_pending_streak_ = same_hyp
                                       ? head_reacq_pending_streak_ + 1 : 1;
            head_reacq_pending_cx_ = ecx;
            head_reacq_pending_cy_ = ecy;
            head_reacq_pending_ms_ = now_head_ms;

            if (head_reacq_pending_streak_ < kHeadReacqExtendedConfirmFrames) {
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[HEAD_RECOVERY] f=%d result=extended_hold n=%d tsu=%d"
                        " d1=%.1f d2=%.1f strict=%.1f ext=%.1f streak=%d/%d",
                        frame_count, (int)recovery_heads_.size(), tsu,
                        ext_d1, ext_d2, head_gate, ext_gate,
                        head_reacq_pending_streak_, kHeadReacqExtendedConfirmFrames);
                    trace_push(line);
                    trace_event_pending_ = true;
                }
                return false;
            }
            sh = ext_sh;
            extended_reacq = true;
        }
    }

    if (sh < 0) {
        if (head_reacq_pending_ms_ >= 0
            && (now_ms() - head_reacq_pending_ms_) > kHeadReacqExtendedMaxGapMs) {
            head_reacq_pending_streak_ = 0;
            head_reacq_pending_ms_ = -1;
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[HEAD_RECOVERY] f=%d result=reject_gate n=%d tsu=%d"
                " d1=%.1f d2=%.1f gate=%.1f extGate=%.1f",
                frame_count, (int)recovery_heads_.size(), tsu,
                head_d1, head_d2, head_gate, ext_gate);
            trace_push(line);
        }
        return false;
    }

    if (!extended_reacq) {
        head_reacq_pending_streak_ = 0;
        head_reacq_pending_ms_ = -1;
    }

    const cv::Rect& h = recovery_heads_[sh];   // xyxy 塞 Rect
    cv::Mat hb = (cv::Mat_<float>(1, 4) <<
        (float)h.x, (float)h.y, (float)h.width, (float)h.height);

    const int owner_person = (sh < (int)recovery_head_owner_person_.size())
                           ? recovery_head_owner_person_[sh] : -1;
    std::string owner_reason;
    const IdentityEvidence owner_evidence =
        body_identity_evidence_for_owner(owner_person, &owner_reason);
    if (owner_evidence == IdentityEvidence::NEGATIVE) {
        const bool scene_risk = person_identity_ambiguity_active(now_ms());
        trace_continuity_gate(
            "HEAD", "block", scene_risk, IdentityEvidence::UNKNOWN,
            owner_evidence, owner_person, owner_reason.c_str());
        return false;
    }

    // 正式绑定人体的头若处于人体交错，只刷新 scene risk；最终是否允许接管由下方
    // owner evidence 统一决定。强正 BODY 可通过，UNKNOWN/NEGATIVE 不能借纯空间头连续
    // 绕过身份门；普通单人/家具遮挡不因“仅有头”本身触发 scene risk。
    if (owner_person >= 0
        && owner_person < (int)recovery_body_boxes_.size()
        && owner_person < (int)recovery_body_valid_.size()
        && recovery_body_valid_[owner_person]) {
        auto rect_iou_xyxy = [](const cv::Rect& a, const cv::Rect& b) {
            float ix1 = (float)std::max(a.x, b.x);
            float iy1 = (float)std::max(a.y, b.y);
            float ix2 = (float)std::min(a.width, b.width);
            float iy2 = (float)std::min(a.height, b.height);
            float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;
            float aa = std::max(0, a.width - a.x) * std::max(0, a.height - a.y);
            float ab = std::max(0, b.width - b.x) * std::max(0, b.height - b.y);
            return inter / std::max(1.f, aa + ab - inter);
        };
        bool owner_crossing = false;
        float max_body_iou = 0.f;
        for (int p = 0; p < (int)recovery_body_boxes_.size(); ++p) {
            if (p == owner_person || p >= (int)recovery_body_valid_.size()
                || !recovery_body_valid_[p]) continue;
            float io = rect_iou_xyxy(recovery_body_boxes_[owner_person],
                                     recovery_body_boxes_[p]);
            max_body_iou = std::max(max_body_iou, io);
            if (io > 0.05f) owner_crossing = true;
        }
        if (owner_crossing) {
            note_person_identity_ambiguity(
                kPersonRiskOwnerCompetition,
                person_identity_context_.close_count, false);
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[HEAD_RECOVERY] f=%d result=owner_competition sh=%d owner=%d body_iou=%.2f",
                    frame_count, sh, owner_person, max_body_iou);
                trace_push(line);
                trace_event_pending_ = true;
            }
        }
    }

    // B7-① 尺寸一致性：独立头对角线须与头部预测框可比。不同景深的旁观者头
    //   （近大远小）与主目标头尺寸差异显著 → 直接拒绝，防"粘上别人的头"。
    {
        float pw = head_pred.at<float>(0, 2) - head_pred.at<float>(0, 0);
        float ph = head_pred.at<float>(0, 3) - head_pred.at<float>(0, 1);
        float cw = (float)(h.width - h.x), chh = (float)(h.height - h.y);
        float pdiag = std::sqrt(pw * pw + ph * ph);
        float cdiag = std::sqrt(cw * cw + chh * chh);
        if (pdiag > 1.f && cdiag > 1.f) {
            float ratio = cdiag / pdiag;
            if (ratio < kHeadSizeRatioMin || ratio > kHeadSizeRatioMax) {
                null_sink << "[HEAD_SIZE] standalone/pred diag ratio=" << ratio
                             << " out of [" << kHeadSizeRatioMin << "," << kHeadSizeRatioMax
                             << "] -> reject" << std::endl;
                return false;
            }
        }
    }

    // ── 空间占用冲突（H1）：独立头落在已确认二级轨迹上部时，弱 continuity 不接管。──
    //   身体路径有 coexist 否决（match_main_target_unified），头部连续性原先没有 → 补齐同一护栏。
    //   失效模式：主目标走到他人身后，遮挡者身体检测偶尔丢帧 → 其头变"独立头" → 被此路径误吸。
    //   用与 matchPersonFaces 相同的"头中心水平在身体框内、纵向上部 [-0.10,0.45]"几何判定归属；
    //   轨迹须已确认(coexist_with_main≥kCoexistConfirm)+新鲜(帧+墙钟)。命中即返回 false → 盲 coast。
    //   直接对齐(主目标正后方)时头框与遮挡者身体重叠、几何本不可分 → 此时否决=保持，正是期望行为。
    //   注意：这只是空间冲突，不是身份 NEGATIVE。本帧真实校正的 secondary 可刷新
    //   scene risk；prediction-only secondary 只阻止本次 continuity，不能制造跨帧身份风险。
    {
        float hcx = (float)(h.x + h.width)  * 0.5f;   // h 为 xyxy 塞 Rect：x=x1,y=y1,width=x2,height=y2
        float hcy = (float)(h.y + h.height) * 0.5f;
        int64_t nowm = now_ms();
        for (auto& st : trackers) {
            if (st->get_is_main() || st->quarantined_) continue;
            if (st->coexist_with_main < kCoexistConfirm) continue;              // 必须"已确认他人"
            if (st->get_time_since_update() > kCoexistVetoMaxTsu) continue;     // 新鲜（帧）
            if (st->last_update_ms_ >= 0 && (nowm - st->last_update_ms_) > kCoexistVetoMaxMs) continue; // 新鲜（墙钟）
            const SecondaryFrameObservation* current_obs =
                secondary_frame_observation(st->get_id());
            const bool current_corrected = current_obs != nullptr;
            cv::Mat ob = current_corrected ? current_obs->box : st->get_state();
            if (ob.empty() || ob.cols < 4) continue;
            float ox1 = ob.at<float>(0, 0), oy1 = ob.at<float>(0, 1);
            float ox2 = ob.at<float>(0, 2), oy2 = ob.at<float>(0, 3);
            float ph  = oy2 - oy1;  if (ph <= 0.f) continue;
            if (hcx < ox1 || hcx > ox2) continue;                              // 头中心水平在他人框内
            if (hcy < oy1 - ph * 0.10f || hcy > oy1 + ph * 0.45f) continue;    // 且在其上部（头区）
            null_sink << "[HEAD_OWNED] standalone head spatially occupied by other id="
                      << st->get_id() << " observation="
                      << (current_corrected ? "current" : "predicted")
                      << " -> block continuity" << std::endl;
            if (current_corrected) {
                note_person_identity_ambiguity(
                    kPersonRiskKnownOther,
                    person_identity_context_.close_count, false);
            }
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[KNOWN_OTHER_SPATIAL] f=%d type=HEAD other=%d"
                    " otherObservation=%s identityNegative=0 action=block_continuity",
                    frame_count, st->get_id(),
                    current_corrected ? "current_corrected" : "prediction_only");
                trace_push(line);
                trace_event_pending_ = true;
            }
            trace_continuity_gate(
                "HEAD", "block", person_identity_ambiguity_active(nowm),
                IdentityEvidence::UNKNOWN, owner_evidence, owner_person,
                current_corrected ? "current_known_other_spatial_conflict"
                                  : "predicted_known_other_spatial_conflict");
            return false;
        }
    }

    const bool owner_ambiguous = sh < (int)recovery_head_owner_ambiguous_.size()
                              && recovery_head_owner_ambiguous_[sh] != 0;
    if (owner_ambiguous) {
        note_person_identity_ambiguity(
            kPersonRiskOwnerCompetition,
            person_identity_context_.close_count, false);
    }
    const bool scene_risk = person_identity_ambiguity_active(now_ms());
    if (scene_risk && owner_evidence != IdentityEvidence::POSITIVE) {
        trace_continuity_gate(
            "HEAD", "block", true, IdentityEvidence::UNKNOWN,
            owner_evidence, owner_person,
            owner_person < 0 ? "scene_risk_standalone_unknown"
                             : "scene_risk_owner_unknown");
        return false;
    }
    trace_continuity_gate(
        "HEAD", "allow", scene_risk, IdentityEvidence::UNKNOWN,
        owner_evidence, owner_person, "permission_pass");

    // 连续真实头观测不再因固定 4 秒预算自行停止。身份安全由每帧重新执行的
    // 轨迹门、次近歧义、尺寸门、人体交错归属门和已确认他人占用门保证。
    if (head_only_since_ms_ < 0) head_only_since_ms_ = now_ms();

    cv::Mat body = reconstruct_body_from_head(hb);
    if (body.empty()) return false;   // 无几何先验，无法重建 → 转盲 coast
    body = clip_reconstructed_body_to_frame(body, "HEAD");
    if (body.empty()) return false;

    trk->update_head(hb);             // 用真实头校正头部 KF（先前 update(empty) 为 no-op，干净）
    head_ego_dx_ = head_ego_dy_ = 0.f;
    trk->correct_body_from_part(body, false);
    // 正常头身比例重构后已经与画面求交：远距遮挡仍是完整人体，近距越界则是
    // 画面内可见部分。同一个裁剪框统一用于 KF、lead 和 PTZ 原始输出。
    update_lead_center(body);
    out_box = stabilize_returned_box(body, OutputSource::HEAD, false);
    if (out_box.empty()) return false;
    last_head_continuity_ms_ = now_ms();
    record_motion_observation(body, MotionObservationSource::HEAD,
                              current_frame_timestamp_ms_);
    head_reacq_pending_streak_ = 0;
    head_reacq_pending_ms_ = -1;
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[HEAD_RECOVERY] f=%d result=accept mode=%s sh=%d owner=%d"
            " score=%.2f tsu=%d d1=%.1f d2=%.1f gate=%.1f",
            frame_count, extended_reacq ? "extended" : "strict",
            sh, owner_person,
            sh < (int)recovery_head_scores_.size() ? recovery_head_scores_[sh] : 0.f,
            tsu, extended_reacq ? ext_d1 : head_d1,
            extended_reacq ? ext_d2 : head_d2,
            extended_reacq ? ext_gate : head_gate);
        trace_push(line);
        trace_event_pending_ = true;
    }
    return true;
}

// 人脸连续性（B）：身体不可见、仅露独立人脸时的强身份维持。
//   与头部连续性关键区别——每帧都用人脸识别复核身份，故：
//     ① 天然免疫"粘错人"（识别是同衣免疫的强身份证据，非纯空间信号）→ 无需时间预算；
//     ② 允许目标在遮挡期已移动（识别到就认，空间门只用于挑选/控预算，不作身份判据）。
//   成功（局部或全画面恢复脸被识别为主目标）→ 由当前头/脸尺度重建体框、置人脸锁、返回 true。
//   失败（无独立脸/无模板/识别不到）→ 返回 false，调用方转盲 coast（宁停勿错）。
bool LightTracker::try_face_only_continuity(const cv::Mat& img, cv::Mat& out_box) {
    auto reject_early = [&](const char* reason) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_RECOVERY] f=%d result=reject_early reason=%s raw=%d"
                " registered=%d history=%d budget=%d",
                frame_count, reason, (int)recovery_faces_.size(),
                face_registered_ ? 1 : 0,
                (last_main_bw_ > 0.f && last_main_bh_ > 0.f) ? 1 : 0,
                face_model_budget_);
            trace_push(line);
        }
        return false;
    };
    if (recovery_faces_.empty()) return reject_early("no_raw_face");
    if (!face_registered_ || !face_recognizer.has_face_template()) {
        face_registered_ = face_recognizer.has_face_template();
        return reject_early("no_template");                     // 无模板无法识别
    }
    if (last_main_bw_ <= 0.f || last_main_bh_ <= 0.f)
        return reject_early("no_body_history");                  // 无历史尺度无法建立搜索优先门
    if (img.empty()) return reject_early("empty_image");

    int main_idx = -1;
    for (int i = 0; i < (int)trackers.size(); ++i)
        if (trackers[i]->get_is_main()) { main_idx = i; break; }
    if (main_idx < 0) return reject_early("no_main_tracker");
    auto& trk = trackers[main_idx];

    const int64_t face_now_ms = now_ms();

    // 搜索参考只决定调度顺序，不再决定“能不能识别”。上一帧已经确认的真实脸位置
    // 优先级最高，避免全局找回后人体 KF 的弱几何校正仍滞留旧位置；随后才是头 KF、
    // 人体 KF 和 lead。全局 rotor 会覆盖参考门之外的脸。
    float pcx, pcy;
    bool have_ref = false;
    if (last_confirmed_face_ms_ >= 0
        && face_now_ms - last_confirmed_face_ms_ <= kFaceRecoveryRefMaxAgeMs
        && last_confirmed_face_box_.width > last_confirmed_face_box_.x
        && last_confirmed_face_box_.height > last_confirmed_face_box_.y) {
        pcx = ((float)last_confirmed_face_box_.x
             + (float)last_confirmed_face_box_.width) * 0.5f;
        pcy = ((float)last_confirmed_face_box_.y
             + (float)last_confirmed_face_box_.height) * 0.5f;
        have_ref = true;
    }

    cv::Mat hp = trk->get_head_pred_box();
    if (!have_ref && trk->has_head_track() && !hp.empty() && hp.cols >= 4
        && trk->get_head_time_since_update() <= kHeadPredMaxAge) {
        hp = hp.clone();
        hp.at<float>(0, 0) += head_ego_dx_; hp.at<float>(0, 2) += head_ego_dx_;
        hp.at<float>(0, 1) += head_ego_dy_; hp.at<float>(0, 3) += head_ego_dy_;
        pcx = (hp.at<float>(0, 0) + hp.at<float>(0, 2)) * 0.5f;
        pcy = (hp.at<float>(0, 1) + hp.at<float>(0, 3)) * 0.5f;
        have_ref = true;
    }
    if (!have_ref) {
        cv::Mat sb = trk->get_state();
        if (!sb.empty() && sb.cols >= 4) {
            pcx = (sb.at<float>(0, 0) + sb.at<float>(0, 2)) * 0.5f + head_ego_dx_;
            pcy = (sb.at<float>(0, 1) + sb.at<float>(0, 3)) * 0.5f + head_ego_dy_;
            have_ref = true;
        } else if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
            pcx = lead_cx_; pcy = lead_cy_;
            have_ref = true;
        } else {
            return reject_early("no_search_reference");
        }
    }

    // 门半径 = 历史体框对角线 × kFaceOnlyGateDiag（丢失期目标可能已移动 → 给足）
    float gate = kFaceOnlyGateDiag
               * std::sqrt(last_main_bw_ * last_main_bw_ + last_main_bh_ * last_main_bh_);

    // [N12] 过载时把预算收到 1（打断慢帧螺旋，仍保留最近脸的恢复机会）。
    // 独立人脸识别统一走 face_recognition_verification 的 standalone 分支：
    //   - 不复用 body source/index，避免把 face idx 当 body idx；
    //   - 空间门只做局部优先级，不再硬删除全画面候选；
    //   - 局部首选 + 全局 rotor 仍只消费共享 top-K，控制额外 NPU 耗时。
    int budget = overload_mode_ ? std::min(1, face_model_budget_)
                                : std::min(kFaceOnlyMaxCand, face_model_budget_);
    // 人体路径可能已识别过同一张脸并耗尽本帧 slot，随后候选又被遮挡/C-identity
    // 闸拒绝。此时仍应让恢复路径读取缓存结果；预算为 0 只禁止新推理，不禁止复用。
    if (budget <= 0) {
        if (face_inference_cache_.empty()) return reject_early("no_budget_or_cache");
        // 只读本帧缓存不产生新推理，可以遍历全部恢复脸，避免已被人体路径识别的
        // 真脸恰好不在当前 rotor 批次而无法复用。
        budget = (int)recovery_faces_.size();
    }
    cv::Point2f ref(pcx, pcy);
    DetectionGroups empty_groups;
    Verification_Result verification = face_recognition_verification(
        cv::Mat::zeros(0, 6, CV_32F), 0,
        empty_groups, 0, img,
        cv::Mat::zeros(0, 2, CV_32S), cv::Mat::zeros(0, 2, CV_32S),
        &ref, gate, budget);
    if (!verification.got_match || !verification.matched_standalone) {
        ++face_recovery_fail_streak_;
        if (face_global_pending_
            && face_now_ms - face_global_pending_ms_ > kFaceGlobalConfirmMaxGapMs) {
            face_global_pending_ = false;
            face_global_pending_streak_ = 0;
            face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
            face_global_pending_ms_ = -1;
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_RECOVERY] f=%d result=reject n=%d budget=%d cache=%d",
                frame_count, (int)recovery_faces_.size(), budget,
                (int)face_inference_cache_.size());
            trace_push(line);
        }
        return false;
    }

    cv::Mat body = verification.standalone_body_box;
    if (body.empty()) return reject_early("body_reconstruct");

    if (verification.standalone_face_idx < 0
        || verification.standalone_face_idx >= (int)recovery_faces_.size())
        return reject_early("bad_face_index");
    const cv::Rect& confirmed_face = recovery_faces_[verification.standalone_face_idx];

    // 同帧若有真实头包含该脸，优先按已学习头身几何重构；它比“脸心 + 固定体高偏移”
    // 更能适应低头/侧头。没有头或几何尚未学会时继续使用人脸重构框。
    int confirmed_head_idx = -1;
    int confirmed_head_area = std::numeric_limits<int>::max();
    float fcx = ((float)confirmed_face.x + (float)confirmed_face.width) * 0.5f;
    float fcy = ((float)confirmed_face.y + (float)confirmed_face.height) * 0.5f;
    for (int h = 0; h < (int)recovery_heads_.size(); ++h) {
        const cv::Rect& hb = recovery_heads_[h];
        if (fcx < hb.x || fcx > hb.width || fcy < hb.y || fcy > hb.height) continue;
        int area = std::max(1, hb.width - hb.x) * std::max(1, hb.height - hb.y);
        if (area < confirmed_head_area) {
            confirmed_head_area = area;
            confirmed_head_idx = h;
        }
    }
    if (confirmed_head_idx >= 0) {
        const cv::Rect& hb = recovery_heads_[confirmed_head_idx];
        cv::Mat head_box = (cv::Mat_<float>(1, 4) <<
            (float)hb.x, (float)hb.y, (float)hb.width, (float)hb.height);
        cv::Mat head_body = reconstruct_body_from_head(head_box);
        if (!head_body.empty()) body = head_body;
    }
    body = clip_reconstructed_body_to_frame(body, "FACE_REC");
    if (body.empty()) return reject_early("body_clip");

    // 远距离命中属于全局身份重定位。FaceReco 正匹配的基础阈值为 0.60；全画面
    // 多人比较会放大偶发误识，因此 >=0.65 直接提交，否则要求同一脸位置连续两帧。
    cv::Mat old_state = trk->get_state();
    float relocate_dist = 0.f;
    bool far_relocate = verification.standalone_global;
    if (!old_state.empty() && old_state.cols >= 4) {
        float ocx = (old_state.at<float>(0, 0) + old_state.at<float>(0, 2)) * 0.5f;
        float ocy = (old_state.at<float>(0, 1) + old_state.at<float>(0, 3)) * 0.5f;
        float bcx = (body.at<float>(0, 0) + body.at<float>(0, 2)) * 0.5f;
        float bcy = (body.at<float>(0, 1) + body.at<float>(0, 3)) * 0.5f;
        relocate_dist = std::hypot(bcx - ocx, bcy - ocy);
        float ow = std::max(1.f, old_state.at<float>(0, 2) - old_state.at<float>(0, 0));
        float oh = std::max(1.f, old_state.at<float>(0, 3) - old_state.at<float>(0, 1));
        far_relocate = far_relocate
                    || relocate_dist > kFaceRelocateBodyDiag * std::hypot(ow, oh);
    }

    if (far_relocate && verification.standalone_sim < kFaceGlobalDirectSim) {
        float fw = std::max(1.f, (float)(confirmed_face.width - confirmed_face.x));
        float fh = std::max(1.f, (float)(confirmed_face.height - confirmed_face.y));
        float same_tol = kFaceGlobalSameHypFaceDiag * std::hypot(fw, fh);
        bool same_hyp = face_global_pending_
                     && face_now_ms - face_global_pending_ms_ <= kFaceGlobalConfirmMaxGapMs
                     && std::hypot(fcx - face_global_pending_cx_,
                                   fcy - face_global_pending_cy_) <= same_tol;
        face_global_pending_streak_ = same_hyp ? face_global_pending_streak_ + 1 : 1;
        face_global_pending_ = true;
        face_global_pending_cx_ = fcx;
        face_global_pending_cy_ = fcy;
        face_global_pending_ms_ = face_now_ms;
        ++face_recovery_fail_streak_;
        if (face_global_pending_streak_ < kFaceGlobalConfirmFrames) {
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_GLOBAL] f=%d result=pending fi=%d sim=%.2f streak=%d/%d"
                    " jump=%.1f",
                    frame_count, verification.standalone_face_idx,
                    verification.standalone_sim, face_global_pending_streak_,
                    kFaceGlobalConfirmFrames, relocate_dist);
                trace_push(line);
                trace_event_pending_ = true;
            }
            return false;
        }
    }

    sync_head_track_from_confirmed_face(confirmed_face, far_relocate);

    // 人脸确认身份 → 置人脸锁（后续帧 FACE_HOLD 以此 TTL 维持，防非人脸特征夺锁）。
    //   不刷新 emb/画廊（重建框为近似，会污染外观模板）——识别本身即身份，逐帧复核足够。
    face_locked_          = true;
    last_face_lock_frame_ = frame_count;
    last_face_lock_ms_    = face_now_ms;
    face_lock_box_        = body.clone();
    last_confirmed_face_box_ = confirmed_face;
    last_confirmed_face_ms_ = face_now_ms;
    last_confirmed_face_frame_ = frame_count;
    head_only_since_ms_   = -1;   // 人脸有强身份证据，不消耗纯 head-only 连续性时间预算
    trk->correct_body_from_part(body, true, far_relocate);
    update_lead_center(body);
    update_body_reid_search_anchor(body);
    out_box = stabilize_returned_box(body, OutputSource::FACE, far_relocate);
    if (out_box.empty()) return reject_early("output_stabilize");
    record_motion_observation(body, MotionObservationSource::FACE_IDENTITY,
                              current_frame_timestamp_ms_);
    face_lock_box_ = body.clone();  // 身份锁仍绑定内部代理，不能被输出防抖状态污染
    last_face_identity_ms_ = face_now_ms;
    last_real_obs_ms_ = face_now_ms;

    // 当前脸恢复的是 target identity，不代表 A/B 人物交错已经结束。scene risk
    // 只由真实竞争证据刷新并在独立 hold 后自然退出，不能被一次强身份提前清除。
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[IDENTITY_RECOVERY] f=%d source=FACE_IDENTITY"
            " sceneRiskStillActive=%d",
            frame_count,
            person_identity_ambiguity_active(face_now_ms) ? 1 : 0);
        trace_push(line);
        trace_event_pending_ = true;
    }

    // 强身份已把主目标定位到新位置：旧身体暂定假设、浮现点和遮挡者归属不能再把
    // 后续身体重捕拉回旧位置。人脸观测可直接让 OCC 进入恢复态，云台无需等待 3600ms。
    clear_body_provisional("face_identity_relocated", "face_only_recovery");
    reacq_defer_count_ = 0;
    pending_active_ = false;
    pending_from_sweep_ = false;
    pending_src_ = pending_idx_ = -1;
    pending_cx_ = pending_cy_ = -1.f;
    if (far_relocate) {
        emergence_valid_ = false;
        emergence_update_ms_ = -1;
        occluder_tracker_id_ = -1;
    }
    if (occlusion_state_ == OcclusionState::OCCLUDED) {
        occlusion_state_ = OcclusionState::RECOVERING;
        recovery_start_ms_ = face_now_ms;
        separation_streak_ = 0;
        trace_event_pending_ = true;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[STATE] f=%d OCC->RECOV reason=face_identity mode=%s jump=%.1f",
                frame_count, far_relocate ? "global" : "local", relocate_dist);
            trace_push(line);
        }
    }
    if (id_switch_alert_) {
        id_switch_alert_ = false;
        suspect_streak_ = 0;
        alert_start_ms_ = 0;
        trace_event_pending_ = true;
    }
    face_global_pending_ = false;
    face_global_pending_streak_ = 0;
    face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
    face_global_pending_ms_ = -1;
    face_recovery_fail_streak_ = 0;
    face_recovery_rotor_ = 0;
    face_recovery_hypotheses_.clear();
    next_face_recovery_hyp_id_ = 1;
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_RECOVERY] f=%d result=accept mode=%s fi=%d sim=%.2f"
            " head=%d jump=%.1f n=%d budget_left=%d",
            frame_count, far_relocate ? "global" : "local",
            verification.standalone_face_idx, verification.standalone_sim,
            confirmed_head_idx, relocate_dist,
            (int)recovery_faces_.size(),
            face_model_budget_);
        trace_push(line);
        trace_event_pending_ = true;
    }
    null_sink << "[FACE_ONLY] recovered by independent face (sim="
                 << verification.standalone_sim
                 << " fi=" << verification.standalone_face_idx << ")" << std::endl;
    return true;
}

// FaceReco 强身份确认后的短时“物理脸轨迹”桥接。
// 目标不是绕过身份识别，而是把一次确认变成连续 PTZ 观测：后续帧仍须真实检测到
// 同一张脸，且其中心/尺寸连续、最近候选明显优于次近候选。FaceReco 仍按原预算
// 持续复核；身份超时、检测断档或脸交错歧义时立即停止，绝不输出冻结框。
bool LightTracker::try_confirmed_face_track_continuity(
    cv::Mat& out_box, const cv::Mat& measured_body_box,
    bool* used_measured_body) {
    if (used_measured_body) *used_measured_body = false;
    const int64_t now_face_ms = now_ms();
    auto reject = [&](const char* reason, float d1 = -1.f, float d2 = -1.f) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_TRACK] f=%d result=reject reason=%s raw=%d"
                " identity_age=%lldms obs_gap=%lldms d1=%.1f d2=%.1f",
                frame_count, reason, (int)recovery_faces_.size(),
                (long long)(last_face_identity_ms_ < 0
                    ? -1 : now_face_ms - last_face_identity_ms_),
                (long long)(last_confirmed_face_ms_ < 0
                    ? -1 : now_face_ms - last_confirmed_face_ms_),
                d1, d2);
            trace_push(line);
            if (std::strcmp(reason, "ambiguous") == 0)
                trace_event_pending_ = true;
        }
        return false;
    };

    if (!face_locked_ || last_face_identity_ms_ < 0)
        return reject("no_identity");
    if (now_face_ms - last_face_identity_ms_ > kFaceTrackIdentityMaxAgeMs)
        return reject("identity_expired");
    if (last_confirmed_face_ms_ < 0
        || now_face_ms - last_confirmed_face_ms_ > kFaceTrackMaxGapMs)
        return reject("observation_gap");
    if (last_confirmed_face_box_.width <= last_confirmed_face_box_.x
        || last_confirmed_face_box_.height <= last_confirmed_face_box_.y)
        return reject("bad_reference");
    if (recovery_faces_.empty()) return reject("no_raw_face");
    if (last_main_bw_ <= 0.f || last_main_bh_ <= 0.f)
        return reject("no_body_history");

    int main_idx = -1;
    for (int i = 0; i < (int)trackers.size(); ++i) {
        if (trackers[i]->get_is_main()) {
            main_idx = i;
            break;
        }
    }
    if (main_idx < 0) return reject("no_main_tracker");

    const float rcx = ((float)last_confirmed_face_box_.x
                     + (float)last_confirmed_face_box_.width) * 0.5f;
    const float rcy = ((float)last_confirmed_face_box_.y
                     + (float)last_confirmed_face_box_.height) * 0.5f;
    const float rw = std::max(1.f, (float)(last_confirmed_face_box_.width
                                        - last_confirmed_face_box_.x));
    const float rh = std::max(1.f, (float)(last_confirmed_face_box_.height
                                        - last_confirmed_face_box_.y));
    const float rdiag = std::hypot(rw, rh);

    int best = -1;
    float d1 = FLT_MAX, d2 = FLT_MAX;
    for (int i = 0; i < (int)recovery_faces_.size(); ++i) {
        const cv::Rect& f = recovery_faces_[i];  // xyxy 塞 Rect
        const float fw = (float)(f.width - f.x);
        const float fh = (float)(f.height - f.y);
        if (fh < kFaceRecogMinFacePx || fw <= 1.f) continue;
        const float fdiag = std::hypot(fw, fh);
        const float size_ratio = fdiag / std::max(1.f, rdiag);
        if (size_ratio < kFaceTrackSizeRatioMin
            || size_ratio > kFaceTrackSizeRatioMax)
            continue;

        const float fcx = ((float)f.x + (float)f.width) * 0.5f;
        const float fcy = ((float)f.y + (float)f.height) * 0.5f;
        const float dist = std::hypot(fcx - rcx, fcy - rcy);
        if (dist > kFaceTrackGateDiag * std::max(rdiag, fdiag)) continue;
        if (dist < d1) {
            d2 = d1;
            d1 = dist;
            best = i;
        } else if (dist < d2) {
            d2 = dist;
        }
    }

    if (best < 0) return reject("no_continuous_face", d1, d2);
    // 两张脸都靠近上一确认位置时，只允许最近脸具有明显距离优势；交错期间宁可
    // 让 FaceReco 决策，也不把检测级连续性跨接到邻人。
    if (d2 < FLT_MAX && d1 > kFaceTrackAmbigRatio * d2)
        return reject("ambiguous", d1, d2);

    const cv::Rect& face = recovery_faces_[best];

    // FACE_TRACK 只是检测框的空间连续性，不能推翻本帧 FaceReco 的明确 mismatch，
    // 也不能在人物交错风险仍存在时用 UNKNOWN 身份接管。所有 permission 判断必须
    // 发生在脸/头/KF/lead/Motion History/真实观测时钟更新之前。
    std::string face_reason;
    IdentityEvidence face_evidence =
        face_identity_evidence_for_box(face, &face_reason);
    const int owner_person = best < (int)recovery_face_owner_person_.size()
                           ? recovery_face_owner_person_[best] : -1;
    const bool owner_ambiguous =
        best < (int)recovery_face_owner_ambiguous_.size()
        && recovery_face_owner_ambiguous_[best] != 0;
    if (owner_ambiguous) {
        note_person_identity_ambiguity(
            kPersonRiskOwnerCompetition,
            person_identity_context_.close_count, false);
    }

    std::string owner_reason;
    IdentityEvidence owner_evidence =
        body_identity_evidence_for_owner(owner_person, &owner_reason);

    const bool scene_risk_before_spatial =
        person_identity_ambiguity_active(now_face_ms);
    if (face_evidence == IdentityEvidence::POSITIVE) {
        trace_continuity_gate(
            "FACE_TRACK", "block", scene_risk_before_spatial, face_evidence,
            owner_evidence, owner_person, "positive_routes_face_identity");
        return reject("identity_positive_route", d1, d2);
    }
    if (face_evidence == IdentityEvidence::NEGATIVE) {
        trace_continuity_gate(
            "FACE_TRACK", "block", scene_risk_before_spatial, face_evidence,
            owner_evidence, owner_person, face_reason.c_str());
        return reject("identity_face_negative", d1, d2);
    }
    if (owner_evidence == IdentityEvidence::NEGATIVE) {
        trace_continuity_gate(
            "FACE_TRACK", "block", scene_risk_before_spatial, face_evidence,
            owner_evidence, owner_person, owner_reason.c_str());
        return reject("identity_owner_negative", d1, d2);
    }

    // 与 HEAD continuity 对齐：落入已确认二级轨迹上部只属于空间冲突。
    // current-corrected 可刷新 scene risk；prediction-only 只阻止当前弱 continuity，
    // 两者都不能把 owner evidence 升级为 NEGATIVE。
    const float selected_fcx = ((float)face.x + (float)face.width) * 0.5f;
    const float selected_fcy = ((float)face.y + (float)face.height) * 0.5f;
    for (auto& st : trackers) {
        if (st->get_is_main() || st->quarantined_) continue;
        if (st->coexist_with_main < kCoexistConfirm) continue;
        if (st->get_time_since_update() > kCoexistVetoMaxTsu) continue;
        if (st->last_update_ms_ >= 0
            && now_face_ms - st->last_update_ms_ > kCoexistVetoMaxMs) continue;
        const SecondaryFrameObservation* current_obs =
            secondary_frame_observation(st->get_id());
        const bool current_corrected = current_obs != nullptr;
        cv::Mat other = current_corrected ? current_obs->box : st->get_state();
        if (other.empty() || other.cols < 4) continue;
        const float ox1 = other.at<float>(0, 0);
        const float oy1 = other.at<float>(0, 1);
        const float ox2 = other.at<float>(0, 2);
        const float oy2 = other.at<float>(0, 3);
        const float oh = oy2 - oy1;
        if (oh <= 0.f || selected_fcx < ox1 || selected_fcx > ox2) continue;
        if (selected_fcy < oy1 - oh * 0.10f
            || selected_fcy > oy1 + oh * 0.45f) continue;
        if (current_corrected) {
            note_person_identity_ambiguity(
                kPersonRiskKnownOther,
                person_identity_context_.close_count, false);
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[KNOWN_OTHER_SPATIAL] f=%d type=FACE_TRACK other=%d"
                " otherObservation=%s identityNegative=0 action=block_continuity",
                frame_count, st->get_id(),
                current_corrected ? "current_corrected" : "prediction_only");
            trace_push(line);
            trace_event_pending_ = true;
        }
        trace_continuity_gate(
            "FACE_TRACK", "block", person_identity_ambiguity_active(now_face_ms),
            face_evidence, owner_evidence, owner_person,
            current_corrected ? "current_known_other_spatial_conflict"
                              : "predicted_known_other_spatial_conflict");
        return reject(current_corrected ? "spatial_current_known_other"
                                        : "spatial_predicted_known_other", d1, d2);
    }

    const bool scene_risk = person_identity_ambiguity_active(now_face_ms);
    if (scene_risk && owner_evidence != IdentityEvidence::POSITIVE) {
        trace_continuity_gate(
            "FACE_TRACK", "block", true, face_evidence,
            owner_evidence, owner_person,
            owner_person < 0 ? "scene_risk_standalone_unknown"
                             : "scene_risk_owner_unknown");
        return reject("identity_scene_risk", d1, d2);
    }
    trace_continuity_gate(
        "FACE_TRACK", "allow", scene_risk, face_evidence,
        owner_evidence, owner_person, "permission_pass");

    cv::Mat body = reconstruct_body_from_face(face);
    if (body.empty()) return reject("body_reconstruct", d1, d2);

    // 有当前真实头时沿用更准确的头身几何；没有头时使用当前脸尺度。
    int head_idx = -1;
    int head_area = std::numeric_limits<int>::max();
    const float fcx = ((float)face.x + (float)face.width) * 0.5f;
    const float fcy = ((float)face.y + (float)face.height) * 0.5f;
    for (int h = 0; h < (int)recovery_heads_.size(); ++h) {
        const cv::Rect& hb = recovery_heads_[h];
        if (fcx < hb.x || fcx > hb.width || fcy < hb.y || fcy > hb.height) continue;
        const int area = std::max(1, hb.width - hb.x)
                       * std::max(1, hb.height - hb.y);
        if (area < head_area) {
            head_area = area;
            head_idx = h;
        }
    }
    if (head_idx >= 0) {
        const cv::Rect& hb = recovery_heads_[head_idx];
        cv::Mat head_box = (cv::Mat_<float>(1, 4) <<
            (float)hb.x, (float)hb.y, (float)hb.width, (float)hb.height);
        cv::Mat head_body = reconstruct_body_from_head(head_box);
        if (!head_body.empty()) body = head_body;
    }
    body = clip_reconstructed_body_to_frame(body, "FACE_TRACK");
    if (body.empty()) return reject("body_clip", d1, d2);

    // 身份证据与几何来源分离：若本帧已经提交了归属明确的真实 BODY，连续脸只
    // 维护物理脸/头轨迹，不再用比例重构框二次校正人体 KF 或覆盖 PTZ 几何。
    // 防御性复核脸体几何；调用方判定与本处不一致时仍走原重构路径。
    bool measured_body_valid = !measured_body_box.empty()
                            && measured_body_box.cols >= 4
                            && cv::checkRange(measured_body_box)
                            && measured_body_box.at<float>(0, 2)
                               > measured_body_box.at<float>(0, 0)
                            && measured_body_box.at<float>(0, 3)
                               > measured_body_box.at<float>(0, 1);
    if (!measured_body_box.empty() && !measured_body_valid)
        return reject("bad_measured_body", d1, d2);
    bool use_measured_body = measured_body_valid
                          && face_body_geometry_consistent(face, measured_body_box);
    // 调用方已经判定 BODY 身份/归属可靠时，物理脸连续性只能确认它，不能凭另一张
    // 空间连续脸反向覆盖强 BODY；真正的人脸身份拉回仍由 FaceReco 仲裁路径负责。
    if (measured_body_valid && !use_measured_body)
        return reject("body_face_mismatch", d1, d2);
    if (use_measured_body) {
        if (used_measured_body) *used_measured_body = true;
        sync_head_track_from_confirmed_face(face, false);
        last_confirmed_face_box_ = face;
        last_confirmed_face_ms_ = now_face_ms;
        last_confirmed_face_frame_ = frame_count;
        face_lock_box_ = measured_body_box.clone();
        out_box = measured_body_box.clone();

        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_TRACK] f=%d result=accept fi=%d head=%d"
                " identity_age=%lldms d1=%.1f d2=%.1f geom=BODY",
                frame_count, best, head_idx,
                (long long)(now_face_ms - last_face_identity_ms_),
                d1, d2 < FLT_MAX ? d2 : -1.f);
            trace_push(line);
            trace_event_pending_ = true;
        }
        return true;
    }

    cv::Mat old_state = trackers[main_idx]->get_state();
    bool relocate = false;
    float jump = 0.f;
    if (!old_state.empty() && old_state.cols >= 4) {
        const float ocx = (old_state.at<float>(0, 0) + old_state.at<float>(0, 2)) * 0.5f;
        const float ocy = (old_state.at<float>(0, 1) + old_state.at<float>(0, 3)) * 0.5f;
        const float bcx = (body.at<float>(0, 0) + body.at<float>(0, 2)) * 0.5f;
        const float bcy = (body.at<float>(0, 1) + body.at<float>(0, 3)) * 0.5f;
        const float ow = std::max(1.f, old_state.at<float>(0, 2)
                                      - old_state.at<float>(0, 0));
        const float oh = std::max(1.f, old_state.at<float>(0, 3)
                                      - old_state.at<float>(0, 1));
        jump = std::hypot(bcx - ocx, bcy - ocy);
        relocate = jump > kFaceRelocateBodyDiag * std::hypot(ow, oh);
    }

    sync_head_track_from_confirmed_face(face, relocate);
    trackers[main_idx]->correct_body_from_part(body, true, relocate);
    last_confirmed_face_box_ = face;
    last_confirmed_face_ms_ = now_face_ms;  // 物理脸观测时间；身份时间戳不在此刷新
    last_confirmed_face_frame_ = frame_count;
    face_lock_box_ = body.clone();
    head_only_since_ms_ = -1;
    update_lead_center(body);
    update_body_reid_search_anchor(body);
    out_box = stabilize_returned_box(body, OutputSource::FACE, relocate);
    if (out_box.empty()) return reject("output_stabilize", d1, d2);
    record_motion_observation(body, MotionObservationSource::FACE_TRACK,
                              current_frame_timestamp_ms_);

    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_TRACK] f=%d result=accept fi=%d head=%d"
            " identity_age=%lldms d1=%.1f d2=%.1f jump=%.1f relocate=%d",
            frame_count, best, head_idx,
            (long long)(now_face_ms - last_face_identity_ms_),
            d1, d2 < FLT_MAX ? d2 : -1.f, jump, relocate ? 1 : 0);
        trace_push(line);
        trace_event_pending_ = true;
    }
    return true;
}

// 头部空间匹配分（标准化中心距，替代裸 IoU）。见 .h 中 kHeadMatchFalloff 说明。
//   分数 = 1 - (头框中心距 / 头部尺寸) / kHeadMatchFalloff，截断到 [0,1]。
//   头部尺寸取两框对角线均值（对长宽比鲁棒）。任一框空/非法/退化 → 返回 0。
float LightTracker::head_match_score(const cv::Mat& head_pred, const cv::Mat& head_cand) const {
    if (head_pred.empty() || head_cand.empty()) return 0.f;
    if (head_pred.cols < 4 || head_cand.cols < 4) return 0.f;

    float px1 = head_pred.at<float>(0, 0), py1 = head_pred.at<float>(0, 1);
    float px2 = head_pred.at<float>(0, 2), py2 = head_pred.at<float>(0, 3);
    float cx1 = head_cand.at<float>(0, 0), cy1 = head_cand.at<float>(0, 1);
    float cx2 = head_cand.at<float>(0, 2), cy2 = head_cand.at<float>(0, 3);

    float pcx = (px1 + px2) * 0.5f, pcy = (py1 + py2) * 0.5f;
    float ccx = (cx1 + cx2) * 0.5f, ccy = (cy1 + cy2) * 0.5f;

    // 头部尺寸 = 两框对角线均值（对长宽比鲁棒）
    float pdiag = std::sqrt((px2 - px1) * (px2 - px1) + (py2 - py1) * (py2 - py1));
    float cdiag = std::sqrt((cx2 - cx1) * (cx2 - cx1) + (cy2 - cy1) * (cy2 - cy1));
    float hsz = 0.5f * (pdiag + cdiag);
    if (hsz < 1e-3f) return 0.f;

    float d = std::sqrt((ccx - pcx) * (ccx - pcx) + (ccy - pcy) * (ccy - pcy));
    float m = 1.f - (d / hsz) / kHeadMatchFalloff;   // 线性衰减，单位=头部尺寸
    return std::max(0.f, std::min(1.f, m));
}

void LightTracker::add_other_det(cv::Mat& result, cv::Mat dets_one, cv::Mat dets_second){
    int new_id = 900;
    std::vector<cv::Mat> extra_rows;
        // 收集 result 中已有的所有检测框 {x1,y1,x2,y2}
    std::set<std::tuple<float, float, float, float>> result_boxes;
    for (int i = 0; i < result.rows; i++) {
        result_boxes.insert({
            result.at<float>(i, 0),
            result.at<float>(i, 1),
            result.at<float>(i, 2),
            result.at<float>(i, 3)
            });
    }

    // 检查 dets_one 中未出现在 result 的框
    for (int i = 0; i < dets_one.rows; i++) {
        auto box = std::make_tuple(
            dets_one.at<float>(i, 0),
            dets_one.at<float>(i, 1),
            dets_one.at<float>(i, 2),
            dets_one.at<float>(i, 3)
        );
        if (result_boxes.find(box) == result_boxes.end()) {
            cv::Mat row = (cv::Mat_<float>(1, 5) <<
                dets_one.at<float>(i, 0),
                dets_one.at<float>(i, 1),
                dets_one.at<float>(i, 2),
                dets_one.at<float>(i, 3),
                new_id++);
            extra_rows.push_back(row);
            result_boxes.insert(box); // 防止 dets_one 和 dets_second 有重复框
        }
    }

    // 检查 dets_second 中未出现在 result 的框
    for (int i = 0; i < dets_second.rows; i++) {
        auto box = std::make_tuple(
            dets_second.at<float>(i, 0),
            dets_second.at<float>(i, 1),
            dets_second.at<float>(i, 2),
            dets_second.at<float>(i, 3)
        );
        if (result_boxes.find(box) == result_boxes.end()) {
            cv::Mat row = (cv::Mat_<float>(1, 5) <<
                dets_second.at<float>(i, 0),
                dets_second.at<float>(i, 1),
                dets_second.at<float>(i, 2),
                dets_second.at<float>(i, 3),
                new_id++);
            extra_rows.push_back(row);
            result_boxes.insert(box);
        }
    }

    // 将新行追加到 result
    for (auto& row : extra_rows) {
        result.push_back(row);
    }

}




bool LightTracker::isFaceFullyInsidePerson(const cv::Rect& face, const cv::Rect& person) {
    return (face.x >= person.x && face.y >= person.y &&
            face.width <= person.width && face.height <= person.height);
}

std::vector<PersonWithFace> LightTracker::matchPersonFaces(
    const std::vector<ObjDetInfo>& detect_list)
{
    // ── 类别约定（新模型）：0=face, 1=body, 2=head ──
    std::vector<ObjDetInfo> persons, faces, heads;
    for (const auto& d : detect_list) {
        if      (d.label == LABEL_BODY) persons.push_back(d);
        else if (d.label == LABEL_FACE) faces.push_back(d);
        else if (d.label == LABEL_HEAD) heads.push_back(d);
    }

    // 只有会进入 dets_one/dets_second 的人体才有资格“拥有”头/脸。此前低于 0.3 的
    // 残缺人体框会先吞掉头脸、随后又被 extract_detections 丢弃，导致身体消失时既没有
    // 人体候选也没有 standalone head/face，云台只能 HOLD。
    constexpr float kSecondaryBodyScore = 0.30f;
    auto body_keeps_parts = [&](size_t idx) {
        return idx < persons.size() && persons[idx].score > kSecondaryBodyScore;
    };

    // 头部连续性（A）/人脸连续性（B）：每帧重建"独立头部/独立人脸"列表（未关联任何人体的框）。
    standalone_heads_.clear();
    standalone_head_scores_.clear();
    standalone_faces_.clear();
    standalone_face_scores_.clear();
    recovery_heads_.clear();
    recovery_head_scores_.clear();
    recovery_head_owner_person_.clear();
    recovery_head_owner_ambiguous_.clear();
    recovery_body_boxes_.clear();
    recovery_body_valid_.clear();
    recovery_body_identity_evidence_.clear();
    recovery_body_identity_reason_.clear();
    recovery_faces_.clear();
    recovery_face_scores_.clear();
    recovery_face_owner_person_.clear();
    recovery_face_owner_ambiguous_.clear();
    recovery_heads_.reserve(heads.size());
    recovery_head_scores_.reserve(heads.size());
    recovery_head_owner_person_.assign(heads.size(), -1);
    recovery_head_owner_ambiguous_.assign(heads.size(), 0);
    recovery_body_boxes_.reserve(persons.size());
    recovery_body_valid_.reserve(persons.size());
    recovery_body_identity_evidence_.assign(
        persons.size(), IdentityEvidence::UNKNOWN);
    recovery_body_identity_reason_.assign(persons.size(), "not_evaluated");
    recovery_faces_.reserve(faces.size());
    recovery_face_scores_.reserve(faces.size());
    recovery_face_owner_person_.assign(faces.size(), -1);
    recovery_face_owner_ambiguous_.assign(faces.size(), 0);
    for (const auto& head : heads) {
        recovery_heads_.push_back(head.box);
        recovery_head_scores_.push_back(head.score);
    }
    for (const auto& face : faces) {
        recovery_faces_.push_back(face.box);
        recovery_face_scores_.push_back(face.score);
    }
    for (size_t i = 0; i < persons.size(); ++i) {
        recovery_body_boxes_.push_back(persons[i].box);
        recovery_body_valid_.push_back(body_keeps_parts(i) ? 1 : 0);
    }

    std::vector<PersonWithFace> result;
    result.reserve(persons.size());
    for (const auto& p : persons) {
        PersonWithFace pf;
        pf.person_box   = p.box;
        pf.person_score = p.score;
        result.push_back(pf);
    }

    if (persons.empty()) {
        // 无人体检测：所有头部/人脸都是独立的（身体被家具全遮挡、仅露头/脸的关键信号）
        for (const auto& head : heads) {
            standalone_heads_.push_back(head.box);
            standalone_head_scores_.push_back(head.score);
        }
        for (const auto& face : faces) {
            standalone_faces_.push_back(face.box);
            standalone_face_scores_.push_back(face.score);
        }
        return result;
    }

    // ── 人脸 → 人体关联：每个人体最多保留一张“上部、居中”的脸 ──
    // 同一人体框被多人/误检脸填满时，旧实现会让后续 FaceKps/FaceReco 对所有脸逐个
    // 推理，既可能把遮挡者的脸混入，也让单帧耗时没有上界。现在只选人体上 55% 内、
    // 最接近人体水平中心且最接近上部脸位的一张；几何相同再比较检测置信度。
    struct FaceChoice {
        bool valid = false;
        cv::Rect box;
        float score = 0.f;
        float geom_cost = FLT_MAX;
    };
    std::vector<FaceChoice> face_choice(persons.size());
    for (size_t face_i = 0; face_i < faces.size(); ++face_i) {
        const auto& face = faces[face_i];
        int best_person = -1;
        int min_area    = std::numeric_limits<int>::max();
        int plausible_owners = 0;

        for (size_t j = 0; j < persons.size(); ++j) {
            if (!body_keeps_parts(j)) continue;
            if (!isFaceFullyInsidePerson(face.box, persons[j].box)) continue;
            ++plausible_owners;

            // cv::Rect 面积 = (x2-x1)*(y2-y1)（本工程 xyxy 语义）
            int area = (persons[j].box.width - persons[j].box.x) * (persons[j].box.height - persons[j].box.y);
            if (area < min_area) {
                min_area    = area;
                best_person = static_cast<int>(j);
            }
        }

        recovery_face_owner_person_[face_i] = best_person;
        recovery_face_owner_ambiguous_[face_i] = plausible_owners > 1 ? 1 : 0;

        if (best_person != -1) {
            const cv::Rect& pb = persons[best_person].box;
            float pw = (float)(pb.width - pb.x);
            float ph = (float)(pb.height - pb.y);
            float fcx = ((float)face.box.x + (float)face.box.width) * 0.5f;
            float fcy = ((float)face.box.y + (float)face.box.height) * 0.5f;
            float pcx = ((float)pb.x + (float)pb.width) * 0.5f;
            if (pw <= 1.f || ph <= 1.f) continue;

            float rel_x = std::fabs(fcx - pcx) / pw;
            float rel_y = (fcy - (float)pb.y) / ph;
            // 人脸应处于身体上部且接近中线；不满足者直接忽略，而非当作独立脸，
            // 因为它仍被一个人体框覆盖，常是前景遮挡者或误检。
            if (rel_y < -0.08f || rel_y > 0.55f || rel_x > 0.45f) continue;

            constexpr float kExpectedFaceCy = 0.20f;
            float geom_cost = 1.5f * rel_x + std::fabs(rel_y - kExpectedFaceCy);
            FaceChoice& choice = face_choice[best_person];
            if (!choice.valid || geom_cost < choice.geom_cost - 1e-4f
                || (std::fabs(geom_cost - choice.geom_cost) <= 1e-4f
                    && face.score > choice.score)) {
                choice.valid = true;
                choice.box = face.box;
                choice.score = face.score;
                choice.geom_cost = geom_cost;
            }
        } else {
            // 未关联到任何人体 → 独立人脸（身体被遮挡、仅露脸；人脸连续性 B 的输入）
            standalone_faces_.push_back(face.box);
            standalone_face_scores_.push_back(face.score);
        }
    }
    for (size_t i = 0; i < face_choice.size(); ++i) {
        if (!face_choice[i].valid) continue;
        result[i].face_box.push_back(face_choice[i].box);
        result[i].face_scores.push_back(face_choice[i].score);
    }

    // ── 头部 → 人体 关联 ──
    //   头部应位于人体上部：要求头部中心水平落在人体框内、纵向位于人体上 45%
    //   （允许略高出框顶以容忍检测噪声）。一个人体取置信度最高的一个头。
    for (size_t head_i = 0; head_i < heads.size(); ++head_i) {
        const auto& head = heads[head_i];
        const cv::Rect& hb = head.box;  // xyxy
        float h_cx = (hb.x + hb.width)  * 0.5f;
        float h_cy = (hb.y + hb.height) * 0.5f;

        int best_person = -1;
        int min_area    = std::numeric_limits<int>::max();
        int plausible_owners = 0;
        for (size_t j = 0; j < persons.size(); ++j) {
            if (!body_keeps_parts(j)) continue;
            const cv::Rect& pb = persons[j].box;  // xyxy
            float p_x1 = (float)pb.x,     p_y1 = (float)pb.y;
            float p_x2 = (float)pb.width, p_y2 = (float)pb.height;
            float p_h  = p_y2 - p_y1;
            if (p_h <= 0) continue;

            if (h_cx < p_x1 || h_cx > p_x2) continue;                 // 水平在框内
            if (h_cy < p_y1 - p_h * 0.10f) continue;                  // 不能高出框顶太多
            if (h_cy > p_y1 + p_h * 0.45f) continue;                  // 必须在上部

            ++plausible_owners;

            int area = (pb.width - pb.x) * (pb.height - pb.y);
            if (area < min_area) { min_area = area; best_person = (int)j; }
        }

        recovery_head_owner_ambiguous_[head_i] = plausible_owners > 1 ? 1 : 0;

        if (best_person != -1) {
            recovery_head_owner_person_[head_i] = best_person;
            if (!result[best_person].has_head ||
                head.score > result[best_person].head_score) {
                result[best_person].head_box   = head.box;
                result[best_person].head_score = head.score;
                result[best_person].has_head   = true;
            }
        } else {
            // 未关联到任何人体 → 独立头部（身体可能被家具遮挡）
            standalone_heads_.push_back(head.box);
            standalone_head_scores_.push_back(head.score);
        }
    }

    return result;
}


/**
* detections : xyxy
* img: frame
*/
std::pair<cv::Mat,int> LightTracker::update(const cv::Mat& img, const cv::Rect& mainBox, const float& z) {
    if (!initialized_) {
        throw TrackerRuntimeError("LightTracker is not initialized");
    }
    // 帧边界：构造清零本帧模型统计，析构（任意 return 分支）打印每帧耗时/调用次数汇总。
    fxprof::ScopedFrame _frame_prof;

    int64_t ts_ms = now_ms();
    // [N9] 帧间隔诊断写入缓冲 trace 文件（不再每帧 UART 直出）。
    int64_t frame_gap_ms = (last_frame_time != 0) ? (ts_ms - last_frame_time) : 0;

    // 帧间隔（秒）：供 M1 把 px/frame 的 KF 速度换算成帧率无关的 px/sec 阈值。
    // 必须在覆盖 last_frame_time 之前算。首帧/断流(>1s 视为异常)回退默认值。
    if (last_frame_time != 0) {
        int64_t dt_ms = ts_ms - last_frame_time;
        frame_dt_sec_ = (dt_ms > 0 && dt_ms < 1000)
                      ? (float)dt_ms / 1000.f : kDefaultDtSec;
    } else {
        frame_dt_sec_ = kDefaultDtSec;
    }

    if(last_frame_time != 0){
        // 超过 3 秒无帧 → 场景切换，重置全部状态
        if ((ts_ms - last_frame_time) > 3000) {
            reset();
        }
    }

    current_frame_timestamp_ms_ = ts_ms;

    last_frame_time = now_ms();

    frame_count++;
    expire_person_identity_ambiguity(ts_ms);
    secondary_frame_observations_.clear();
    prune_relative_motion_history(ts_ms);
    pose_cache_.clear();
    pose_cache_.reserve(kPoseBudgetPerFrame);
    pose_budget_used_ = 0;
    cv::Mat result;
	int person_cnt = 0;
    frame_measurement_reliability_ = MeasurementReliability::NONE;

    // 告警 TTL 必须按“处理帧”推进，而不能藏在仅 matched=true 才会继续的
    // update_quality_monitor() 内。否则连续未匹配正是最需要超时自愈的场景，
    // 却会因为函数提前 return 而让 alert 永久保持。
    if (id_switch_alert_ && alert_start_ms_ > 0
        && (ts_ms - alert_start_ms_) > kAlertTimeoutMs) {
        const int64_t alert_age_ms = ts_ms - alert_start_ms_;
        id_switch_alert_ = false;
        suspect_streak_ = 0;
        alert_start_ms_ = 0;
        null_sink << "[ALERT_TIMEOUT] auto-cleared after " << alert_age_ms
                     << " ms" << std::endl;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[ALERT_GATE] f=%d action=timeout age=%lldms",
                frame_count, (long long)alert_age_ms);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }

    // [N9] 帧间隔诊断写入完整 trace 时间线。
    if (kMatchTrace) {
        char _tl[kTraceLineLen];
        std::snprintf(_tl, sizeof(_tl), "[f%d] frame_dt=%lldms",
                      frame_count, (long long)frame_gap_ms);
        trace_push(_tl);
    }

    // [N12] 过载降级计数：dt 连续偏高进入、连续偏低退出（打断"慢帧→更多推理→更慢"螺旋）。
    //   迟滞带（DtLo~DtHi 之间）打断连续计数但不改模式。
    if (frame_dt_sec_ > kOverloadDtHi) {
        overload_lo_ = 0;
        if (++overload_hi_ >= kOverloadOnN) overload_mode_ = true;
    } else if (frame_dt_sec_ < kOverloadDtLo) {
        overload_hi_ = 0;
        if (++overload_lo_ >= kOverloadOffN) overload_mode_ = false;
    } else {
        overload_hi_ = 0; overload_lo_ = 0;
    }

    // ── 10 FPS 帧级预算初始化 ──
    // 人脸优先与候选 Pose 预算独立。OCC/RECOVERING
    // 不再要求上一帧已经降到 UPPER/HEAD_ONLY：主目标刚躲到多人墙后时，可见度仍
    // 可能冻结在 FULL/HALF，而画面里其它人的人体框又会让 no_body=false。若继续
    // 等低可见度成立，正确的远处脸会长期只有普通 1-slot 预算。
    // 持续危险期不再为 Pose 自动提频。
    int64_t budget_blind_ms = get_blind_ms();
    bool long_or_unknown_blind = !trackers.empty()
                              && (budget_blind_ms < 0
                                  || budget_blind_ms >= kReacqProbationMs);
    bool face_priority_request = !overload_mode_
                              && (id_switch_alert_ || pending_from_sweep_
                                  || face_global_pending_
                                  || face_recovery_fail_streak_ > 0
                                  || occlusion_state_ != OcclusionState::CLEAR
                                  || long_or_unknown_blind);
    face_priority_frame_ = face_priority_request && face_priority_streak_ == 0;
    face_priority_streak_ = face_priority_frame_ ? 1 : 0;
    face_model_budget_ = overload_mode_ ? 1
                        : (face_priority_frame_ ? kFaceBudgetPriority : kFaceBudgetNormal);
    frame_allow_secondary_reid_ = false;
    face_inference_cache_.clear();

    img_h = img.rows;
    img_w = img.cols;

    //null_sink << "main box from camera ,  yeah!!!!!!!!!!: " << mainBox << std::endl;
    //printf("=========[f 9] mainBox.x=%d mainBox.y=%d ======== \n", mainBox.x, mainBox.y); //fxdebug

    if (mainBox.area() > 0) {
        std::tie(result, person_cnt) = setMainTarget(img, mainBox);
        return { result, person_cnt };
    }

    frame_output_source_ = OutputSource::NONE;
    coast_weight_ = 0.f;
    preserve_face_search_state_ = false;
    prefer_body_geometry_output_ = false;

    // [N10] 自运动前馈：按 β·dt 闭合上次输出误差，平移 lead/smooth/emergence 软参考点
    //   （只动软参考，绝不碰 KF）。无有效未闭合误差/超时 → 位移置 0（no-op）。
    apply_ego_feedforward();

    // [N9/N10/N11] 帧收尾闭包：算头锚定瞄准点 + 更新自运动误差/β 采样 + 决策 trace 刷出。
    //   out_box 为本帧真实观测框或有限短时预测框（空 = 无主框输出，云台保持）。
    auto finalize_frame = [&](const cv::Mat& out_box) {
        bool  out_valid = !out_box.empty() && out_box.cols >= 4;
        float ocx = -1.f, ocy = -1.f;
        if (frame_output_source_ == OutputSource::BODY
            || frame_output_source_ == OutputSource::HEAD
            || frame_output_source_ == OutputSource::FACE) {
            if (frozen_prediction_.lifecycle != PredictionLifecycle::IDLE
                && frozen_prediction_.last_output_ms >= 0
                && out_valid && kMatchTrace) {
                const float real_cx = 0.5f * (out_box.at<float>(0, 0)
                                           + out_box.at<float>(0, 2));
                const float real_cy = 0.5f * (out_box.at<float>(0, 1)
                                           + out_box.at<float>(0, 3));
                const float recovery_error = std::hypot(
                    real_cx - frozen_prediction_.last_output_cx,
                    real_cy - frozen_prediction_.last_output_cy);
                const char* real_src = frame_output_source_ == OutputSource::BODY
                                     ? "BODY"
                                     : frame_output_source_ == OutputSource::HEAD
                                     ? "HEAD" : "FACE";
                const int64_t recovery_now = now_ms();
                const int64_t prediction_elapsed_ms = std::max<int64_t>(
                    0, recovery_now - frozen_prediction_.prediction_start_ms);
                const int64_t motion_age_ms = std::max<int64_t>(
                    0, recovery_now - frozen_prediction_.anchor_timestamp_ms);
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[SHORT_PRED_RECOVERY] f=%d src=%s lifecycle=%s"
                    " anchor_ts=%lld start_ts=%lld motion_age=%lldms"
                    " pred_elapsed=%lldms mode=%s"
                    " pred_center=(%.1f,%.1f) real_center=(%.1f,%.1f) error=%.1f",
                    frame_count, real_src,
                    frozen_prediction_.lifecycle == PredictionLifecycle::ACTIVE
                        ? "ACTIVE" : "EXHAUSTED",
                    (long long)frozen_prediction_.anchor_timestamp_ms,
                    (long long)frozen_prediction_.prediction_start_ms,
                    (long long)motion_age_ms, (long long)prediction_elapsed_ms,
                    prediction_mode_name(frozen_prediction_.mode),
                    frozen_prediction_.last_output_cx,
                    frozen_prediction_.last_output_cy,
                    real_cx, real_cy, recovery_error);
                trace_push(line);
                trace_event_pending_ = true;
            }
            clear_short_prediction("real_observation");
        }
        if (out_valid) {
            ocx = (out_box.at<float>(0, 0) + out_box.at<float>(0, 2)) * 0.5f;
            ocy = (out_box.at<float>(0, 1) + out_box.at<float>(0, 3)) * 0.5f;
            note_returned_box(out_box);
        }
        compute_aim_point(out_box);                                        // [N11]
        note_output_for_ego(out_valid, ocx, ocy,
                            frame_output_source_ == OutputSource::BODY);   // [N10]
        if (kMatchTrace) {
            const char* src = frame_output_source_ == OutputSource::BODY ? "BODY"
                            : frame_output_source_ == OutputSource::HEAD ? "HEAD"
                            : frame_output_source_ == OutputSource::FACE ? "FACE"
                            : frame_output_source_ == OutputSource::PREDICTED ? "PRED" : "NONE";
            const char* meas =
                frame_measurement_reliability_ == MeasurementReliability::RELIABLE ? "REL"
              : frame_measurement_reliability_ == MeasurementReliability::UNCERTAIN ? "UNC"
                                                                                     : "NONE";
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[OUTPUT] f=%d src=%s meas=%s valid=%d weight=%.2f"
                " box=(%.0f,%.0f,%.0f,%.0f)",
                frame_count, src, meas, out_valid ? 1 : 0, coast_weight_,
                out_valid ? out_box.at<float>(0, 0) : -1.f,
                out_valid ? out_box.at<float>(0, 1) : -1.f,
                out_valid ? out_box.at<float>(0, 2) : -1.f,
                out_valid ? out_box.at<float>(0, 3) : -1.f);
            trace_push(line);
        }
        bool has_main_target = false;
        for (const auto& trk : trackers) {
            if (trk && trk->get_is_main()) {
                has_main_target = true;
                break;
            }
        }
        trace_prediction_coverage(has_main_target);
        // [N9] 硬事件（遮挡状态切换/警报翻转）立即 flush；软事件（未匹配/暂定）
        //   按最小间隔 flush；否则按心跳节流。文件本身已经连续保留全部前后文。
        bool alert_now  = id_switch_alert_;
        bool hard_event = (occlusion_state_ != trace_prev_occ_) || (alert_now != trace_prev_alert_);
        bool soft_event = trace_event_pending_ || (!out_valid && !trackers.empty());
        if (hard_event) {
            trace_flush("state", true);
        } else if (soft_event && (frame_count - trace_last_dump_frame_) >= kTraceDumpMinGapFrames) {
            trace_flush("event", true);
        } else if ((frame_count - trace_last_dump_frame_) >= kTraceHeartbeatFrames) {
            trace_flush("hb", false);
        }
        trace_prev_occ_      = occlusion_state_;
        trace_prev_alert_    = alert_now;
        trace_event_pending_ = false;
    };

    std::vector<ObjDetInfo> detect_list;
    std::vector<PersonWithFace> PersonFace;

    if (detector.run(img, 0.2, detect_list) != 0) {
        frame_output_source_ = OutputSource::NONE;
        coast_weight_ = 0.f;
        ERROR_LOG("Detector run failed; frame is not treated as an empty detection");
        throw TrackerRuntimeError("Detector inference failed");
    }
    
    PersonFace = matchPersonFaces(detect_list);

    auto det_groups = extract_detections(PersonFace);
    // person_cnt 表示本帧进入跟踪候选体系的人体检测总数：高分框与低分框互斥，
    // 因此直接相加即可。统一在分组完成后赋值，保证后续所有提前返回分支语义一致。
    person_cnt = det_groups.dets_one.rows + det_groups.dets_second.rows;

    int viable_recovery_face_count = 0;
    float min_recovery_face_score = 1.f;
    float max_recovery_face_score = 0.f;
    for (int i = 0; i < (int)recovery_faces_.size(); ++i) {
        const cv::Rect& f = recovery_faces_[i];
        if ((float)(f.height - f.y) >= kFaceRecogMinFacePx)
            ++viable_recovery_face_count;
        if (i < (int)recovery_face_scores_.size()) {
            min_recovery_face_score =
                std::min(min_recovery_face_score, recovery_face_scores_[i]);
            max_recovery_face_score =
                std::max(max_recovery_face_score, recovery_face_scores_[i]);
        }
    }
    if (recovery_faces_.empty())
        min_recovery_face_score = 0.f;

    int associated_high_faces = 0;
    for (const auto& faces : det_groups.dets_one_face)
        associated_high_faces += faces.rows;
    int associated_low_faces = 0;
    for (const auto& faces : det_groups.dets_second_face)
        associated_low_faces += faces.rows;
    if (kMatchTrace && !trackers.empty()) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_POOL] f=%d raw=%d viable=%d standalone=%d"
            " assoc_hi=%d assoc_lo=%d score=%.2f..%.2f det_thr=0.20",
            frame_count, (int)recovery_faces_.size(), viable_recovery_face_count,
            (int)standalone_faces_.size(), associated_high_faces, associated_low_faces,
            min_recovery_face_score, max_recovery_face_score);
        trace_push(line);
    }

    // 入口只能依据上一帧状态申请优先帧；本帧 detector 证明没有任何脸时立即退回
    // 普通预算，让 assign_cascade 正常刷新 Pose。小于识别尺寸的脸虽然保留在原始池
    // 供 trace 解释，但无法产生可靠 embedding，也不能为它们白白跳过 Pose。
    if (face_priority_frame_ && viable_recovery_face_count == 0) {
        face_priority_frame_ = false;
        face_priority_streak_ = 0;
        face_model_budget_ = kFaceBudgetNormal;
    }

    if (kMatchTrace && face_priority_frame_ && !recovery_faces_.empty()) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_BUDGET] f=%d mode=priority budget=%d faces=%d st=%d blind=%lldms",
            frame_count, face_model_budget_, (int)recovery_faces_.size(),
            (int)occlusion_state_, (long long)get_blind_ms());
        trace_push(line);
    }

    // ── 无身体的身份优先帧 ───────────────────────────────────────
    // 画面内所有身体完全不可见、却检测到独立脸时，不能等待 OCCLUDED/HEAD_ONLY
    // 等迟滞状态成立才开始做人脸识别。脸是这类帧唯一的强身份证据：本帧直接
    // 预留全部独立脸配额，随后由 try_face_only_continuity 逐帧做 FaceKps +
    // FaceReco；FaceKps 质量门会滤掉侧脸/残缺脸。零人体帧不运行 Pose/ReID，
    // 因而最多三张脸仍在 10 FPS 预算内。过载模式保留一张最近脸的恢复机会。
    const bool no_body_detections = det_groups.dets_one.rows == 0
                                 && det_groups.dets_second.rows == 0;
    const bool head_face_identity_frame = no_body_detections
                                       && !recovery_faces_.empty();
    if (head_face_identity_frame && !overload_mode_) {
        face_model_budget_ = std::max(face_model_budget_, kFaceOnlyMaxCand);
        null_sink << "[FACE_PRIORITY] no-body frame: recovery_faces="
                     << recovery_faces_.size()
                     << " heads=" << recovery_heads_.size()
                     << " budget=" << face_model_budget_ << std::endl;
    }

    if (no_body_detections) {
        if(trackers.size() != 0){
            auto info = get_predicted_tracks();
            cleanup_expired_trackers();

            // ── GMC：即使本帧零检测也要推进相机运动补偿 ──
            //   estimate_camera_motion 是 prev_gray_ 唯一的更新点。若零检测帧
            //   跳过它，prev_gray_ 会停在空档前一帧 → 检测恢复后 GMC 跨越整个
            //   空档误估（位移过大被判异常/或被当作单帧补偿）。同时头部连续性
            //   用的头部预测框会停留在旧坐标系，与当前帧独立头检测错位。
            //   故此处调用 GMC 推进 prev_gray_，并把主目标头部预测框 + 平滑中心
            //   补偿到当前帧坐标系（与 assign_cascade 中的补偿一致）。
            if (gmc_enabled_) {
                cv::Mat gmc_M = estimate_camera_motion(img, info.trks);
                if (!gmc_M.empty()) {
                    for (auto& trk : trackers) {
                        if (trk->get_is_main() && trk->has_head_track()) {
                            cv::Mat hp = trk->get_head_pred_box();
                            if (!hp.empty()) {
                                warp_box_inplace(hp, gmc_M);
                                trk->set_head_pred_box(hp);
                            }
                        }
                    }
                    if (smooth_cx_ > 0.f && smooth_cy_ > 0.f) {
                        cv::Point2f nc = warp_point(smooth_cx_, smooth_cy_, gmc_M);
                        smooth_cx_ = nc.x; smooth_cy_ = nc.y;
                    }
                    if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
                        cv::Point2f lc = warp_point(lead_cx_, lead_cy_, gmc_M);
                        lead_cx_ = lc.x; lead_cy_ = lc.y;
                    }
                    if (last_confirmed_face_ms_ >= 0) {
                        cv::Mat fb = (cv::Mat_<float>(1, 4) <<
                            (float)last_confirmed_face_box_.x,
                            (float)last_confirmed_face_box_.y,
                            (float)last_confirmed_face_box_.width,
                            (float)last_confirmed_face_box_.height);
                        warp_box_inplace(fb, gmc_M);
                        last_confirmed_face_box_ = cv::Rect(
                            (int)std::lround(fb.at<float>(0, 0)),
                            (int)std::lround(fb.at<float>(0, 1)),
                            (int)std::lround(fb.at<float>(0, 2)),
                            (int)std::lround(fb.at<float>(0, 3)));
                    }
                    if (face_global_pending_) {
                        cv::Point2f fp = warp_point(face_global_pending_cx_,
                                                    face_global_pending_cy_, gmc_M);
                        face_global_pending_cx_ = fp.x;
                        face_global_pending_cy_ = fp.y;
                    }
                    // KF 状态级补偿（与 assign_cascade 一致，使补偿逐帧累计）
                    for (auto& trk : trackers)
                        trk->apply_camera_motion(gmc_M);
                }
            }

            // 人脸连续性（B）：身体无检测但露出独立人脸 → 识别命中即按当前脸尺度重建体框维持。
            // 人脸是强身份证据，优先于纯空间的 head-only；额外耗时由内部 top-K 预算控制。
            // 零人体检测时不会运行 Pose/ReID。上方“身份优先帧”已预留独立脸预算；
            // 这里仍只消费该共享预算，不与其它人脸路径叠加。
            cv::Mat fc_box;
            bool face_identity_hit = try_face_only_continuity(img, fc_box);
            bool face_track_hit = !face_identity_hit
                               && try_confirmed_face_track_continuity(fc_box);
            if (face_identity_hit || face_track_hit) {
                frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
                last_real_obs_ms_ = ts_ms;    // 真实人脸检测（身份确认或安全桥接）→ 刷新盲跟时钟
                head_reacq_pending_streak_ = 0;
                head_reacq_pending_ms_ = -1;
                coast_weight_ = kObservedPartWeight;
                frame_output_source_ = OutputSource::FACE;
                null_sink << (face_identity_hit
                    ? "[FACE_ONLY] no-det: recovered by independent face"
                    : "[FACE_TRACK] no-det: continued confirmed physical face")
                             << std::endl;
                cv::Mat _mr = build_main_row(fc_box);
                finalize_frame(_mr);
                return { _mr, person_cnt };
            }

            // 头部连续性（A）：身体全无检测，但若独立头部仍在头部 KF 预测附近，
            // 说明主目标只是身体被家具遮挡 → 由头部重建身体框维持跟踪，避免盲丢。
            cv::Mat hc_box;
            if (try_head_continuity(hc_box)) {
                frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
                try_register_face_from_confirmed_head(img);
                last_real_obs_ms_ = ts_ms;    // [N1] 真实头部观测 → 刷新盲跟时钟
                coast_weight_ = kObservedPartWeight;
                frame_output_source_ = OutputSource::HEAD;
                null_sink << "[HEAD] continuity (no body det): head-tracked" << std::endl;
                cv::Mat _mr = build_main_row(hc_box);
                finalize_frame(_mr);
                return { _mr, person_cnt };   // 真实头部观测 → 真值输出
            }

            // 丢失后的 PTZ blind slide 由外部 PTZ 控制层使用最后真实框独立完成。
            // Tracker 不再输出 image-space PRED，避免两层同时驱动云台。
            clear_short_prediction("external_ptz_blind");
            null_sink << "[HOLD] no-det blind=" << get_blind_ms()
                      << "ms (external_ptz_blind)" << std::endl;
        }

        coast_weight_ = 0.f;   // 本帧无主框输出
        finalize_frame(cv::Mat());
        return { cv::Mat::zeros(0, 5, CV_32F), person_cnt };
    }

    null_sink << "dets one size : " << det_groups.dets_one.rows << " dets second size : " << det_groups.dets_second.rows << std::endl;

    if(trackers.size() == 0 ){
        add_other_det(result, det_groups.dets_one, det_groups.dets_second);
        finalize_frame(cv::Mat());   // 无主目标 → 无输出（ego/aim 置空 + trace 刷）
        return {result, person_cnt};
    }

    result = assign_cascade(det_groups, img);

    bool got_main = false;
    cv::Mat out_box;   // 本帧真实观测框或有限短时预测框（空 = 无输出）
    int main_result_row = -1;
    for (int i = 0; i < result.rows; ++i) {
        if (static_cast<int>(result.at<float>(i, 4)) == main_id + 1) {
            main_result_row = i;
            break;
        }
    }

    // BODY 尚未被本帧人脸身份解释时继续维护确认脸轨迹。若该脸能明确归属于
    // 当前 BODY，则保留真实人体几何；只有 BODY 归属仍不明确时才让重构框覆盖。
    // 身份观察和几何输出分开，且不新增模型调用。
    cv::Mat face_track_override_box;
    cv::Mat measured_body_box;
    if (main_result_row >= 0 && prefer_body_geometry_output_)
        measured_body_box = result.row(main_result_row).colRange(0, 4).clone();
    bool face_track_used_body = false;
    bool face_track_continues = main_result_row >= 0
                             && preserve_face_search_state_
                             && try_confirmed_face_track_continuity(
                                    face_track_override_box, measured_body_box,
                                    &face_track_used_body);
    bool face_track_override = face_track_continues
                            && !face_track_used_body;

    if (main_result_row >= 0) {
        got_main = true;
        last_real_obs_ms_ = ts_ms;
        head_reacq_pending_streak_ = 0;
        head_reacq_pending_ms_ = -1;

        if (face_track_override) {
            for (int c = 0; c < 4; ++c)
                result.at<float>(main_result_row, c) =
                    face_track_override_box.at<float>(0, c);
            out_box = face_track_override_box.clone();
            coast_weight_ = kObservedPartWeight;
            frame_output_source_ = OutputSource::FACE;
            head_only_since_ms_ = -1;
            null_sink << "[FACE_TRACK] overrode unresolved BODY output" << std::endl;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_TRACK] f=%d action=override_unverified_body row=%d",
                    frame_count, main_result_row);
                trace_push(line);
                trace_event_pending_ = true;
            }
        } else {
            if (face_track_continues && face_track_used_body
                && kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[BOX_ARB] f=%d action=keep_body reason=face_confirms_owner row=%d",
                    frame_count, main_result_row);
                trace_push(line);
                trace_event_pending_ = true;
            }
            // CLEAR 的真实身体结束找脸周期；但未完成人脸仲裁的 BODY 不能清空 rotor，
            // 否则普通 1-slot 会永远重复扫描旧位置最近脸。
            if (occlusion_state_ == OcclusionState::CLEAR
                && !preserve_face_search_state_) {
                face_recovery_rotor_ = 0;
                face_recovery_fail_streak_ = 0;
                face_recovery_hypotheses_.clear();
                next_face_recovery_hyp_id_ = 1;
                face_global_pending_ = false;
                face_global_pending_streak_ = 0;
                face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
                face_global_pending_ms_ = -1;
            }
            last_main_bw_ = result.at<float>(main_result_row, 2)
                          - result.at<float>(main_result_row, 0);
            last_main_bh_ = result.at<float>(main_result_row, 3)
                          - result.at<float>(main_result_row, 1);
            cv::Mat raw_body_out = result.row(main_result_row).colRange(0, 4).clone();
            update_lead_center(raw_body_out);
            out_box = stabilize_returned_box(raw_body_out,
                                             OutputSource::BODY, false);
            for (int c = 0; c < 4; ++c)
                result.at<float>(main_result_row, c) = out_box.at<float>(0, c);
            coast_weight_ = 1.0f;
            frame_output_source_ = OutputSource::BODY;
            head_only_since_ms_ = -1;
            if (frame_measurement_reliability_ == MeasurementReliability::RELIABLE) {
                record_motion_observation(raw_body_out,
                                          MotionObservationSource::BODY,
                                          ts_ms);
            }
        }
    }

    if (!got_main) {
        // 人脸连续性（B）优先：身体未匹配但独立脸识别命中 → 用脸重建体框。
        // 这是强身份证据；失败后再回落到纯空间的头部连续性。
        cv::Mat hc_box, fc_box;
        bool face_identity_hit = try_face_only_continuity(img, fc_box);
        bool face_track_hit = !face_identity_hit
                           && try_confirmed_face_track_continuity(fc_box);
        if (face_identity_hit || face_track_hit) {
            frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
            last_real_obs_ms_ = ts_ms;      // 真实人脸检测（身份确认或安全桥接）→ 刷新盲跟时钟
            head_reacq_pending_streak_ = 0;
            head_reacq_pending_ms_ = -1;
            coast_weight_ = kObservedPartWeight;
            frame_output_source_ = OutputSource::FACE;
            null_sink << (face_identity_hit
                ? "[FACE_ONLY] unmatched: recovered by independent face"
                : "[FACE_TRACK] unmatched: continued confirmed physical face")
                         << std::endl;
            cv::Mat _fr = build_main_row(fc_box);
            out_box = _fr.colRange(0, 4).clone();
            result.push_back(_fr);
        } else if (try_head_continuity(hc_box)) {
            frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
            try_register_face_from_confirmed_head(img);
            // 头部连续性（A）：身体未匹配，但若独立头部仍在头部 KF 预测附近 →
            // 由头部重建身体框维持跟踪（引导搜索门），避免盲丢。
            last_real_obs_ms_ = ts_ms;      // [N1] 真实头部观测 → 刷新盲跟时钟
            coast_weight_ = kObservedPartWeight;
            frame_output_source_ = OutputSource::HEAD;
            null_sink << "[HEAD] continuity: body occluded, head-tracked" << std::endl;
            cv::Mat _hr = build_main_row(hc_box);
            out_box = _hr.colRange(0, 4).clone();
            result.push_back(_hr);          // 真实头部观测 → 真值输出
        } else {
            clear_short_prediction("external_ptz_blind");
            coast_weight_ = 0.f;        // 本帧无主框输出
            null_sink << "[HOLD] unmatched blind=" << get_blind_ms()
                      << "ms (external_ptz_blind)" << std::endl;
        }
    }

    add_other_det(result, det_groups.dets_one, det_groups.dets_second);

    finalize_frame(out_box);   // [N9/N10/N11] 帧收尾

    null_sink << "result: " << result << std::endl;
    null_sink << std::endl;
    return { result, person_cnt };
}


LightTracker::DetectionGroups
LightTracker::extract_detections(const std::vector<PersonWithFace>& output_results) {
    DetectionGroups groups;
    // null_sink << "Extracting detections into groups..." << std::endl;
    if (output_results.empty()) {
        groups.dets_one = cv::Mat::zeros(0, 5, CV_32F);
        groups.dets_second = cv::Mat::zeros(0, 5, CV_32F);
        groups.dets_one_head = cv::Mat::zeros(0, 5, CV_32F);
        groups.dets_second_head = cv::Mat::zeros(0, 5, CV_32F);
        return groups;
    }

    std::vector<cv::Rect> high_boxes;
    std::vector<float> high_scores;
    std::vector<std::vector<cv::Rect>> high_faces;
    std::vector<std::vector<float>> high_face_scores;
    std::vector<cv::Rect> high_heads;
    std::vector<float> high_head_scores;
    std::vector<char>  high_head_has;

    std::vector<cv::Rect> low_boxes;
    std::vector<float> low_scores;
    std::vector<std::vector<cv::Rect>> low_faces;
    std::vector<std::vector<float>> low_face_scores;
    std::vector<cv::Rect> low_heads;
    std::vector<float> low_head_scores;
    std::vector<char>  low_head_has;

    for (const auto& person : output_results) {
        if (person.person_score > det_thresh) {
            high_boxes.push_back(person.person_box);
            high_scores.push_back(person.person_score);
            high_faces.push_back(person.face_box);
            high_face_scores.push_back(person.face_scores);
            high_heads.push_back(person.head_box);
            high_head_scores.push_back(person.head_score);
            high_head_has.push_back(person.has_head ? 1 : 0);
        } else if (person.person_score > 0.3f) { // 低阈值可配置
            low_boxes.push_back(person.person_box);
            low_scores.push_back(person.person_score);
            low_faces.push_back(person.face_box);
            low_face_scores.push_back(person.face_scores);
            low_heads.push_back(person.head_box);
            low_head_scores.push_back(person.head_score);
            low_head_has.push_back(person.has_head ? 1 : 0);
        }
    }

    // 处理高置信度组
    int num_high = (int)high_boxes.size();
    if (num_high > 0) {
        groups.dets_one = cv::Mat(num_high, 5, CV_32F);
        groups.dets_one_face.resize(num_high);
        for (int i = 0; i < num_high; ++i) {
            const auto& box = high_boxes[i];
            groups.dets_one.at<float>(i, 0) = (float)box.x;
            groups.dets_one.at<float>(i, 1) = (float)box.y;
            groups.dets_one.at<float>(i, 2) = (float)box.width;
            groups.dets_one.at<float>(i, 3) = (float)box.height;
            groups.dets_one.at<float>(i, 4) = high_scores[i];

            // 构造该人体框对应的多人脸 Mat
            const auto& faces = high_faces[i];
            int num_faces = (int)faces.size();
            if (num_faces > 0) {
                cv::Mat face_mat(num_faces, 5, CV_32F);
                for (int f = 0; f < num_faces; ++f) {
                    face_mat.at<float>(f, 0) = (float)faces[f].x;
                    face_mat.at<float>(f, 1) = (float)faces[f].y;
                    face_mat.at<float>(f, 2) = (float)faces[f].width;
                    face_mat.at<float>(f, 3) = (float)faces[f].height;
                    face_mat.at<float>(f, 4) = high_face_scores[i][f]; // 可选：如果需要存储人脸置信度，可以在这里添加一列
                }
                groups.dets_one_face[i] = face_mat;
            } else {
                groups.dets_one_face[i] = cv::Mat::zeros(0, 5, CV_32F);
            }
        }

        // 头部框（与 dets_one 行对齐；无头则该行全 0）
        groups.dets_one_head = cv::Mat::zeros(num_high, 5, CV_32F);
        for (int i = 0; i < num_high; ++i) {
            if (high_head_has[i]) {
                groups.dets_one_head.at<float>(i, 0) = (float)high_heads[i].x;
                groups.dets_one_head.at<float>(i, 1) = (float)high_heads[i].y;
                groups.dets_one_head.at<float>(i, 2) = (float)high_heads[i].width;
                groups.dets_one_head.at<float>(i, 3) = (float)high_heads[i].height;
                groups.dets_one_head.at<float>(i, 4) = high_head_scores[i];
            }
        }
    } else {
        groups.dets_one = cv::Mat::zeros(0, 5, CV_32F);
        groups.dets_one_head = cv::Mat::zeros(0, 5, CV_32F);
    }

    // 处理低置信度组
    int num_low = (int)low_boxes.size();
    if (num_low > 0) {
        groups.dets_second = cv::Mat(num_low, 5, CV_32F);
        groups.dets_second_face.resize(num_low);
        for (int i = 0; i < num_low; ++i) {
            const auto& box = low_boxes[i];
            groups.dets_second.at<float>(i, 0) = (float)box.x;
            groups.dets_second.at<float>(i, 1) = (float)box.y;
            groups.dets_second.at<float>(i, 2) = (float)box.width;
            groups.dets_second.at<float>(i, 3) = (float)box.height;
            groups.dets_second.at<float>(i, 4) = low_scores[i];

            const auto& faces = low_faces[i];
            int num_faces = (int)faces.size();
            if (num_faces > 0) {
                cv::Mat face_mat(num_faces, 5, CV_32F);
                for (int f = 0; f < num_faces; ++f) {
                    face_mat.at<float>(f, 0) = (float)faces[f].x;
                    face_mat.at<float>(f, 1) = (float)faces[f].y;
                    face_mat.at<float>(f, 2) = (float)faces[f].width;
                    face_mat.at<float>(f, 3) = (float)faces[f].height;
                    face_mat.at<float>(f, 4) = low_face_scores[i][f]; // 可选：如果需要存储人脸置信度，可以在这里添加一列
                }
                groups.dets_second_face[i] = face_mat;
            } else {
                groups.dets_second_face[i] = cv::Mat::zeros(0, 5, CV_32F);
            }
        }

        // 头部框（与 dets_second 行对齐；无头则该行全 0）
        groups.dets_second_head = cv::Mat::zeros(num_low, 5, CV_32F);
        for (int i = 0; i < num_low; ++i) {
            if (low_head_has[i]) {
                groups.dets_second_head.at<float>(i, 0) = (float)low_heads[i].x;
                groups.dets_second_head.at<float>(i, 1) = (float)low_heads[i].y;
                groups.dets_second_head.at<float>(i, 2) = (float)low_heads[i].width;
                groups.dets_second_head.at<float>(i, 3) = (float)low_heads[i].height;
                groups.dets_second_head.at<float>(i, 4) = low_head_scores[i];
            }
        }
    } else {
        groups.dets_second = cv::Mat::zeros(0, 5, CV_32F);
        groups.dets_second_head = cv::Mat::zeros(0, 5, CV_32F);
    }

    return groups;
}


cv::Mat LightTracker::compute_embedding(const cv::Mat& img,
    const cv::Mat& bbox) {
    cv::Mat reid_feature;
    cv::Rect box(bbox.at<float>(0, 0), bbox.at<float>(0, 1),
        bbox.at<float>(0, 2), bbox.at<float>(0, 3));
    emb_model.run(img, box, reid_feature);
    return reid_feature;
}

TrackerInfo LightTracker::get_predicted_tracks() {
    TrackerInfo info;

    if (trackers.empty()) {
        return info;
    }
    // 预分配并全零初始化。正常情况下 KalmanBoxTracker 会先恢复有限快照；若快照和
    // 强观测都不可用，失败轨迹仍保持零行，绝不让未初始化内存或 NaN 进入下游 IoU。
    info.trks           = cv::Mat::zeros((int)trackers.size(), 4, CV_32F);
    info.velocities     = cv::Mat::zeros((int)trackers.size(), 2, CV_32F);
    info.speeds         = cv::Mat::zeros((int)trackers.size(), 1, CV_32F);
    info.last_boxes     = cv::Mat::zeros((int)trackers.size(), 4, CV_32F);
    info.k_observations = cv::Mat::zeros((int)trackers.size(), 4, CV_32F);
    const int64_t blind_ms = get_body_blind_ms();
    for (int t = 0; t < trackers.size(); ++t) {
        trackers[t]->set_frame_interval(frame_dt_sec_);
        trackers[t]->set_long_coast(
            trackers[t]->get_is_main()
            && blind_ms >= kKfLongBlindFreezeMs);
        cv::Mat pred = trackers[t]->predict();
        // 头部轨迹与人体 KF 同帧推进（仅主目标维护头部轨迹）
        if (trackers[t]->get_is_main())
            trackers[t]->predict_head();

        // 数值异常必须进入与 MATCH 相同的文件 trace，不能只写已编译为空操作的
        // null_sink；这样现场可区分“身份门拒绝”和“KF 已恢复/重建”。
        for (const std::string& event : trackers[t]->consume_numerical_events()) {
            const char* tag = "[KF_EVENT]";
            const char* detail = event.c_str();
            if (event.rfind("NAN ", 0) == 0) {
                tag = "[KF_NAN]";
                detail += 4;
            } else if (event.rfind("RESET ", 0) == 0) {
                tag = "[KF_RESET]";
                detail += 6;
            } else if (event.rfind("REJECT ", 0) == 0) {
                tag = "[KF_REJECT]";
                detail += 7;
            }
            char line[240];
            std::snprintf(line, sizeof(line),
                          "%s f=%d id=%d main=%d %s",
                          tag, frame_count, trackers[t]->get_id(),
                          trackers[t]->get_is_main() ? 1 : 0, detail);
            trace_push(line);
            trace_event_pending_ = true;
        }

        const bool pred_ok = !pred.empty() && pred.type() == CV_32F
            && pred.rows >= 1 && pred.cols >= 4
            && cv::checkRange(pred.row(0).colRange(0, 4),
                              true, nullptr, -FLT_MAX, FLT_MAX)
            && pred.at<float>(0, 2) - pred.at<float>(0, 0) >= 1.f
            && pred.at<float>(0, 3) - pred.at<float>(0, 1) >= 1.f;
        if (pred_ok) {
            int idx = t;
            pred.row(0).colRange(0, 4).copyTo(info.trks.row(idx));
            trackers[t]->get_velocity().row(0).copyTo(info.velocities.row(idx));
            info.speeds.at<float>(idx, 0) = trackers[t]->get_speed();
            trackers[t]->get_last_observation().row(0).copyTo(info.last_boxes.row(idx));
            trackers[t]->get_last_observation().copyTo(info.k_observations.row(idx));
        } else {
            char line[192];
            std::snprintf(line, sizeof(line),
                          "[KF_NAN] f=%d id=%d main=%d stage=output_decode row_zeroed=1",
                          frame_count, trackers[t]->get_id(),
                          trackers[t]->get_is_main() ? 1 : 0);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }

    return info;
}

MainNonMainSplit
LightTracker::split_main_nomain(const cv::Mat& trks) {
    MainNonMainSplit split;
    //null_sink << "in split" << std::endl;
    for (int i = 0; i < trackers.size(); ++i) {
        if (trackers[i]->get_is_main())
            split.main_idx.push_back(i);
        else
            split.nm_idx.push_back(i);
    }

    if (!split.main_idx.empty()) {
        split.main_trks = cv::Mat(1, 4, CV_32F);
        trks.row(split.main_idx[0]).copyTo(split.main_trks.row(0));
    }

    if (!split.nm_idx.empty()) {
        split.nm_trks = cv::Mat(split.nm_idx.size(), 4, CV_32F);
        for (size_t i = 0; i < split.nm_idx.size(); ++i) {
            trks.row(split.nm_idx[i]).copyTo(split.nm_trks.row(i));
        }
    }

    return split;
}


std::vector<float> LightTracker::get_kps10(std::vector<FaceKeypoint> kps_106) {
	std::vector<float> kps_10;
	// 崩溃防线（A10）：Facekps 任一失败路径（未初始化/预处理失败/NPU 推理失败/后处理失败）
	// 返回空 points；旧实现无检查直接索引 [87] —— 空 vector 的 operator[] 是未定义行为
	// （data() 常为 nullptr → 近空指针解引用 → SIGSEGV）。段错误不是 C++ 异常，
	// fx_tracker_run 的 try/catch 拦不住 → 进程死 → 看门狗整机重启。
	// 返回空 vector，调用方按"本帧无人脸关键点"处理（识别/注册跳过）。
	if (kps_106.size() < 106) return kps_10;
	kps_10.push_back(kps_106[37].x); kps_10.push_back(kps_106[37].y);
	kps_10.push_back(kps_106[87].x); kps_10.push_back(kps_106[87].y);
	kps_10.push_back(kps_106[86].x); kps_10.push_back(kps_106[86].y);
	kps_10.push_back(kps_106[65].x); kps_10.push_back(kps_106[65].y);
	kps_10.push_back(kps_106[61].x); kps_10.push_back(kps_106[61].y);
	return kps_10;
}

// 一个 slot 对应一次 FaceKps，且其成功后紧随一次 FaceReco。近场验证、全画面
// 扫描、独立脸恢复和延迟注册均经此处扣减，保证它们不能在同一帧互相叠加。
bool LightTracker::take_face_model_slot() {
    if (face_model_budget_ <= 0) return false;
    --face_model_budget_;
    return true;
}

std::pair<float, std::string> LightTracker::face_recognition_inference(const cv::Mat& face_boxes, cv::Mat img) {
    float best_sim = 0.f;
    std::string best_name = "unknown";

    for(int i = 0; i < face_boxes.rows; i++) {
        cv::Rect box(face_boxes.at<float>(i, 0), face_boxes.at<float>(i, 1), face_boxes.at<float>(i, 2), face_boxes.at<float>(i, 3));
        // [N6] 人脸最小尺寸门：脸高 < kFaceRecogMinFacePx 识别必败（且远距小脸单张误识是
        //   face_lock/emb 大步刷新的污染入口）→ 跳过，省 2 次 NPU/脸。box 为 xyxy 塞 Rect：脸高 = y2−y。
        if ((float)(box.height - box.y) < kFaceRecogMinFacePx) {
            bool cached_small = false;
            for (const auto& cached : face_inference_cache_) {
                if (cached.box.x == box.x && cached.box.y == box.y
                    && cached.box.width == box.width
                    && cached.box.height == box.height) {
                    cached_small = true;
                    break;
                }
            }
            if (!cached_small) {
                face_inference_cache_.push_back({
                    box, 0.f, "unknown", IdentityEvidence::UNKNOWN,
                    "face_too_small"
                });
            }
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_RESULT] f=%d result=skip_size box=(%d,%d,%d,%d) h=%d min=%.0f",
                    frame_count, box.x, box.y, box.width, box.height,
                    box.height - box.y, kFaceRecogMinFacePx);
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[FACE_EVIDENCE] f=%d box=(%d,%d,%d,%d) status=UNKNOWN reason=face_too_small",
                    frame_count, box.x, box.y, box.width, box.height);
                trace_push(line);
            }
            continue;
        }

        // 同一张检测脸可能同时绑定人体、又进入恢复候选。先复用本帧结果，避免
        // FaceKps + FaceReco 被两条逻辑路径重复调度；unknown/低质量同样缓存。
        bool cache_hit = false;
        float cached_sim = 0.f;
        std::string cached_name = "unknown";
        IdentityEvidence cached_evidence = IdentityEvidence::UNKNOWN;
        std::string cached_reason = "not_evaluated";
        for (const auto& cached : face_inference_cache_) {
            if (cached.box.x != box.x || cached.box.y != box.y
                || cached.box.width != box.width || cached.box.height != box.height)
                continue;
            cache_hit = true;
            cached_sim = cached.sim;
            cached_name = cached.name;
            cached_evidence = cached.evidence;
            cached_reason = cached.reason;
            if (cached.name != "unknown" && cached.sim > best_sim) {
                best_sim = cached.sim;
                best_name = cached.name;
            }
            break;
        }
        if (cache_hit) {
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_RESULT] f=%d result=cache box=(%d,%d,%d,%d)"
                    " sim=%.2f name=%s evidence=%s reason=%s",
                    frame_count, box.x, box.y, box.width, box.height,
                    cached_sim, cached_name.c_str(),
                    identity_evidence_name(cached_evidence), cached_reason.c_str());
                trace_push(line);
            }
            continue;
        }

        if (!take_face_model_slot()) {
            face_inference_cache_.push_back({
                box, 0.f, "unknown", IdentityEvidence::UNKNOWN, "budget_unavailable"
            });
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_RESULT] f=%d result=skip_budget box=(%d,%d,%d,%d)",
                    frame_count, box.x, box.y, box.width, box.height);
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[FACE_EVIDENCE] f=%d box=(%d,%d,%d,%d) status=UNKNOWN reason=budget_unavailable",
                    frame_count, box.x, box.y, box.width, box.height);
                trace_push(line);
            }
            break;
        }
        FaceKeypointResult face_fk = detector_fk.run(img, box);
        std::vector<float> kps_10 = get_kps10(face_fk.points);
        if (kps_10.size() < 10) {
            face_inference_cache_.push_back({
                box, 0.f, "unknown", IdentityEvidence::UNKNOWN, "kps_failed"
            });
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_RESULT] f=%d result=reject_kps box=(%d,%d,%d,%d) points=%d",
                    frame_count, box.x, box.y, box.width, box.height,
                    (int)kps_10.size());
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[FACE_EVIDENCE] f=%d box=(%d,%d,%d,%d) status=UNKNOWN reason=kps_failed",
                    frame_count, box.x, box.y, box.width, box.height);
                trace_push(line);
            }
            continue;   // A10：关键点推理失败 → 本脸跳过（防下游用空关键点对齐）
        }
        // 人脸检测框存在不代表可安全识别：用五点对称性、尺寸和边界完整性过滤
        // 侧脸/强遮挡/截断脸。这样“无身体身份优先帧”会逐帧尝试，但不会让低质量
        // 人脸产生错误的强身份确认。
        float face_q = evaluate_face_quality(kps_10, box, img.cols, img.rows);
        if (face_q < kFaceRecognitionMinQ) {
            null_sink << "[FACE_SKIP] low recognition quality q=" << face_q
                         << " < " << kFaceRecognitionMinQ << std::endl;
            face_inference_cache_.push_back({
                box, 0.f, "unknown", IdentityEvidence::UNKNOWN, "quality_failed"
            });
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_RESULT] f=%d result=reject_quality box=(%d,%d,%d,%d)"
                    " q=%.2f min=%.2f",
                    frame_count, box.x, box.y, box.width, box.height,
                    face_q, kFaceRecognitionMinQ);
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[FACE_EVIDENCE] f=%d box=(%d,%d,%d,%d) status=UNKNOWN reason=quality_failed",
                    frame_count, box.x, box.y, box.width, box.height);
                trace_push(line);
            }
            continue;
        }
        FaceRecognitionResult recognition_result =
            face_recognizer.recognition(img, kps_10, face_boxes.at<float>(i, 4));
        IdentityEvidence evidence = IdentityEvidence::UNKNOWN;
        if (recognition_result.comparison_completed) {
            evidence = recognition_result.user_name == "unknown"
                     ? IdentityEvidence::NEGATIVE : IdentityEvidence::POSITIVE;
        }
        face_inference_cache_.push_back({
            box, recognition_result.similarity, recognition_result.user_name,
            evidence, recognition_result.reason
        });
        if (evidence == IdentityEvidence::NEGATIVE) {
            for (int fi = 0; fi < (int)recovery_faces_.size(); ++fi) {
                if (recovery_faces_[fi].x != box.x || recovery_faces_[fi].y != box.y
                    || recovery_faces_[fi].width != box.width
                    || recovery_faces_[fi].height != box.height) continue;
                if (fi >= (int)recovery_face_owner_person_.size()
                    || recovery_face_owner_person_[fi] < 0
                    || (fi < (int)recovery_face_owner_ambiguous_.size()
                        && recovery_face_owner_ambiguous_[fi])) break;
                int owner = recovery_face_owner_person_[fi];
                if (owner < (int)recovery_body_boxes_.size()) {
                    const cv::Rect& rb = recovery_body_boxes_[owner];
                    cv::Mat owner_box = (cv::Mat_<float>(1, 4) <<
                        (float)rb.x, (float)rb.y,
                        (float)rb.width, (float)rb.height);
                    mark_body_identity_evidence(
                        owner_box, IdentityEvidence::NEGATIVE,
                        "associated_face_mismatch");
                }
                break;
            }
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_RESULT] f=%d result=infer box=(%d,%d,%d,%d)"
                " q=%.2f sim=%.2f name=%s evidence=%s reason=%s",
                frame_count, box.x, box.y, box.width, box.height,
                face_q, recognition_result.similarity,
                recognition_result.user_name.c_str(),
                identity_evidence_name(evidence), recognition_result.reason.c_str());
            trace_push(line);
            std::snprintf(line, sizeof(line),
                "[FACE_EVIDENCE] f=%d box=(%d,%d,%d,%d) status=%s reason=%s",
                frame_count, box.x, box.y, box.width, box.height,
                identity_evidence_name(evidence), recognition_result.reason.c_str());
            trace_push(line);
        }
        if(recognition_result.user_name != "unknown") {
            if(recognition_result.similarity > best_sim) {
                best_sim = recognition_result.similarity;
                best_name = recognition_result.user_name;
            }
        }
    }
    return { best_sim, best_name };
}


Verification_Result LightTracker::face_recognition_verification(
    const cv::Mat& near_box, int box_n,
    const DetectionGroups& det_groups, int main_idx,
    cv::Mat img, cv::Mat matched_one, cv::Mat matched_second,
    const cv::Point2f* standalone_ref,
    float standalone_gate,
    int standalone_budget,
    const cv::Point2f* standalone_alt_ref,
    bool standalone_identity_only)
{
    (void)standalone_alt_ref;
    null_sink << "in face recog verification, box_n :  " << box_n << std::endl;
    null_sink << "near_box : " << std::endl << near_box << std::endl;

    auto make_no_match = []() {
        Verification_Result no_match;
        no_match.got_match = false;
        no_match.matches_main_one = cv::Mat::zeros(0, 2, CV_32S);
        no_match.matches_main_second = cv::Mat::zeros(0, 2, CV_32S);
        no_match.matched_standalone = false;
        no_match.standalone_face_idx = -1;
        no_match.standalone_sim = 0.f;
        no_match.standalone_global = false;
        no_match.standalone_body_box = cv::Mat();
        return no_match;
    };

    if (img.empty())
        return make_no_match();

    const bool have_body_candidates = !near_box.empty() && near_box.cols >= 6 && box_n > 0;
    const bool allow_standalone = (standalone_identity_only
                                   || (standalone_ref != nullptr && standalone_gate > 0.f))
                               && standalone_budget > 0
                               && face_registered_
                               && face_recognizer.has_face_template()
                               && last_main_bw_ > 0.f
                               && last_main_bh_ > 0.f
                               && !recovery_faces_.empty();
    if (!have_body_candidates && !allow_standalone)
        return make_no_match();

    float best_sim = 0.f;
    int best_idx = -1;
    int best_source = -1;
    bool best_standalone = false;
    bool best_standalone_global = false;
    float second_sim = 0.f;   // B1(a)：跨候选次优相似度（余量护栏用）

    auto accept_candidate = [&](float sim, int source, int index, bool standalone,
                                bool standalone_global = false) {
        if (sim > best_sim) {
            second_sim      = best_sim;
            best_sim        = sim;
            best_idx        = index;
            best_source     = source;
            best_standalone = standalone;
            best_standalone_global = standalone && standalone_global;
        } else if (sim > second_sim) {
            second_sim = sim;
        }
    };

    if (have_body_candidates) {
        int n = std::min(box_n, near_box.rows);
        for (int i = 0; i < n; i++) {
            int source = static_cast<int>(near_box.at<float>(i, 4));
            int index  = static_cast<int>(near_box.at<float>(i, 5));

            // ── 边界检查 ──────────────────────────────────
            cv::Mat face_boxes;
            if (source == 0) {
                if (index < 0 || index >= (int)det_groups.dets_one_face.size())
                    continue;
                face_boxes = det_groups.dets_one_face[index];
            }
            else if (source == 1) {
                if (index < 0 || index >= (int)det_groups.dets_second_face.size())
                    continue;
                face_boxes = det_groups.dets_second_face[index];
            }
            else {
                continue;  // 未知 source，跳过。独立人脸有显式分支，不复用 body source/index。
            }

            // ── 无人脸则跳过，不浪费推理 ─────────────────
            if (face_boxes.empty() || face_boxes.rows == 0)
                continue;

            null_sink << "in inference face box : " << std::endl << face_boxes << std::endl;
            auto [sim, name] = face_recognition_inference(face_boxes, img);
            if (name != "unknown")
                accept_candidate(sim, source, index, false);
        }
    }

    // ── 独立/恢复人脸分支：显式处理没有被最终人体提交的人脸，不把 face index
    //   伪装成 body index。旧实现多人时把门外脸硬删除，再永远取最近 Top-K：目标若已
    //   横移到障碍物后，旧位置附近的人会持续占满预算，真脸从不进入 FaceReco。
    //   现在距离只决定首轮优先级：门内最近脸优先，同时保留一个门外探索位；其余候选
    //   按跨帧稳定空间假设的“最近推理帧”公平调度。每帧仍严格受共享预算限制。
    if (allow_standalone) {
        struct StandaloneCand {
            int idx;
            float dist;
            bool outside_gate;
            int hyp_id;
            int last_inferred_frame;
        };
        std::vector<StandaloneCand> pool;
        pool.reserve(recovery_faces_.size());

        // 清理已经离开画面的短期假设。这里按帧龄而非毫秒，只用于公平调度，不参与
        // 身份判定；即使帧率波动也不会改变任何接受/拒绝门。
        face_recovery_hypotheses_.erase(
            std::remove_if(face_recovery_hypotheses_.begin(),
                           face_recovery_hypotheses_.end(),
                           [&](const FaceRecoveryHypothesis& h) {
                               return frame_count - h.last_seen_frame
                                    > kFaceRecoveryHypMaxAgeFrames;
                           }),
            face_recovery_hypotheses_.end());
        std::vector<int> claimed_hyp_ids;
        claimed_hyp_ids.reserve(recovery_faces_.size());

        auto hyp_claimed = [&](int id) {
            return std::find(claimed_hyp_ids.begin(), claimed_hyp_ids.end(), id)
                != claimed_hyp_ids.end();
        };
        auto find_hyp = [&](int id) -> FaceRecoveryHypothesis* {
            for (auto& h : face_recovery_hypotheses_)
                if (h.id == id) return &h;
            return nullptr;
        };

        for (int i = 0; i < (int)recovery_faces_.size(); ++i) {
            const cv::Rect& f = recovery_faces_[i];  // xyxy 塞 Rect
            if ((float)(f.height - f.y) < kFaceRecogMinFacePx) {
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[FACE_CAND] f=%d idx=%d result=reject_size h=%d min=%.0f",
                        frame_count, i, f.height - f.y, kFaceRecogMinFacePx);
                    trace_push(line);
                }
                continue;
            }
            float fcx = ((float)f.x + (float)f.width)  * 0.5f;
            float fcy = ((float)f.y + (float)f.height) * 0.5f;
            float dx = standalone_ref ? fcx - standalone_ref->x : 0.f;
            float dy = standalone_ref ? fcy - standalone_ref->y : 0.f;
            float dist = std::sqrt(dx * dx + dy * dy);

            float fw = std::max(1.f, (float)(f.width - f.x));
            float fh = std::max(1.f, (float)(f.height - f.y));
            float fdiag = std::hypot(fw, fh);
            FaceRecoveryHypothesis* best_hyp = nullptr;
            float best_hyp_cost = std::numeric_limits<float>::max();
            if (!standalone_identity_only) for (auto& h : face_recovery_hypotheses_) {
                if (hyp_claimed(h.id)) continue;
                float hcx = ((float)h.box.x + (float)h.box.width) * 0.5f;
                float hcy = ((float)h.box.y + (float)h.box.height) * 0.5f;
                float hw = std::max(1.f, (float)(h.box.width - h.box.x));
                float hh = std::max(1.f, (float)(h.box.height - h.box.y));
                float hdiag = std::hypot(hw, hh);
                float center_dist = std::hypot(fcx - hcx, fcy - hcy);
                float size_ratio = fdiag / std::max(1.f, hdiag);
                if (center_dist > kFaceRecoveryHypMatchDiag * std::max(fdiag, hdiag)
                    || size_ratio < 0.45f || size_ratio > 2.2f)
                    continue;
                float cost = center_dist / std::max(1.f, std::max(fdiag, hdiag))
                           + 0.35f * std::fabs(std::log(size_ratio));
                if (cost < best_hyp_cost) {
                    best_hyp_cost = cost;
                    best_hyp = &h;
                }
            }
            if (best_hyp == nullptr) {
                face_recovery_hypotheses_.push_back({
                    next_face_recovery_hyp_id_++, f, frame_count, -1000000
                });
                best_hyp = &face_recovery_hypotheses_.back();
            } else {
                best_hyp->box = f;
                best_hyp->last_seen_frame = frame_count;
            }
            claimed_hyp_ids.push_back(best_hyp->id);
            pool.push_back({
                i, dist, !standalone_identity_only && dist > standalone_gate,
                best_hyp->id, best_hyp->last_inferred_frame
            });
        }
        std::sort(pool.begin(), pool.end(),
                  [](const StandaloneCand& a, const StandaloneCand& b) {
                      if (a.outside_gate != b.outside_gate)
                          return !a.outside_gate;  // 门内优先，但门外不删除
                      return a.dist < b.dist;
                  });

        std::vector<int> selected;
        selected.reserve(std::min((int)pool.size(), standalone_budget));
        auto add_selected = [&](int pos) {
            if (pos < 0 || pos >= (int)pool.size()) return;
            for (int p : selected) if (p == pos) return;
            selected.push_back(pos);
        };

        const int take = std::min((int)pool.size(), standalone_budget);
        if (take > 0 && !pool.empty() && standalone_identity_only) {
            for (int slot = 0; slot < take; ++slot)
                add_selected((face_recovery_rotor_ + slot) % (int)pool.size());
        } else if (take > 0 && !pool.empty()) {
            // 临界全局命中等待复验时，把与上帧假设最接近的脸固定在本帧第一个 slot，
            // 否则公平调度可能在确认帧跳去另一张脸，使“两帧确认”永远无法完成。
            if (face_global_pending_
                && now_ms() - face_global_pending_ms_ <= kFaceGlobalConfirmMaxGapMs) {
                int pending_pos = -1;
                float pending_d = std::numeric_limits<float>::max();
                for (int p = 0; p < (int)pool.size(); ++p) {
                    const cv::Rect& f = recovery_faces_[pool[p].idx];
                    float cx = ((float)f.x + (float)f.width) * 0.5f;
                    float cy = ((float)f.y + (float)f.height) * 0.5f;
                    float d = std::hypot(cx - face_global_pending_cx_,
                                         cy - face_global_pending_cy_);
                    float fw = std::max(1.f, (float)(f.width - f.x));
                    float fh = std::max(1.f, (float)(f.height - f.y));
                    float tol = kFaceGlobalSameHypFaceDiag * std::hypot(fw, fh);
                    if (d <= tol && d < pending_d) {
                        pending_d = d;
                        pending_pos = p;
                    }
                }
                add_selected(pending_pos);
            }

            // 首次恢复先利用最近门内脸；之后只按固定周期复查。其余 slot 交给
            // “最久未推理”的物理脸，避免每帧固定浪费一个位置在旧位置路人上。
            if ((int)selected.size() < take
                && (face_recovery_fail_streak_ == 0
                    || face_recovery_fail_streak_ % kFaceRecoveryLocalRetryFrames == 0)) {
                int nearest_local = -1;
                for (int p = 0; p < (int)pool.size(); ++p) {
                    if (!pool[p].outside_gate) { nearest_local = p; break; }
                }
                add_selected(nearest_local);
            }

            // 预算>=2时显式保留一个门外探索位，并优先选择最久未推理的物理脸。
            if ((int)selected.size() < take && take >= 2) {
                int oldest_outside = -1;
                for (int p = 0; p < (int)pool.size(); ++p) {
                    if (!pool[p].outside_gate) continue;
                    if (oldest_outside < 0
                        || pool[p].last_inferred_frame
                           < pool[oldest_outside].last_inferred_frame
                        || (pool[p].last_inferred_frame
                            == pool[oldest_outside].last_inferred_frame
                            && pool[p].dist < pool[oldest_outside].dist))
                        oldest_outside = p;
                }
                add_selected(oldest_outside);
            }

            // 剩余位置按 last_inferred_frame 升序补齐。检测器即使每帧重排数组，
            // 只要物理脸的中心/尺寸连续，仍会优先覆盖尚未真正送过模型的第四张脸。
            std::vector<int> fair_order(pool.size());
            std::iota(fair_order.begin(), fair_order.end(), 0);
            std::sort(fair_order.begin(), fair_order.end(),
                      [&](int a, int b) {
                          if (pool[a].last_inferred_frame != pool[b].last_inferred_frame)
                              return pool[a].last_inferred_frame
                                   < pool[b].last_inferred_frame;
                          return pool[a].dist < pool[b].dist;
                      });
            for (int pos : fair_order) {
                if ((int)selected.size() >= take) break;
                add_selected(pos);
            }
        }

        if (kMatchTrace) {
            int outside_n = 0;
            for (const auto& c : pool) if (c.outside_gate) ++outside_n;
            char picks[96];
            picks[0] = '\0';
            size_t used = 0;
            for (int pos : selected) {
                int infer_age = pool[pos].last_inferred_frame < 0
                              ? -1 : frame_count - pool[pos].last_inferred_frame;
                int wrote = std::snprintf(picks + used, sizeof(picks) - used,
                    "%s%d:h%d%c/%.0f/a%d", used == 0 ? "" : ",",
                    pool[pos].idx, pool[pos].hyp_id,
                    pool[pos].outside_gate ? 'G' : 'L', pool[pos].dist, infer_age);
                if (wrote <= 0 || (size_t)wrote >= sizeof(picks) - used) break;
                used += (size_t)wrote;
            }
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_SEARCH] f=%d viable=%d outside=%d selected=%d budget=%d"
                " rotor=%d pending=%d identity_only=%d picks=%s",
                frame_count, (int)pool.size(), outside_n, (int)selected.size(),
                standalone_budget, face_recovery_rotor_, face_global_pending_ ? 1 : 0,
                standalone_identity_only ? 1 : 0, picks);
            trace_push(line);
        }

        for (int pos : selected) {
            int fidx = pool[pos].idx;
            const cv::Rect& f = recovery_faces_[fidx];
            cv::Mat one(1, 5, CV_32F);
            one.at<float>(0, 0) = (float)f.x;
            one.at<float>(0, 1) = (float)f.y;
            one.at<float>(0, 2) = (float)f.width;
            one.at<float>(0, 3) = (float)f.height;
            one.at<float>(0, 4) = (fidx < (int)recovery_face_scores_.size())
                                ? recovery_face_scores_[fidx] : 1.f;

            auto [sim, name] = face_recognition_inference(one, img);
            // 预算为 0 的缓存复用调用会遍历全部恢复脸；只有本帧缓存中确有结果
            // （真实推理或先前路径已推理）才算“获得过识别机会”。不能把 skip_budget
            // 的第四张脸标成已处理，否则下一帧公平调度仍会饿死它。
            bool evaluated = false;
            for (const auto& cached : face_inference_cache_) {
                if (cached.box.x == f.x && cached.box.y == f.y
                    && cached.box.width == f.width && cached.box.height == f.height) {
                    evaluated = true;
                    break;
                }
            }
            if (evaluated) {
                if (FaceRecoveryHypothesis* h = find_hyp(pool[pos].hyp_id)) {
                    h->last_inferred_frame = frame_count;
                    pool[pos].last_inferred_frame = frame_count;
                }
            }
            if (name != "unknown")
                accept_candidate(sim, 2, fidx, true, pool[pos].outside_gate);
        }

        if (!pool.empty()) {
            face_recovery_rotor_ = (face_recovery_rotor_
                                   + std::max(1, (int)selected.size()))
                                  % (int)pool.size();
        }
    }

    if (best_idx != -1 && best_standalone) {
        cv::Mat body = reconstruct_body_from_face(recovery_faces_[best_idx]);
        if (body.empty()) {
            null_sink << "[FACE_ONLY] standalone face matched but body reconstruction failed"
                         << std::endl;
            best_idx = -1;
            best_standalone = false;
            best_standalone_global = false;
            best_sim = 0.f;
        }
    }

    if (best_idx != -1 && second_sim > 0.f
        && (best_sim - second_sim) < kFaceSimMargin) {
        null_sink << "[FACE_MARGIN] ambiguous faces best=" << best_sim
                     << " second=" << second_sim << " < margin " << kFaceSimMargin
                     << " -> reject face confirm" << std::endl;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_MARGIN] f=%d result=reject best=%.2f second=%.2f margin=%.2f",
                frame_count, best_sim, second_sim, kFaceSimMargin);
            trace_push(line);
        }
        best_idx = -1;
    }

    if (best_idx != -1) {
        Verification_Result result;
        result.got_match = true;
        result.matches_main_one = cv::Mat::zeros(0, 2, CV_32S);
        result.matches_main_second = cv::Mat::zeros(0, 2, CV_32S);
        result.matched_standalone = best_standalone;
        result.standalone_face_idx = best_standalone ? best_idx : -1;
        result.standalone_sim = best_standalone ? best_sim : 0.f;
        result.standalone_global = best_standalone && best_standalone_global;
        result.standalone_body_box = best_standalone
                                   ? reconstruct_body_from_face(recovery_faces_[best_idx])
                                   : cv::Mat();
        if (!best_standalone) {
            if (best_source == 0)
                result.matches_main_one = (cv::Mat_<int>(1, 2) << main_idx, best_idx);
            else
                result.matches_main_second = (cv::Mat_<int>(1, 2) << main_idx, best_idx);
        }
        return result;
    }

    return make_no_match();
}

Face_Match LightTracker::face_recognition_match(const cv::Mat& near_box, int box_n,  const DetectionGroups& det_groups, cv::Mat matched_one, cv::Mat matched_second, int main_idx, cv::Mat img) {

    Verification_Result verification = face_recognition_verification(near_box, box_n, det_groups, main_idx, img, matched_one, matched_second);

    null_sink << "verification got_match : " << verification.got_match << std::endl;
    null_sink << "verification matches_main_one : " << std::endl << verification.matches_main_one << std::endl;
    null_sink << "verification matches_main_second : " << std::endl << verification.matches_main_second << std::endl;

    if (verification.got_match && !verification.matched_standalone) {
        if (verification.matches_main_one.rows > 0) {
            if(matched_one.empty() || verification.matches_main_one.at<int>(0, 1) != matched_one.at<int>(main_idx, 1)) {
                // trackers[0]->reset_mainEmb();
                // int det_idx = verification.matches_main_one.at<int>(0, 1);
                // null_sink << "emb box : " << det_groups.dets_one.row(det_idx) << std::endl;
                // cv::Mat newEmb = compute_embedding(img, det_groups.dets_one.row(det_idx));
                // trackers[0]->update_emb(newEmb, 1.f);
                matched_one = verification.matches_main_one;
            }
        }
        else if (verification.matches_main_second.rows > 0) {
            if(matched_second.empty() || verification.matches_main_second.at<int>(0, 1) != matched_second.at<int>(main_idx, 1)) {
                // trackers[0]->reset_mainEmb();
                // int det_idx = verification.matches_main_second.at<int>(0, 1);
                // null_sink << "emb box : " << det_groups.dets_second.row(det_idx) << std::endl;
                // cv::Mat newEmb = compute_embedding(img, det_groups.dets_second.row(det_idx));
                // trackers[0]->update_emb(newEmb, 1.f);
                matched_second = verification.matches_main_second;
            }
        }
    }
    Face_Match face_match;
    face_match.matched_one = matched_one;
    face_match.matched_second = matched_second;
    face_match.matched_standalone = verification.matched_standalone;
    face_match.standalone_face_idx = verification.standalone_face_idx;
    face_match.standalone_sim = verification.standalone_sim;
    face_match.standalone_global = verification.standalone_global;
    face_match.standalone_body_box = verification.standalone_body_box;

    return face_match;
}


cv::Mat LightTracker::generate_final_results() {
    cv::Mat results(0, 5, CV_32F);
    results.reserve(trackers.size());

    for (const auto& tracker : trackers) {
        // 主目标是已确认轨迹：丢失后重捕时不应再受 min_hits 约束。coast 期间
        // predict() 会把 hit_streak 清零，若仍要求 hit_streak>=min_hits，则重捕后
        // 需再累积 min_hits 帧才输出 → 这几帧 got_main=false 被误判为"仍丢失" →
        // 多滑行 2~3 帧（重捕滞后/卡顿），且 predictor 被持续喂 on_missing，干扰
        // C-identity 复核。故主目标豁免 min_hits（重捕到真实检测即立刻输出）。
        // 注：被 C-identity 延迟接受的帧走 unmatched → update(empty) → tsu>=1，
        // 仍会被下方 time_since_update<1 拦下，豁免不影响延迟语义。
        bool confirmed = tracker->get_is_main()
                       || tracker->get_hit_streak() >= min_hits
                       || frame_count <= min_hits;
        // B3：隔离轨迹不输出——疑主目标影子，输出会在 App 端主框上叠一个幻影人。
        if (!tracker->get_is_main() && tracker->quarantined_) continue;
        if (tracker->get_time_since_update() < 1 && confirmed) {

            cv::Mat bbox;
            // A6：last_observation 改 -1 哨兵初始化后，"有真实观测"判定与 OC-SORT
            // 同约定（sum>=0）；旧 countNonZero 会把 [-1,-1,-1,-1] 哨兵当真实框输出。
            if (cv::sum(tracker->get_last_observation())[0] >= 0)
                bbox = tracker->get_last_observation();
            else
                bbox = tracker->get_state();

            cv::Mat row = (cv::Mat_<float>(1, 5) <<
                bbox.at<float>(0, 0), bbox.at<float>(0, 1),
                bbox.at<float>(0, 2), bbox.at<float>(0, 3),
                static_cast<float>(tracker->get_id() + 1));
            results.push_back(row);
        }
    }

    return results;
}





// ============================================================
// assign_cascade 简化：去掉状态机
// ============================================================
cv::Mat LightTracker::assign_cascade(const DetectionGroups& det_groups,
                                      const cv::Mat& img)
{
    TrackerInfo tracker_info = get_predicted_tracks();

    // ══════════════════════════════════════════════════════════
    // GMC 相机运动补偿（#3b）
    //   估计"上一处理帧→当前处理帧"的背景仿射，把 KF 预测框搬到
    //   当前帧坐标系后再关联。估计失败时 M 为空 → 跳过补偿（恒等）。
    //   注意：传入未补偿的 trks 仅用于构造前景掩膜（目标当前预测位置）。
    // ══════════════════════════════════════════════════════════
    cv::Mat gmc_M;
    if (gmc_enabled_)
        gmc_M = estimate_camera_motion(img, tracker_info.trks);

    if (!gmc_M.empty()) {
        for (int i = 0; i < tracker_info.trks.rows; ++i) {
            cv::Mat row = tracker_info.trks.row(i);  // 共享底层数据 → 原地修改
            warp_box_inplace(row, gmc_M);
        }
        // 主目标平滑中心也随相机运动平移（保持 PTZ 中心先验有效）
        if (smooth_cx_ > 0.f && smooth_cy_ > 0.f) {
            cv::Point2f nc = warp_point(smooth_cx_, smooth_cy_, gmc_M);
            smooth_cx_ = nc.x;
            smooth_cy_ = nc.y;
        }
        // 引导中心同样补偿（它是丢失期搜索门中心 + teleport 复核参考点，
        // 不补偿则云台回扫期间滞后一个帧间位移，距离判定被系统性偏移）
        if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
            cv::Point2f lc = warp_point(lead_cx_, lead_cy_, gmc_M);
            lead_cx_ = lc.x;
            lead_cy_ = lc.y;
        }
        if (last_confirmed_face_ms_ >= 0) {
            cv::Mat fb = (cv::Mat_<float>(1, 4) <<
                (float)last_confirmed_face_box_.x,
                (float)last_confirmed_face_box_.y,
                (float)last_confirmed_face_box_.width,
                (float)last_confirmed_face_box_.height);
            warp_box_inplace(fb, gmc_M);
            last_confirmed_face_box_ = cv::Rect(
                (int)std::lround(fb.at<float>(0, 0)),
                (int)std::lround(fb.at<float>(0, 1)),
                (int)std::lround(fb.at<float>(0, 2)),
                (int)std::lround(fb.at<float>(0, 3)));
        }
        if (face_global_pending_) {
            cv::Point2f fp = warp_point(face_global_pending_cx_,
                                        face_global_pending_cy_, gmc_M);
            face_global_pending_cx_ = fp.x;
            face_global_pending_cy_ = fp.y;
        }
        // 主目标头部预测框也随相机运动补偿（与 body trks 同一变换）
        for (auto& trk : trackers) {
            if (trk->get_is_main() && trk->has_head_track()) {
                cv::Mat hp = trk->get_head_pred_box();
                if (!hp.empty()) {
                    warp_box_inplace(hp, gmc_M);
                    trk->set_head_pred_box(hp);
                }
            }
        }
        // KF 状态级补偿（关键）：以上行级/写回补偿只对"本帧副本"有效——
        // head_pred_box_ 下帧会被 predict_head() 从内部状态重新生成而覆盖，
        // body KF 状态则从未被补偿 → 多帧丢失 + 云台移动时预测累计滞后
        // （丢失越久 IoU/头部门控越失真）。状态级补偿使其逐帧累计正确。
        for (auto& trk : trackers)
            trk->apply_camera_motion(gmc_M);
    }

    MainNonMainSplit main_nomain = split_main_nomain(tracker_info.trks);

  // ====== 用新函数替代 occlusion_check ======
    ProximityInfo proximity = collect_nearby_dets(
        main_nomain.main_trks,
        det_groups.dets_one, 
        det_groups.dets_second);

    cv::Mat results = match_main_target_unified(
        tracker_info, det_groups, proximity,
        main_nomain.main_idx, img);

    cleanup_expired_trackers();

    return results;
}

// ============================================================
// 核心：统一匹配函数（工业级防 ID switch 版本）
// ============================================================
cv::Mat LightTracker::match_main_target_unified(
    const TrackerInfo& info,
    const DetectionGroups& det_groups,
    const ProximityInfo& proximity,
    const std::vector<int>& main_indices,
    const cv::Mat& img)
{
    if (main_indices.empty()) {
        return generate_final_results();
    }

    cv::Mat dets_one = det_groups.dets_one;
    cv::Mat dets_second = det_groups.dets_second;

    int main_trk_idx = main_indices[0];
    cv::Mat trk_box = info.trks.row(main_trk_idx).colRange(0, 4);
    cv::Mat main_emb = trackers[main_trk_idx]->get_emb();
    cv::Mat anchor_emb = trackers[main_trk_idx]->get_anchor_emb();
    cv::Mat near_box = proximity.match_candidates;
    const bool ptz_identity_recovery = spatial_prior_invalid_
                                    && !ptz_blind_anchor_box_.empty();
    int ptz_global_explore_row = -1;

    // ── 主目标头部预测框（已含 GMC 补偿）+ 新鲜度 ──
    //   头部轨迹存在、预测框非空、且距上次观测不太久 → 头部信号可信
    cv::Mat main_head_pred = trackers[main_trk_idx]->get_head_pred_box();
    int  head_tsu = trackers[main_trk_idx]->get_head_time_since_update();
    bool head_track_fresh = trackers[main_trk_idx]->has_head_track()
                          && !main_head_pred.empty()
                          && head_tsu <= kHeadPredMaxAge;

    // B6：本帧墙钟（ms TTL 统一用；帧数 TTL 在 27~374ms 帧间隔波动下语义漂移）
    const int64_t now_match_ms = now_ms();

    // ══════════════════════════════════════════════════════════
    // 遮挡状态机
    // ══════════════════════════════════════════════════════════
    //
    // 触发条件：overlap_count >= 2（检测框真正重叠 = 遮挡/交错）
    // 而非 close_det_count（仅距离近 = 人群中很多人都近但不遮挡）
    //
    // CLEAR → OCCLUDED:    overlap >= 2（有人的框与主目标重叠）
    // OCCLUDED → RECOVERING: overlap <= 1（分离了）
    // RECOVERING → CLEAR:   持续 kRecoveryMs 毫秒无重叠
    // 超时保护: OCCLUDED 超过 kMaxOcclusionMs（墙钟）→ 强制 CLEAR
    //
    int overlap_count   = proximity.overlap_count;
    int close_det_count = proximity.close_det_count;

    // [N3] onset 专用 overlap：用主目标 last_observation（滞后≤1帧）而非 KF 预测框算 IoU。
    //   无 GMC 时云台移动帧 KF 预测框系统性滞后 → 用陈旧框算重叠会漏判/误判遮挡触发。
    //   退出判定（OCCLUDED→RECOVERING / RECOVERING→OCCLUDED）仍用 proximity.overlap_count。
    int onset_overlap = overlap_count;
    {
        cv::Mat mlo = trackers[main_trk_idx]->get_last_observation();
        if (!mlo.empty() && mlo.cols >= 4 && mlo.at<float>(0, 0) >= 0.f) {
            const float kOnsetIou = 0.25f;   // 与 collect_nearby_dets overlap_iou_thresh 一致
            onset_overlap = 0;
            auto cnt = [&](const cv::Mat& d) {
                for (int i = 0; i < d.rows; ++i)
                    if (Utils::iou_single(mlo, d.row(i).colRange(0, 4)) > kOnsetIou) ++onset_overlap;
            };
            cnt(dets_one); cnt(dets_second);
        }
    }

    // PersonIdentityAmbiguityContext 只由真实人物竞争触发。close>=2 只是辅助信息，
    // 不能单独把普通多人场景标为身份歧义。scene risk 与当前是否已由强身份找回目标分离。
    const int main_tsu_onset = trackers[main_trk_idx]->get_time_since_update();
    const int64_t onset_blind_ms = get_body_blind_ms();
    const bool onset_recent = (main_tsu_onset <= kOcclusionOnsetMaxTsu)
                           && (onset_blind_ms >= 0
                               && onset_blind_ms <= kOcclusionOnsetMaxMs);
    const bool direct_person_overlap = onset_recent && onset_overlap >= 2;
    const bool merge_onset = !direct_person_overlap
        && person_identity_context_.last_direct_competition_frame == frame_count - 1
        && onset_overlap <= 1 && proximity.match_candidates.rows <= 1;
    const bool alert_local_competition = id_switch_alert_ && close_det_count >= 2;
    if (direct_person_overlap) {
        note_person_identity_ambiguity(
            kPersonRiskOverlap, close_det_count, true);
    } else if (merge_onset) {
        note_person_identity_ambiguity(
            kPersonRiskMergeOnset, close_det_count, false);
    }
    if (alert_local_competition) {
        note_person_identity_ambiguity(
            kPersonRiskAlertCompetition, close_det_count, false);
    }
    bool person_identity_risk_active =
        person_identity_ambiguity_active(now_match_ms);

    if (occlusion_state_ == OcclusionState::CLEAR) {
        // B2 onset 护栏：主目标须"近期被看到"才允许进入 OCCLUDED（对称于 F7 退出护栏）。
        //   长 coast 期两个路人穿过冻结 KF 框会凑出 overlap>=2 → 伪遮挡（错误 occluder id
        //   + 危险期全量 ReID 风暴）。predict() 已 +1，tsu<=kOcclusionOnsetMaxTsu ≈ 最近 2 帧内有真实命中。
        // [N3] onset 用 last_observation 版 overlap；[N7] 帧数 AND 墙钟双闸（近期真被看到）
        if (onset_overlap >= 2 && onset_recent) {
            occlusion_state_ = OcclusionState::OCCLUDED;
            occlusion_start_frame_ = frame_count;
            occ_start_ms_ = now_match_ms;          // B6：超时改墙钟
            occ_kf_clean_ = true;   // 新遮挡开始，KF 尚未被污染
            if (!main_emb.empty())
                pre_occ_emb_ = main_emb.clone();

            // GMC 开启时保留旧绝对方向兜底；无 GMC 时绝对方向含 PTZ 表观运动，
            // emergence 只能来自下方同帧双真实 relative history。
            if (gmc_enabled_ && !main_indices.empty()) {
                pre_occ_velocity_ = trackers[main_indices[0]]->get_velocity().clone();
            } else {
                pre_occ_velocity_ = cv::Mat();
            }

            // ── 识别遮挡者：与主目标 IoU 最高的非主 tracker ──
            float max_occ_iou = 0.f;
            occluder_tracker_id_ = -1;
            cv::Mat occluder_box_onset;   // 遮挡起始时遮挡者预测框（算浮现朝向用）
            for (int t = 0; t < (int)trackers.size(); ++t) {
                if (std::find(main_indices.begin(), main_indices.end(), t) != main_indices.end())
                    continue;
                float iou = Utils::iou_single(info.trks.row(t), trk_box);
                if (iou > max_occ_iou) {
                    max_occ_iou = iou;
                    occluder_tracker_id_ = trackers[t]->get_id();
                    occluder_box_onset = info.trks.row(t).clone();
                }
            }
            // 无 GMC：只使用 main/secondary 同帧双真实 BODY 的相对中心变化；没有稳定
            // history 就保持 invalid。不能再用 main→occluder 位置或绝对 pre-occ 方向猜身份。
            // GMC 开启路径保持旧几何/速度兜底，避免扩大本轮行为边界。
            emergence_dir_x_ = 0.f; emergence_dir_y_ = 0.f;
            if (occluder_tracker_id_ >= 0 && !occluder_box_onset.empty()) {
                int samples = 0;
                float consistency = 0.f;
                bool relative_valid = estimate_relative_emergence_direction(
                    occluder_tracker_id_, emergence_dir_x_, emergence_dir_y_,
                    &samples, &consistency);
                const char* source = relative_valid ? "relative_history" : "none";
                if (!relative_valid && gmc_enabled_) {
                    float mcx = (trk_box.at<float>(0,0) + trk_box.at<float>(0,2)) * 0.5f;
                    float mcy = (trk_box.at<float>(0,1) + trk_box.at<float>(0,3)) * 0.5f;
                    float ocx = (occluder_box_onset.at<float>(0,0) + occluder_box_onset.at<float>(0,2)) * 0.5f;
                    float ocy = (occluder_box_onset.at<float>(0,1) + occluder_box_onset.at<float>(0,3)) * 0.5f;
                    float dx = ocx - mcx, dy = ocy - mcy;
                    float dn = std::sqrt(dx*dx + dy*dy);
                    if (dn > 1e-3f) {
                        emergence_dir_x_ = dx / dn;
                        emergence_dir_y_ = dy / dn;
                        source = "gmc_geometry";
                    } else if (pre_occ_velocity_.type() == CV_32F
                               && pre_occ_velocity_.total() >= 2) {
                        const float* v = (const float*)pre_occ_velocity_.data;
                        float vn = std::sqrt(v[0]*v[0] + v[1]*v[1]);
                        if (vn > 1e-3f) {
                            emergence_dir_x_ = v[0]/vn;
                            emergence_dir_y_ = v[1]/vn;
                            source = "gmc_pre_occ";
                        }
                    }
                }
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[EMERGE] f=%d other=%d source=%s permission=search_hold"
                        " samples=%d consistency=%.2f dir=(%.3f,%.3f)",
                        frame_count, occluder_tracker_id_, source,
                        samples, consistency,
                        emergence_dir_x_, emergence_dir_y_);
                    trace_push(line);
                }
            } else if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[EMERGE] f=%d other=-1 source=none permission=search_hold"
                    " samples=0 consistency=0.00 dir=(0.000,0.000)",
                    frame_count);
                trace_push(line);
            }
            null_sink << "[STATE] CLEAR -> OCCLUDED (overlap=" << overlap_count
                      << " occluder_id=" << occluder_tracker_id_ << ")" << std::endl;
        }
    } else if (occlusion_state_ == OcclusionState::OCCLUDED) {
        // overlap_count 含主目标自身检测：onset 需 主+遮挡者(=2)，但深度遮挡时
        // 主目标未被检测，单个遮挡者也会令 overlap=1 → 会在"主目标仍被挡住"时
        // 误判为已分离、过早退出 OCCLUDED（状态机抖动）。故仅当主目标近期有真实
        // 人体命中，或有经严格连续性/人脸身份门确认的部件观测时才允许退出。
        // 部件新鲜度独立记录，不篡改 body tracker 的 time_since_update 语义。
        int  main_tsu_fsm = trackers[main_trk_idx]->get_time_since_update();
        bool head_recently_seen = last_head_continuity_ms_ >= 0
                               && (now_match_ms - last_head_continuity_ms_) <= kPartObsRecentMs;
        bool face_recently_seen = last_face_identity_ms_ >= 0
                               && (now_match_ms - last_face_identity_ms_) <= kPartObsRecentMs;
        bool part_recently_seen = head_recently_seen || face_recently_seen;
        bool main_recently_seen = (main_tsu_fsm <= 1) || part_recently_seen;
        // F7：需连续 kSeparationConfirmFrames 帧满足"疑似已分离"才真正进 RECOVERING，
        //   单帧检测抖动（遮挡者漏检令 overlap 瞬时掉到 1）不足以过早重新采信 KF 空间信号。
        if (overlap_count <= 1 && main_recently_seen) separation_streak_++;
        else                                          separation_streak_ = 0;
        if (separation_streak_ >= kSeparationConfirmFrames) {
            occlusion_state_ = OcclusionState::RECOVERING;
            recovery_start_ms_ = now_match_ms;   // [N7] 恢复期改墙钟
            separation_streak_ = 0;
            null_sink << "[STATE] OCCLUDED -> RECOVERING (overlap<=1, main seen x"
                      << kSeparationConfirmFrames << ")" << std::endl;
        } else if (now_match_ms - occ_start_ms_ > kMaxOcclusionMs) {   // B6：墙钟超时
            occlusion_state_ = OcclusionState::CLEAR;
            separation_streak_ = 0;
            null_sink << "[STATE] OCCLUDED -> CLEAR (timeout)" << std::endl;
        }
    } else { // RECOVERING
        if (overlap_count >= 2) {
            occlusion_state_ = OcclusionState::OCCLUDED;
            occlusion_start_frame_ = frame_count;
            occ_start_ms_ = now_match_ms;          // B6：超时改墙钟
            null_sink << "[STATE] RECOVERING -> OCCLUDED (re-occluded)" << std::endl;
        } else if (now_match_ms - recovery_start_ms_ > kRecoveryMs) {   // [N7] 墙钟
            occlusion_state_ = OcclusionState::CLEAR;
            null_sink << "[STATE] RECOVERING -> CLEAR (confirmed)" << std::endl;
        }
    }

    bool in_danger = (occlusion_state_ != OcclusionState::CLEAR);
    const int body_search_tsu = trackers[main_trk_idx]->get_time_since_update();
    const int64_t body_search_blind_ms = get_body_blind_ms();
    const bool body_crowded_frame = proximity.close_det_count >= 3
                                 || proximity.all_candidates.rows >= 4;
    const bool normal_body_global_search = !proximity.all_candidates.empty()
                                 && (in_danger || id_switch_alert_
                                     // predict() 在匹配前已把健康轨迹 tsu 加到 1；>1 才表示上一帧 BODY 漏检。
                                     || body_search_tsu > 1
                                     || body_search_blind_ms < 0
                                     || body_search_blind_ms >= kReacqProbationMs
                                     || body_crowded_frame);
    const bool body_global_search = !ptz_identity_recovery && normal_body_global_search;
    // 正常单人帧保持原 top-5 局部路径；拥挤、危险或任意一次 BODY miss 后，候选集合
    // 切为全画面。全画面只扩大“可被轮到”的集合，不扩大本帧 ReID 模型预算。
    if (body_global_search)
        near_box = proximity.all_candidates;
    if (ptz_identity_recovery && !proximity.all_candidates.empty()) {
        auto in_local = [&](const cv::Mat& candidate) {
            const int src = (int)candidate.at<float>(0, 4);
            const int idx = (int)candidate.at<float>(0, 5);
            for (int r = 0; r < near_box.rows; ++r) {
                if ((int)near_box.at<float>(r, 4) == src
                    && (int)near_box.at<float>(r, 5) == idx)
                    return true;
            }
            return false;
        };
        std::vector<int> extras;
        for (int r = 0; r < proximity.all_candidates.rows; ++r)
            if (!in_local(proximity.all_candidates.row(r))) extras.push_back(r);
        if (!extras.empty()) {
            const int pick = extras[ptz_blind_explore_rotor_ % extras.size()];
            ptz_blind_explore_rotor_ = (ptz_blind_explore_rotor_ + 1) % extras.size();
            near_box.push_back(proximity.all_candidates.row(pick));
            ptz_global_explore_row = near_box.rows - 1;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[PTZ_RECOVERY] f=%d phase=%d anchor_local=%d global_slot_row=%d all=%d",
                    frame_count, (int)ptz_blind_phase_, proximity.match_candidates.rows,
                    ptz_global_explore_row, proximity.all_candidates.rows);
                trace_push(line);
            }
        }
    }
    if (ptz_identity_recovery && kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[SPATIAL_GATE] f=%d disabled_by_ptz_slide=1 iou=0 head=0 motion=0 local_anchor=%d",
            frame_count, ptz_blind_anchor_box_.empty() ? 0 : 1);
        trace_push(line);
    }

    // ── 遮挡期浮现点：遮挡者当前框 + 起始朝向 → 目标将从遮挡者远侧浮现处，仅用于
    //   recovery search/ReID hypothesis/separation HOLD；不写 trusted lead、不作身份否决。
    //   相机输出仍保持不动(Option A)。危险期 + 有遮挡者轨迹 + 朝向有效
    //   才算；遮挡者已删 / furniture(无轨迹) / 静止(朝向退化) → emergence_valid_=false → 回退原地保持。──
    emergence_valid_ = false;
    if (in_danger && occluder_tracker_id_ >= 0
        && (emergence_dir_x_ != 0.f || emergence_dir_y_ != 0.f)) {
        for (int t = 0; t < (int)trackers.size(); ++t) {
            if (trackers[t]->get_id() != occluder_tracker_id_) continue;
            cv::Mat ob = info.trks.row(t);
            if (ob.empty() || ob.cols < 4) break;
            float ocx = (ob.at<float>(0,0) + ob.at<float>(0,2)) * 0.5f;
            float ocy = (ob.at<float>(0,1) + ob.at<float>(0,3)) * 0.5f;
            float ow  = ob.at<float>(0,2) - ob.at<float>(0,0);
            float oh  = ob.at<float>(0,3) - ob.at<float>(0,1);
            float odiag = std::sqrt(ow*ow + oh*oh);
            emergence_cx_ = ocx + emergence_dir_x_ * kEmergencePushDiag * odiag;
            emergence_cy_ = ocy + emergence_dir_y_ * kEmergencePushDiag * odiag;
            // issue 3：浮现点必须在画面内，否则 recovery search 会偏向画面边角。
            if (emergence_cx_ < 0.f || emergence_cx_ > (float)img_w
                || emergence_cy_ < 0.f || emergence_cy_ > (float)img_h) {
                null_sink << "[EMERGE] off-screen (" << emergence_cx_ << "," << emergence_cy_
                             << ") -> hold" << std::endl;
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[EMERGE_POINT] f=%d other=%d result=invalid reason=offscreen"
                        " source=%s center=(%.1f,%.1f) permission=search_hold",
                        frame_count, occluder_tracker_id_,
                        gmc_enabled_ ? "gmc_or_relative" : "relative_history",
                        emergence_cx_, emergence_cy_);
                    trace_push(line);
                }
                break;   // emergence_valid_ 保持 false → 回退 update_lead_center(pr.box)
            }
            emergence_valid_ = true;
            emergence_update_ms_ = now_match_ms;   // [N8] 刷新浮现点时钟（零检测帧不进本函数 → 超时作废）
            null_sink << "[EMERGE] occluder id=" << occluder_tracker_id_
                         << " far-side gate=(" << emergence_cx_ << "," << emergence_cy_ << ")" << std::endl;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[EMERGE_POINT] f=%d other=%d result=valid reason=inside_frame"
                    " source=%s center=(%.1f,%.1f) permission=search_hold",
                    frame_count, occluder_tracker_id_,
                    gmc_enabled_ ? "gmc_or_relative" : "relative_history",
                    emergence_cx_, emergence_cy_);
                trace_push(line);
            }
            break;
        }
    }

    const bool pose_refresh_due = occlusion_state_ == OcclusionState::CLEAR
                               && (last_committed_pose_frame_ < 0
                                   || frame_count - last_committed_pose_frame_ >= kPoseInferEveryN);
    bool periodic_face_due = face_recognition_every_n_frames > 0
                          && (frame_count % face_recognition_every_n_frames == 0);
    // 二级 ReID 绝不抢占危险期、人脸优先帧、Pose 刷新帧或周期人脸验证帧的预算。
    // 其他 CLEAR 帧可用它做低频他人轨迹刷新。
    frame_allow_secondary_reid_ = !in_danger && !body_global_search && !face_priority_frame_
                              && !pose_refresh_due && !periodic_face_due;

    // ── 人脸硬锚定（#1）：人脸锁是否仍在有效期内 ──
    // 最近 kFaceLockTTLMs 毫秒内确认过人脸 → 身份已知 → 危险期收紧身份门槛
    bool face_lock_active = face_locked_
                          && (now_match_ms - last_face_lock_ms_ <= kFaceLockTTLMs);   // B6：墙钟 TTL

    // 遮挡/恢复期冻结平滑中心 → PTZ 不被误导
    if (!id_switch_alert_ && !in_danger)
        update_smooth_center(trk_box);

    // 选择最可靠的 embedding 做匹配
    // OCCLUDED/RECOVERING → 用遮挡前快照或 anchor（绝不用可能已污染的当前emb）
    cv::Mat match_emb = main_emb;
    if (in_danger && !pre_occ_emb_.empty())
        match_emb = pre_occ_emb_;
    else if (in_danger && !anchor_emb.empty())
        match_emb = anchor_emb;
    else if (id_switch_alert_ && !anchor_emb.empty())
        match_emb = anchor_emb;

    // ──────────────────────────────────────────────────────
    // STEP 1：对 near_box 中每个候选打分
    // ──────────────────────────────────────────────────────
    struct Candidate {
        int   near_idx;
        int   source;
        int   det_index;
        float iou;              // Kalman预测框与检测框重叠（空间连续性）
        float reid_sim;         // 与当前emb的余弦相似度
        float anchor_sim;       // 与不可变anchor的余弦相似度
        float oks_score;        // 关键点空间匹配（帧间位置跟踪）
        float body_shape_sim;   // 骨骼比例相似度（逐维相对偏差度量，软性身份 tie-breaker）
        bool  has_shape;        // body_shape 是否可用
        float anti_occ;         // 远离遮挡者预测位置的得分（排除法）
        float head_match;       // 候选头部框 vs 主目标预测头部框 的标准化中心距分（遮挡期可靠空间判别）
        bool  has_head;         // 头部信号是否可用
        float total;
        cv::Mat emb;            // 候选 ReID 特征（仅 do_reid 候选非空）；供"已知他人"外观排除
        float shoulder_cont;    // 候选肩中点 vs 主目标预测肩中点的连续性 [0,1]（转身期空间判别）
        bool  has_shoulder;
        bool  face_hold = false;// 人脸锁硬保持：本候选是"空间延续 confirmed 轨迹者" → 放宽 anchor 硬门（身份由人脸定）
        bool  reid_evaluated = false; // 本帧确实得到有效 ReID；未调度/推理失败均为 identity unknown
        bool  global_explore = false; // 来自全图公平探索槽；远端普通命中须走同假设复验
        int   body_hyp_id = -1;       // 跨帧物理人体假设，不能使用 detector 数组下标代替
        bool  qualified_pending = false;
        float pending_center_dist = -1.f;
        float pending_box_iou = -1.f;
        float pending_size_ratio = -1.f;
    };

    std::vector<Candidate> candidates;

    // ── 人体 ReID 固定预算调度 ──
    // 正常单人仍按原 IoU/距离/头部预排；拥挤、危险或 BODY miss 后同时维护全画面
    // 物理人体假设。每帧保留局部利用槽，并把其余槽给“本轮最久未推理”的候选，
    // 同龄候选才按可信圆心由近及远。距离只决定顺序，绝不永久删除远处候选。
    std::vector<char> do_reid(near_box.rows, 0);
    std::vector<char> global_explore_reid(near_box.rows, 0);
    std::vector<int> body_hyp_for_row(near_box.rows, -1);
    std::vector<float> body_dist_for_row(near_box.rows, 0.f);
    std::vector<int> body_ring_for_row(near_box.rows, 0);
    {
        int k_budget = in_danger
                     ? kReidMaxCandDanger
                     : kReidMaxCandClear + ((proximity.close_det_count >= 3) ? 1 : 0);
        k_budget = std::min(k_budget, near_box.rows);

        float rcx, rcy;
        const char* global_reference_source = "kf";
        if (ptz_identity_recovery) {
            rcx = (ptz_blind_anchor_box_.at<float>(0, 0)
                 + ptz_blind_anchor_box_.at<float>(0, 2)) * 0.5f;
            rcy = (ptz_blind_anchor_box_.at<float>(0, 1)
                 + ptz_blind_anchor_box_.at<float>(0, 3)) * 0.5f;
            global_reference_source = "ptz_blind_anchor";
        } else if (body_reid_anchor_cx_ >= 0.f && body_reid_anchor_cy_ >= 0.f) {
            rcx = body_reid_anchor_cx_; rcy = body_reid_anchor_cy_;
            global_reference_source = "body_identity_anchor";
        } else if (recovery_search_center(rcx, rcy)) {
            global_reference_source = emergence_valid_
                && current_frame_timestamp_ms_ >= emergence_update_ms_
                && current_frame_timestamp_ms_ - emergence_update_ms_
                   <= kEmergenceMaxAgeMs ? "emergence" : "trusted_lead";
        } else {
            rcx = (trk_box.at<float>(0, 0) + trk_box.at<float>(0, 2)) * 0.5f;
            rcy = (trk_box.at<float>(0, 1) + trk_box.at<float>(0, 3)) * 0.5f;
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[RECOVERY_REFERENCE] f=%d use=global_reid source=%s"
                " center=(%.1f,%.1f) permission=search_only",
                frame_count, global_reference_source, rcx, rcy);
            trace_push(line);
        }
        const float tw = std::max(1.f, trk_box.at<float>(0, 2) - trk_box.at<float>(0, 0));
        const float th = std::max(1.f, trk_box.at<float>(0, 3) - trk_box.at<float>(0, 1));
        const float frame_diag = std::max(1.f, std::hypot((float)img_w, (float)img_h));
        const float raw_scale = body_reid_anchor_diag_ > 0.f
                              ? body_reid_anchor_diag_ : std::hypot(tw, th);
        // 近距离裁剪框可能接近整屏、远距框可能极小；钳制只影响环排序，不影响资格。
        const float radial_scale = std::max(24.f, std::min(raw_scale, 0.45f * frame_diag));

        body_reid_hypotheses_.erase(
            std::remove_if(body_reid_hypotheses_.begin(), body_reid_hypotheses_.end(),
                [&](const BodyReidHypothesis& h) {
                    return frame_count - h.last_seen_frame > kBodyReidHypMaxAgeFrames;
                }),
            body_reid_hypotheses_.end());

        std::vector<int> claimed_hyp_ids;
        int new_hyp_count = 0;
        auto is_claimed = [&](int id) {
            return std::find(claimed_hyp_ids.begin(), claimed_hyp_ids.end(), id)
                   != claimed_hyp_ids.end();
        };
        auto hyp_by_id = [&](int id) -> BodyReidHypothesis* {
            for (auto& h : body_reid_hypotheses_) if (h.id == id) return &h;
            return nullptr;
        };
        auto hyp_iou = [](const BodyReidHypothesis& h,
                          float x1, float y1, float x2, float y2) {
            const float ix1 = std::max(h.x1, x1), iy1 = std::max(h.y1, y1);
            const float ix2 = std::min(h.x2, x2), iy2 = std::min(h.y2, y2);
            const float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
            const float inter = iw * ih;
            const float ha = std::max(0.f, h.x2 - h.x1) * std::max(0.f, h.y2 - h.y1);
            const float ba = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
            return inter / std::max(1.f, ha + ba - inter);
        };

        for (int k = 0; k < near_box.rows; ++k) {
            const float x1 = near_box.at<float>(k, 0), y1 = near_box.at<float>(k, 1);
            const float x2 = near_box.at<float>(k, 2), y2 = near_box.at<float>(k, 3);
            const float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
            const float bw = std::max(1.f, x2 - x1), bh = std::max(1.f, y2 - y1);
            const float diag = std::hypot(bw, bh);
            body_dist_for_row[k] = std::hypot(cx - rcx, cy - rcy);
            const float ring_pos = body_dist_for_row[k] / radial_scale;
            body_ring_for_row[k] = ring_pos <= 1.5f ? 0
                                     : ring_pos <= 3.f ? 1
                                     : ring_pos <= 5.f ? 2 : 3;

            BodyReidHypothesis* best_h = nullptr;
            float best_cost = FLT_MAX, second_cost = FLT_MAX;
            for (auto& h : body_reid_hypotheses_) {
                if (is_claimed(h.id)) continue;
                const float hcx = (h.x1 + h.x2) * 0.5f, hcy = (h.y1 + h.y2) * 0.5f;
                const float hw = std::max(1.f, h.x2 - h.x1), hh = std::max(1.f, h.y2 - h.y1);
                const float hdiag = std::hypot(hw, hh);
                const float cdist = std::hypot(cx - hcx, cy - hcy);
                const float size_ratio = diag / std::max(1.f, hdiag);
                if (cdist > kBodyReidHypMatchDiag * std::max(diag, hdiag)
                    || size_ratio < 0.40f || size_ratio > 2.50f) continue;
                const float cost = cdist / std::max(1.f, std::max(diag, hdiag))
                                 + 0.35f * std::fabs(std::log(size_ratio))
                                 + 0.25f * (1.f - hyp_iou(h, x1, y1, x2, y2));
                if (cost < best_cost) {
                    second_cost = best_cost; best_cost = cost; best_h = &h;
                } else if (cost < second_cost) {
                    second_cost = cost;
                }
            }
            // 交叉/合并/分裂时不继承旧人的“已扫描”状态；新建假设的代价只是多一次推理，
            // 比把重新出现的主目标错误标成已处理而永久饿死安全。
            if (best_h == nullptr
                || (second_cost < FLT_MAX
                    && second_cost - best_cost < kBodyReidHypAmbigCostGap)) {
                BodyReidHypothesis h;
                h.id = next_body_reid_hyp_id_++;
                h.x1 = x1; h.y1 = y1; h.x2 = x2; h.y2 = y2;
                h.first_seen_frame = h.last_seen_frame = frame_count;
                body_reid_hypotheses_.push_back(h);
                best_h = &body_reid_hypotheses_.back();
                ++new_hyp_count;
            } else {
                best_h->x1 = x1; best_h->y1 = y1;
                best_h->x2 = x2; best_h->y2 = y2;
                best_h->last_seen_frame = frame_count;
            }
            claimed_hyp_ids.push_back(best_h->id);
            body_hyp_for_row[k] = best_h->id;
        }

        if (body_global_search && !body_reid_global_active_) {
            body_reid_global_active_ = true;
            ++body_reid_scan_epoch_;
            body_reid_scan_epoch_frame_ = frame_count;
        } else if (!body_global_search) {
            body_reid_global_active_ = false;
        }
        if (body_global_search && body_reid_scan_epoch_frame_ >= 0) {
            bool any_epoch_unscanned = false;
            for (int id : body_hyp_for_row) {
                BodyReidHypothesis* h = hyp_by_id(id);
                if (h && h->last_attempt_frame < body_reid_scan_epoch_frame_) {
                    any_epoch_unscanned = true;
                    break;
                }
            }
            if (!any_epoch_unscanned && !body_hyp_for_row.empty()) {
                ++body_reid_scan_epoch_;
                body_reid_scan_epoch_frame_ = frame_count;
            }
        }

        std::vector<std::pair<float,int>> pre;
        pre.reserve(near_box.rows);
        // 局部利用仍沿用原 lead/KF 参考，保证快速运动和云台表观位移下的即时跟随；
        // 只有全局探索的环顺序使用“不被弱匹配污染”的可信身份圆心。
        float local_rcx, local_rcy;
        const char* local_reference_source = "kf";
        if (ptz_identity_recovery) {
            local_rcx = rcx;
            local_rcy = rcy;
            local_reference_source = "ptz_blind_anchor";
        } else if (!recovery_search_center(local_rcx, local_rcy)) {
            local_rcx = (trk_box.at<float>(0,0) + trk_box.at<float>(0,2)) * 0.5f;
            local_rcy = (trk_box.at<float>(0,1) + trk_box.at<float>(0,3)) * 0.5f;
        } else {
            local_reference_source = emergence_valid_
                && current_frame_timestamp_ms_ >= emergence_update_ms_
                && current_frame_timestamp_ms_ - emergence_update_ms_
                   <= kEmergenceMaxAgeMs ? "emergence" : "trusted_lead";
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[RECOVERY_REFERENCE] f=%d use=local_reid source=%s"
                " center=(%.1f,%.1f) permission=search_only",
                frame_count, local_reference_source, local_rcx, local_rcy);
            trace_push(line);
        }
        for (int k = 0; k < near_box.rows; ++k) {
            if (ptz_identity_recovery && k == ptz_global_explore_row)
                continue;  // 只通过保留的 global slot 调度，不能抢局部身份复验槽。
            cv::Mat d = near_box.row(k).colRange(0, 4);
            float key = ptz_identity_recovery ? 0.f : Utils::iou_single(trk_box, d);
            const float dcx = (d.at<float>(0,0) + d.at<float>(0,2)) * 0.5f;
            const float dcy = (d.at<float>(0,1) + d.at<float>(0,3)) * 0.5f;
            const float local_dist = std::hypot(dcx - local_rcx, dcy - local_rcy);
            key += kPrerankDistW * std::max(0.f,
                   1.f - local_dist / (2.f * radial_scale));
            if (!ptz_identity_recovery && in_danger && head_track_fresh) {
                int psrc = (int)near_box.at<float>(k, 4);
                int pidx = (int)near_box.at<float>(k, 5);
                const cv::Mat& phead_src = (psrc == 0) ? det_groups.dets_one_head
                                                       : det_groups.dets_second_head;
                if (pidx >= 0 && pidx < phead_src.rows) {
                    cv::Mat ph = phead_src.row(pidx).colRange(0, 4);
                    if (cv::countNonZero(ph) > 0)
                        key += kPrerankHeadW * head_match_score(main_head_pred, ph);
                }
            }
            pre.push_back({key, k});
        }
        std::sort(pre.begin(), pre.end(),
                  [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                      return a.first > b.first;
                  });

        std::vector<int> selected;
        selected.reserve(k_budget);
        auto add_selected = [&](int row, bool explore) {
            if (row < 0 || row >= near_box.rows || (int)selected.size() >= k_budget) return;
            if (do_reid[row]) return;
            do_reid[row] = 1;
            global_explore_reid[row] = explore ? 1 : 0;
            selected.push_back(row);
        };

        int local_slots = k_budget;
        if (ptz_identity_recovery && ptz_global_explore_row >= 0)
            local_slots = std::max(0, k_budget - 1);
        if (body_global_search) {
            const bool long_blind = body_search_blind_ms < 0
                                 || body_search_blind_ms >= kBodyReidLongBlindMs
                                 || body_search_tsu > 3;
            if (long_blind) {
                local_slots = std::min(kBodyReidLongBlindLocalSlots, k_budget);
            } else {
                const int explore_slots = (body_search_tsu <= 1 && !in_danger
                                           && !id_switch_alert_)
                                        ? 1 : kBodyReidExploreSlots;
                local_slots = std::max(1, k_budget - explore_slots);
            }
        }

        // BODY provisional 优先按稳定物理 hyp 复验，几何只作二次安全门。
        // detector source/index 变化不影响同一 hyp；也不允许另一人仅因进入旧中心获得复验槽。
        if (pending_body_hyp_id_ >= 0) {
            int pending_row = -1;
            float pending_dist = FLT_MAX;
            bool exact_hyp_seen = false;
            for (int k = 0; k < near_box.rows; ++k) {
                if (body_hyp_for_row[k] != pending_body_hyp_id_) continue;
                exact_hyp_seen = true;
                float center_dist = FLT_MAX, box_iou = 0.f, size_ratio = 0.f;
                if (!ptz_identity_recovery && !body_provisional_geometry(
                        near_box.row(k).colRange(0, 4),
                        center_dist, box_iou, size_ratio))
                    continue;
                if (ptz_identity_recovery || center_dist < pending_dist) {
                    pending_dist = center_dist;
                    pending_row = k;
                    if (ptz_identity_recovery) break;
                }
            }
            if (exact_hyp_seen && pending_row < 0) {
                clear_body_provisional("geometry_jump", "reid_schedule");
            } else if (!exact_hyp_seen) {
                trace_body_provisional(
                    "retain", pending_body_hyp_id_, pending_body_hyp_id_,
                    pending_src_, pending_idx_, -1.f, -1.f,
                    pending_body_reid_, pending_body_anchor_,
                    reacq_defer_count_, kReacqMaxDefer,
                    -1.f, -1.f, -1.f, "missing-but-retained", "reid_schedule");
            }
            add_selected(pending_row, body_global_search || ptz_identity_recovery);
        } else if (!ptz_identity_recovery && pending_active_) {
            // generic pending 保留原有中心复验语义，但不获得 BODY continuation 优先权。
            int pending_row = -1;
            float pending_dist = FLT_MAX;
            for (int k = 0; k < near_box.rows; ++k) {
                const float cx = (near_box.at<float>(k,0) + near_box.at<float>(k,2)) * 0.5f;
                const float cy = (near_box.at<float>(k,1) + near_box.at<float>(k,3)) * 0.5f;
                const float d = std::hypot(cx - pending_cx_, cy - pending_cy_);
                const float bw = std::max(1.f, near_box.at<float>(k,2) - near_box.at<float>(k,0));
                const float bh = std::max(1.f, near_box.at<float>(k,3) - near_box.at<float>(k,1));
                if (d <= kProvisionalPosTolFactor * std::hypot(bw, bh) && d < pending_dist) {
                    pending_dist = d;
                    pending_row = k;
                }
            }
            add_selected(pending_row, body_global_search);
        }

        for (const auto& p : pre) {
            if ((int)selected.size() >= local_slots) break;
            add_selected(p.second, false);
        }

        // PTZ 已实际滑动后，旧图像空间只用于围绕最后真实 BODY 的候选池，
        // 不再决定身份排序。仍保留一个既有预算槽作全图公平探索，避免局部框
        // 内的遮挡者长期占满 ReID 调度。
        if (ptz_identity_recovery && ptz_global_explore_row >= 0)
            add_selected(ptz_global_explore_row, true);

        if (body_global_search && (int)selected.size() < k_budget) {
            std::vector<int> fair_order(near_box.rows);
            std::iota(fair_order.begin(), fair_order.end(), 0);
            std::sort(fair_order.begin(), fair_order.end(), [&](int a, int b) {
                BodyReidHypothesis* ha = hyp_by_id(body_hyp_for_row[a]);
                BodyReidHypothesis* hb = hyp_by_id(body_hyp_for_row[b]);
                const int la = ha ? ha->last_attempt_frame : -1000000;
                const int lb = hb ? hb->last_attempt_frame : -1000000;
                const bool ua = body_reid_scan_epoch_frame_ < 0
                             || la < body_reid_scan_epoch_frame_;
                const bool ub = body_reid_scan_epoch_frame_ < 0
                             || lb < body_reid_scan_epoch_frame_;
                if (ua != ub) return ua;
                if (ua && ub) {
                    const int fa = ha ? ha->first_seen_frame : frame_count;
                    const int fb = hb ? hb->first_seen_frame : frame_count;
                    if (fa != fb) return fa < fb; // 持续有新人进入时，旧的远端候选不能饥饿
                    if (body_ring_for_row[a] != body_ring_for_row[b])
                        return body_ring_for_row[a] < body_ring_for_row[b];
                } else if (la != lb) {
                    return la < lb;
                }
                return body_dist_for_row[a] < body_dist_for_row[b];
            });
            for (int row : fair_order) {
                if ((int)selected.size() >= k_budget) break;
                add_selected(row, true);
            }
        }
        // 候选少或探索槽去重后仍有余量时，用原预排补齐，但绝不超过 k_budget。
        for (const auto& p : pre) {
            if ((int)selected.size() >= k_budget) break;
            add_selected(p.second, false);
        }

        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[BODY_POOL] f=%d global=%d local=%d all=%d hyp=%d new=%d"
                " epoch=%d center=(%.0f,%.0f) scale=%.0f budget=%d",
                frame_count, body_global_search ? 1 : 0,
                proximity.match_candidates.rows, proximity.all_candidates.rows,
                (int)body_reid_hypotheses_.size(), new_hyp_count,
                body_reid_scan_epoch_, rcx, rcy, radial_scale, k_budget);
            trace_push(line);
            char picks[192]; picks[0] = '\0'; size_t used = 0;
            for (int row : selected) {
                BodyReidHypothesis* h = hyp_by_id(body_hyp_for_row[row]);
                const int age = (!h || h->last_attempt_frame < -999000)
                              ? -1 : frame_count - h->last_attempt_frame;
                const int wrote = std::snprintf(picks + used, sizeof(picks) - used,
                    "%s%d/%d:h%d:r%d:%c:a%d", used ? "," : "",
                    (int)near_box.at<float>(row,4), (int)near_box.at<float>(row,5),
                    body_hyp_for_row[row], body_ring_for_row[row],
                    global_explore_reid[row] ? 'G' : 'L', age);
                if (wrote <= 0 || (size_t)wrote >= sizeof(picks) - used) break;
                used += (size_t)wrote;
            }
            std::snprintf(line, sizeof(line),
                "[REID_SCAN] f=%d selected=%d budget=%d picks=%s",
                frame_count, (int)selected.size(), k_budget, picks);
            trace_push(line);
            if (ptz_identity_recovery) {
                std::snprintf(line, sizeof(line),
                    "[IDENTITY_REID_SCAN] f=%d mode=ptz_anchor_local_plus_global"
                    " local_slots=%d global_row=%d selected=%d budget=%d",
                    frame_count, local_slots, ptz_global_explore_row,
                    (int)selected.size(), k_budget);
                trace_push(line);
            }
        }
    }

    for (int k = 0; k < near_box.rows; ++k) {
        cv::Mat box = near_box.row(k);
        cv::Mat det_bbox = box.colRange(0, 4);
        int src = static_cast<int>(box.at<float>(0, 4));
        int idx = static_cast<int>(box.at<float>(0, 5));

        Candidate c;
        c.near_idx  = k;
        c.source    = src;
        c.det_index = idx;
        c.body_hyp_id = (k < (int)body_hyp_for_row.size()) ? body_hyp_for_row[k] : -1;
        c.global_explore = k < (int)global_explore_reid.size()
                        && global_explore_reid[k] != 0;

        // (a) IoU
        c.iou = ptz_identity_recovery ? 0.f : Utils::iou_single(trk_box, det_bbox);

        // (b)(b2) ReID + Anchor——未调度候选是 identity unknown，不得把 0 分
        // 解释为“已经识别为不同人”，也不得靠纯几何参与主身份提交。
        if (do_reid[k]) {
            cv::Mat cand_feature = compute_embedding(img, det_bbox);
            c.emb = cand_feature;              // 保存（已归一化）供外观排除
            // A12：ReID 推理失败（NPU 故障帧）返回空特征；注册期失败则 match_emb 恒空。
            // 空 Mat 进 dot() 抛 cv::Exception —— 旧接口（Track 无 try/catch）下进程即死
            // → 看门狗整机重启。任一方为空 → 本帧无外观信号（0 分，会被 anchor 门自然
            // 拦下），交由空间/头部信号兜底，宁可漏配不崩溃。
            c.reid_evaluated = !cand_feature.empty();
            c.reid_sim = (cand_feature.empty() || match_emb.empty())
                       ? 0.f : (float)cand_feature.dot(match_emb);
            // Q1：按主目标当前可见度带比对（半身查询↔半身参考）——本带无样本时 anchor_sim_vis
            //   内部回退 anchor_sim_max（原行为）。这样低可见度下真目标的 anchor 不再被全身参考误低，
            //   既救 fusion 权重也救 anchor 硬门（真目标不再被误删/误否决）。
            c.anchor_sim = cand_feature.empty() ? 0.f
                         : trackers[main_trk_idx]->has_anchor()
                         ? trackers[main_trk_idx]->anchor_sim_vis(cand_feature, visible_ratio_ema_)
                         : c.reid_sim;
            if (c.body_hyp_id >= 0) {
                for (auto& h : body_reid_hypotheses_) {
                    if (h.id == c.body_hyp_id) {
                        h.last_attempt_frame = frame_count;
                        if (c.reid_evaluated)
                            h.last_inferred_frame = frame_count;
                        break;
                    }
                }
            }
        } else {
            c.reid_sim   = 0.f;
            c.anchor_sim = 0.f;
            c.reid_evaluated = false;
        }

        c.oks_score = 0.f; c.body_shape_sim = 0.f; c.has_shape = false;
        c.has_shoulder = false; c.shoulder_cont = 0.f;

        // (e) 遮挡者排斥（排除法：不是遮挡者 → 就是主目标）
        //     查找遮挡者 tracker 当前预测位置，候选离它越远越好
        c.anti_occ = 1.f;  // 默认：无遮挡者信息 → 不惩罚
        if (!ptz_identity_recovery && occluder_tracker_id_ >= 0 && in_danger) {
            for (int t = 0; t < (int)trackers.size(); ++t) {
                if (trackers[t]->get_id() == occluder_tracker_id_) {
                    cv::Mat occ_pred = info.trks.row(t);
                    float occ_iou = Utils::iou_single(occ_pred, det_bbox);
                    c.anti_occ = 1.f - occ_iou;
                    break;
                }
            }
        }

        // (f) 头部一致性：候选关联头部框 vs 主目标预测头部框 的标准化中心距分
        //     遮挡期人体框互相重叠 → body IoU 被置 0（陷阱）；头部很少重叠 →
        //     头部位置是该窗口最可靠的空间判别信号。改用标准化中心距分（head_match）
        //     而非裸 IoU：头框小 + 帧间位移大时 IoU 会断崖归零（误否决真目标）。
        c.head_match = 0.f;
        c.has_head = false;
        if (!ptz_identity_recovery && head_track_fresh) {
            const cv::Mat& head_src = (src == 0) ? det_groups.dets_one_head
                                                 : det_groups.dets_second_head;
            if (idx >= 0 && idx < head_src.rows) {
                cv::Mat cand_head = head_src.row(idx).colRange(0, 4);
                if (cv::countNonZero(cand_head) > 0) {
                    c.head_match = head_match_score(main_head_pred, cand_head);
                    c.has_head = true;
                }
            }
        }

        // 评分推迟到 PASS 2（先收集所有候选信号，再判定外观歧义）
        candidates.push_back(c);
    }

    // ──────────────────────────────────────────────────────
    // 自适应融合权重（#2）：外观歧义检测
    //   ReID 与 anchor 使用不同的统计分布：前者是当前/遮挡前特征的余弦，后者是
    //   多样本画廊最大余弦。二者各自取最高两个候选；若都很高且差距极小，
    //   说明外观无法区分（典型同衣场景）→ 降低外观权重，把判别权重
    //   转移到运动/形状/遮挡排除等其余信号上。
    // ──────────────────────────────────────────────────────
    bool reid_ambiguous = false;
    {
        float r1 = -1.f, r2 = -1.f, a1 = -1.f, a2 = -1.f;
        for (const auto& cc : candidates) {
            if (cc.reid_sim > r1) { r2 = r1; r1 = cc.reid_sim; }
            else if (cc.reid_sim > r2) { r2 = cc.reid_sim; }
            if (cc.anchor_sim > a1) { a2 = a1; a1 = cc.anchor_sim; }
            else if (cc.anchor_sim > a2) { a2 = cc.anchor_sim; }
        }
        bool raw_reid_ambiguous = r1 > kReidAmbiguousMin && r2 > kReidAmbiguousMin
                               && (r1 - r2) < kReidAmbiguousGap;
        bool anchor_ambiguous = a1 > kAnchorAmbiguousMin && a2 > kAnchorAmbiguousMin
                             && (a1 - a2) < kAnchorAmbiguousGap;
        if (raw_reid_ambiguous || anchor_ambiguous) {
            reid_ambiguous = true;
            null_sink << "[ADAPTIVE] appearance ambiguous reid=(" << r1 << "," << r2
                         << ") anchor=(" << a1 << "," << a2 << ")"
                      << " -> down-weight appearance" << std::endl;
        }
    }

    // 融合权重（顺序：reid, shape, oks, iou, center, anti, head）
    //   基础表只填前 6 项，head 由后续按状态自适应注入（聚合初始化余项自动置 0）
    struct FuseWeights { float reid, shape, oks, iou, center, anti, head, shoulder; };

    // ── 可见度可靠性乘子（按 visibility_state_ 选）──
    //   ReID/shape/oks 需较完整躯干 → 半身/上身大幅降权；
    //   iou/center/anti（空间 + 排除）相对鲁棒 → 维持或略升；
    //   vis_head_floor：头部预算下限（躯干越少越依赖头部 KF 空间判别）。
    //   FULL 全为 1.0、floor=0.10（= 原 CLEAR head 预算）→ 常见路径行为不变。
    float vr_reid = 1.f, vr_shape = 1.f, vr_oks = 1.f,
          vr_iou = 1.f, vr_center = 1.f, vr_anti = 1.f;
    float vis_head_floor = 0.10f;
    switch (visibility_state_) {
        case VisibilityState::FULL:
            vr_reid=1.0f; vr_shape=1.0f; vr_oks=1.0f; vr_iou=1.0f; vr_center=1.0f; vr_anti=1.0f;
            vis_head_floor=0.08f; break;
        case VisibilityState::MOSTLY_FULL:
            vr_reid=0.9f; vr_shape=0.9f; vr_oks=0.8f; vr_iou=1.0f; vr_center=1.0f; vr_anti=1.0f;
            vis_head_floor=0.10f; break;
        case VisibilityState::HALF:
            vr_reid=0.75f; vr_shape=0.4f; vr_oks=0.15f; vr_iou=0.9f; vr_center=1.1f; vr_anti=1.0f;
            vis_head_floor=0.15f; break;
        case VisibilityState::UPPER:
            // Q1：anchor 现按可见度带比对（上身查询↔上身参考，见 anchor_sim_vis）→ 低可见度 reid
            //   不再是死重；新 ReID 模型更可信，上身也给到中等权重，但仍低于 HALF。
            vr_reid=0.45f; vr_shape=0.05f; vr_oks=0.0f; vr_iou=0.6f; vr_center=1.2f; vr_anti=0.9f;
            vis_head_floor=0.22f; break;
        case VisibilityState::HEAD_ONLY:
            // 仅头部可见时 ReID 仍降权，主身份以人脸/头部 KF 为主；保留少量强外观纠偏能力。
            vr_reid=0.20f; vr_shape=0.0f; vr_oks=0.0f; vr_iou=0.4f; vr_center=1.3f; vr_anti=0.8f;
            vis_head_floor=0.30f; break;
    }

    // ──────────────────────────────────────────────────────
    // 朝向可靠性（第三可靠性轴）：转身/光照 → 外观退化 → 路由到肩部连续性
    //   appearance_rel ∈ [0,1]：1=外观完全可信；<1=转身(frontalness)或光照突变致退化。
    //   正面/未知(frontalness_<0) → rel=1 → 乘子=1、注入=0 → 常见路径行为完全不变。
    // ──────────────────────────────────────────────────────
    float appearance_rel = 1.f;
    if (frontalness_ >= 0.f)
        appearance_rel = std::min(1.f, std::max(0.f, frontalness_ / kFrontalLowThresh));
    bool lighting_suspect = false;
    if (shoulder_geom_valid_ && prev_incumbent_anchor_ >= kLightingPrevAnchorMin) {
        // 光照突变：正面(rel 高)但"几何连续的本人"anchor 较上帧骤降 → 外观才是不可信方
        float inc_cont = -1.f, inc_anchor = 0.f;
        for (const auto& cc : candidates)
            if (cc.has_shoulder && cc.shoulder_cont > inc_cont) {
                inc_cont = cc.shoulder_cont; inc_anchor = cc.anchor_sim;
            }
        if (inc_cont >= kShoulderContIncumbent
            && (prev_incumbent_anchor_ - inc_anchor) >= kAnchorDropSuspect) {
            lighting_suspect = true;
            appearance_rel = std::min(appearance_rel, kLightingRel);
        }
    }
    bool  orient_engaged = (appearance_rel < kRelEngage);
    float or_reid  = kOrientReidMul  + (1.f - kOrientReidMul)  * appearance_rel;
    float or_shape = kOrientShapeMul + (1.f - kOrientShapeMul) * appearance_rel;
    // 有界保持窗口：几何仅"维持"不"提交"，超 kOrientHoldMaxMs 回落严格档 → coast
    int64_t now_orient_ms = now_ms();
    if (orient_engaged) { if (orient_low_since_ms_ < 0) orient_low_since_ms_ = now_orient_ms; }
    else                { orient_low_since_ms_ = -1; }
    bool orient_hold_ok = orient_engaged && orient_low_since_ms_ >= 0
                        && (now_orient_ms - orient_low_since_ms_) <= kOrientHoldMaxMs;
    if (orient_engaged) {
        null_sink << "[ORIENT] f=" << frame_count << " frontal=" << frontalness_
                     << " rel=" << appearance_rel << (lighting_suspect ? " (lighting)" : " (turn)")
                     << " or_reid=" << or_reid << " hold_ok=" << orient_hold_ok << std::endl;
    }

    // ──────────────────────────────────────────────────────
    // PASS 2：按遮挡状态选基础权重 → 可见度重加权 → 歧义自适应 → 计算 total
    // ──────────────────────────────────────────────────────
    for (auto& c : candidates) {
        if (!c.reid_evaluated) {
            c.total = -1.f;
            continue;
        }
        float reid_combined = std::max(c.reid_sim, c.anchor_sim);

        // 注：body_shape（w.shape）已降权为"软性 tie-breaker"。骨骼比例是 PTZ 距离/
        //   视角下的弱生物特征，新度量虽有动态范围但仍噪声大 → 只给小权重打破平局，
        //   把判别预算让给 reid/空间/排除（anti）。has_shape 的权重对比 else 分支即体现
        //   "省下的 shape 预算转移到了 reid/iou/anti"。
        FuseWeights w{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        if (ptz_identity_recovery) {
            w = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        } else if (occlusion_state_ == OcclusionState::OCCLUDED) {
            // 遮挡中：纯身份 + 遮挡者排斥（IoU/OKS 是陷阱）
            w = {0.70f, 0.f, 0.f, 0.f, 0.f, 0.30f};
        } else if (occlusion_state_ == OcclusionState::RECOVERING) {
            if (occ_kf_clean_) {
                // KF 干净：空间信号可信（iou/center）+ 身份/排除
                // center 槽(第5位)置 0：center_proximity 已停用（见下方 c.total 注释）；
                //   归一化会把这 0.20/0.25 预算按比例转给 reid/iou/anti（空间连续+排除）。
                w = {0.45f, 0.f, 0.f, 0.35f, 0.f, 0.20f};
            } else {
                // KF 脏：空间信号反向 → 退回身份 + 排除法
                w = {0.70f, 0.f, 0.f, 0.f, 0.f, 0.30f};
            }
        } else {
            // CLEAR：正常跟踪，三信号均衡
            w = {0.65f, 0.f, 0.f, 0.35f, 0.f, 0.f};
        }

        // ── 可见度重加权：半身/上身时 ReID/pose 退化 → 按可见度缩放后归一 ──
        //   缩放后重新归一到和为 1，以保持下方歧义/头部注入的"和恒为 1"不变量。
        //   FULL 时 vr 全为 1 → 缩放是恒等、归一无效果 → 行为与原先一致。
        {
            w.reid   *= vr_reid;
            w.shape  *= vr_shape;
            w.oks    *= vr_oks;
            w.iou    *= vr_iou;
            w.center *= vr_center;
            w.anti   *= vr_anti;
            // 朝向门（第三可靠性轴）：转身/光照降 ReID/shape，与可见度乘子同处、共享下方归一
            w.reid   *= or_reid;
            w.shape  *= or_shape;
            float vsum = w.reid + w.shape + w.oks + w.iou + w.center + w.anti;
            if (vsum > 1e-6f) {
                float inv = 1.0f / vsum;
                w.reid *= inv; w.shape *= inv; w.oks *= inv;
                w.iou  *= inv; w.center *= inv; w.anti *= inv;
            } else {
                w = {0.f, 0.f, 0.f, 1.0f, 0.f, 0.f};  // 退化：全压 iou（KF 框邻近，目标相对量；不再用画面中心先验）
            }
        }

        // ── 外观歧义自适应：砍掉 60% 外观权重，等比补到其余判别信号 ──
        //   cut 从 reid 转移到 pool（其余权重之和）→ 权重总和恒为 1
        if (reid_ambiguous) {
            float cut  = w.reid * 0.6f;
            float pool = 1.0f - w.reid;
            w.reid -= cut;
            if (pool > 1e-6f) {
                float kk = 1.0f + cut / pool;
                w.shape *= kk; w.oks *= kk; w.iou *= kk;
                w.center *= kk; w.anti *= kk;
            }
        }

        // ── 头部信号注入（按状态自适应预算）──
        //   仅当该候选有头部、且主目标头部预测新鲜时注入；
        //   等比缩小其余权重后留出 head 预算 → 权重总和恒为 1。
        //   遮挡期 body IoU 是陷阱（已置 0）→ 给头部最高预算作为主空间判别。
        if (!ptz_identity_recovery && c.has_head) {
            float hb;
            if (occlusion_state_ == OcclusionState::OCCLUDED)
                hb = 0.22f;
            else if (occlusion_state_ == OcclusionState::RECOVERING)
                hb = occ_kf_clean_ ? 0.15f : 0.22f;
            else
                hb = 0.08f;  // CLEAR：头部只作 tie-breaker/否决，身份仍由人体外观主导
            // 可见度下限：躯干露出越少（UPPER/HEAD_ONLY）越依赖头部空间判别
            hb = std::max(hb, vis_head_floor);
            float keep = 1.0f - hb;
            w.reid *= keep; w.shape *= keep; w.oks *= keep; w.iou *= keep;
            w.center *= keep; w.anti *= keep;
            w.head = hb;
        }

        // ── 肩部连续性注入（转身/光照期：把降权的外观预算路由到躯干几何）──
        //   仅当朝向门启用且候选有肩：按 (1-rel) 分配预算，等比缩其余(含 head)保持和=1。
        //   正面/未知 → orient_engaged=false → 不注入 → 行为不变。
        if (!ptz_identity_recovery && orient_engaged && shoulder_geom_valid_ && c.has_shoulder) {
            float sb = kShoulderContBudget * (1.f - appearance_rel);
            if (sb > 1e-3f) {
                float keep = 1.f - sb;
                w.reid *= keep; w.shape *= keep; w.oks *= keep; w.iou *= keep;
                w.center *= keep; w.anti *= keep; w.head *= keep;
                w.shoulder = sb;
            }
        }

        // center_proximity 已移除：遮挡时遮挡者继承画面中心（云台原先把主目标居中）→
        //   该先验反而帮遮挡者夺锁；它是"位置先验"而非身份证据，且依赖云台居中假设。
        //   w.center 基础权重已全部置 0，预算经上面的归一化按比例转给 reid/iou/anti/head。
        //   （c.center_proximity 已彻底删除：连预排(ReID 候选筛选)也不再用中心先验，
        //    中心先验完全退出匹配路径；w.center 槽保留为 0 仅为权重向量定长。）
        c.total = w.reid   * reid_combined
                + w.shape  * c.body_shape_sim
                + w.oks    * c.oks_score
                + w.iou    * c.iou
                + w.anti   * c.anti_occ
                + w.head   * c.head_match
                + w.shoulder * c.shoulder_cont;

        null_sink << "cand[" << c.near_idx << "] reid=" << c.reid_sim
                  << " anchor=" << c.anchor_sim
                  << " shape=" << c.body_shape_sim
                  << " oks=" << c.oks_score
                  << " iou=" << c.iou
                  << " anti_occ=" << c.anti_occ
                  << " head=" << c.head_match << (c.has_head ? "(Y)" : "(N)")
                  << " total=" << c.total
                  << " state=" << (int)occlusion_state_
                  << (occ_kf_clean_ ? " kf=clean" : " kf=dirty")
                  << " vis=" << (int)visibility_state_
                  << "/" << visible_ratio_ema_
                  << (reid_ambiguous ? " AMB" : "")
                  << " src=" << c.source << " idx=" << c.det_index << std::endl;
    }

    // ── 人脸锁硬保持（FACE_HOLD，防人脸↔特征左右摇摆的主机制）──
    //   人脸确认后 TTL 内(face_lock_active)：把"空间延续 confirmed 轨迹"的候选直接判为主目标——
    //   给决定性加成(kFaceHoldBonus，排序永居首 + 歧义 gap 拉大) + 下方 veto 循环放宽其 anchor 硬门
    //   (身份已由人脸确定，新模型半身外观弱不该否决它) → 任何非人脸特征都夺不走锁。
    //   关键：锚定 confirmed 轨迹自身的前向估计(KF 预测 trk_box / last_obs 取 IoU 大者)，而非"上帧
    //   赢家"last_obs——硬保持使真目标每帧都赢 → 参考永不漂到冒充者，自洽(修复旧粘滞"加到错的匹配上")。
    //   无空间延续候选(真目标确已离开/被挡) → 不保持 → 落 coast/重捕，绝不切到特征偏好的陌生人。
    //   仍受 coexist 否决(已确认他人不被保持)约束(在下方 veto 循环)。不受 id_switch_alert_ 抑制(身份=人脸)。
    // 陈旧 face lock 只能保护孤立场景的空间延续；多人交错时，单凭 2.4s 内的旧脸
    // 去给最大 IoU 人体硬加成，可能把交错后的另一人当成“同一条轨迹”。当前帧真正
    // 的人脸识别仍在 STEP 3 保持最高优先级，可直接覆盖人体匹配。
    if (!ptz_identity_recovery && face_lock_active && close_det_count <= 1) {
        cv::Mat last_obs = trackers[main_trk_idx]->get_last_observation();
        bool has_lo = !last_obs.empty() && cv::sum(last_obs)[0] >= 0.f;
        int hold_ci = -1; float hold_iou = kFaceHoldMinIou;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            cv::Mat cb = near_box.row(candidates[ci].near_idx).colRange(0, 4);
            float io_kf = Utils::iou_single(trk_box, cb);            // KF 前向预测（confirmed 轨迹）
            float io_lo = has_lo ? Utils::iou_single(last_obs, cb) : 0.f;
            float io = std::max(io_kf, io_lo);
            if (io > hold_iou) { hold_iou = io; hold_ci = (int)ci; }
        }
        if (hold_ci >= 0) {
            candidates[hold_ci].total    += kFaceHoldBonus;
            candidates[hold_ci].face_hold = true;
            null_sink << "[FACE_HOLD] hard hold k=" << candidates[hold_ci].near_idx
                         << " iou=" << hold_iou << " (identity=face, features can't switch)" << std::endl;
        } else {
            null_sink << "[FACE_HOLD] no spatially-continuous candidate -> coast/reacq (no switch)"
                         << std::endl;
        }
    }
    // ── 非人脸在位者迟滞（G1）：给"上帧锁定的物理人"近分平局优先，防同衣目标间逐帧翻转 ──
    //   在位者 = 与主 tracker 上一帧真实命中框(last_observation)空间连续的候选（≠画面中心先验）。
    //   加成只影响排序 / 歧义 gap，不削弱任何硬否决。仅"有邻人 且 非警报期"时启用（有脸锁时走上面的硬保持）。
    if (!ptz_identity_recovery && !face_lock_active && close_det_count >= 2 && !id_switch_alert_) {
        cv::Mat last_obs = trackers[main_trk_idx]->get_last_observation();
        int main_tsu_inc = trackers[main_trk_idx]->get_time_since_update();
        int64_t inc_blind_ms = get_body_blind_ms();
        if (!last_obs.empty() && cv::sum(last_obs)[0] >= 0.f
            && main_tsu_inc <= kIncumbentMaxTsu
            && inc_blind_ms >= 0 && inc_blind_ms <= kIncumbentMaxMs) {
            int inc_ci = -1; float inc_iou = kIncumbentMinIou;
            for (size_t ci = 0; ci < candidates.size(); ++ci) {
                cv::Mat cb = near_box.row(candidates[ci].near_idx).colRange(0, 4);
                float io = Utils::iou_single(last_obs, cb);
                if (io > inc_iou) { inc_iou = io; inc_ci = (int)ci; }
            }
            if (inc_ci >= 0) {
                candidates[inc_ci].total += kIncumbentHysteresis;
                null_sink << "[INCUMBENT] +" << kIncumbentHysteresis
                             << " k=" << candidates[inc_ci].near_idx
                             << " iou_lastobs=" << inc_iou << std::endl;
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.total > b.total;
              });

    const float base_pose_gap = in_danger ? kAmbiguousGapDanger : kAmbiguousGapClear;
    if (reid_ambiguous && occlusion_state_ != OcclusionState::OCCLUDED
        && candidates.size() >= 2
        && (candidates[0].total - candidates[1].total) < base_pose_gap
        && candidates[0].reid_evaluated && candidates[1].reid_evaluated) {
        Candidate& a = candidates[0];
        Candidate& b = candidates[1];
        const int trace_a_source = a.source, trace_a_index = a.det_index;
        const int trace_b_source = b.source, trace_b_index = b.det_index;
        cv::Mat box_a = near_box.row(a.near_idx).colRange(0, 4);
        cv::Mat box_b = near_box.row(b.near_idx).colRange(0, 4);
        request_pose(img, box_a, a.source, a.det_index, PoseReason::AMBIGUITY_RESOLVE);
        request_pose(img, box_b, b.source, b.det_index, PoseReason::AMBIGUITY_RESOLVE);
        PoseCacheEntry* pa = find_cached_pose(a.source, a.det_index);
        PoseCacheEntry* pb = find_cached_pose(b.source, b.det_index);
        const bool paired = pa && pb
                         && pa->status == PoseRequestStatus::AVAILABLE
                         && pb->status == PoseRequestStatus::AVAILABLE;
        if (paired) {
            PoseScoreDetail da = compute_candidate_pose_detail(pa->pose, box_a, trk_box);
            PoseScoreDetail db = compute_candidate_pose_detail(pb->pose, box_b, trk_box);
            const bool compare_oks = da.has_oks && db.has_oks;
            const bool compare_shape = da.has_shape && db.has_shape;
            const bool compare_shoulder = da.has_shoulder && db.has_shoulder;
            auto refine = [&](Candidate& c, const PoseScoreDetail& d) {
                c.oks_score = d.oks; c.body_shape_sim = d.body_shape;
                c.has_shape = compare_shape; c.has_shoulder = compare_shoulder;
                c.shoulder_cont = compare_shoulder
                                ? shoulder_cont_score(d.shoulder_mid, trk_box) : 0.f;
                if (compare_oks) c.total += 0.12f * d.oks;
                if (compare_shape) c.total += 0.08f * d.body_shape;
                if (compare_shoulder && orient_engaged)
                    c.total += kShoulderContBudget * (1.f - appearance_rel)
                             * c.shoulder_cont;
            };
            refine(a, da);
            refine(b, db);
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& x, const Candidate& y) {
                          return x.total > y.total;
                      });
        }
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[POSE_PAIR] f=%d paired=%d used=%d a=%d/%d b=%d/%d",
                frame_count, paired ? 1 : 0, pose_budget_used_,
                trace_a_source, trace_a_index, trace_b_source, trace_b_index);
            trace_push(line);
        }
    }

    // 在普通 winner 仲裁前识别 exact BODY provisional continuation。
    // baseline 始终是 provisional 起始帧，绝不滚动更新，防止渐进漂移。
    if (pending_body_hyp_id_ >= 0) {
        int pending_ci = -1;
        for (int ci = 0; ci < (int)candidates.size(); ++ci) {
            Candidate& c = candidates[ci];
            if (c.body_hyp_id != pending_body_hyp_id_) continue;
            if (!c.reid_evaluated || c.emb.empty()) break;
            float center_dist = -1.f, box_iou = -1.f, size_ratio = -1.f;
            const cv::Mat body_box = near_box.row(c.near_idx).colRange(0, 4);
            if (!ptz_identity_recovery && !body_provisional_geometry(
                    body_box, center_dist, box_iou, size_ratio)) {
                clear_body_provisional(
                    "geometry_jump", "continuation_qualify",
                    c.source, c.det_index, c.reid_sim, c.anchor_sim,
                    center_dist, box_iou, size_ratio);
                break;
            }
            if (c.reid_sim + kProvisionalScoreEpsilon < pending_body_reid_) {
                clear_body_provisional(
                    "reid_degraded", "continuation_qualify",
                    c.source, c.det_index, c.reid_sim, c.anchor_sim,
                    center_dist, box_iou, size_ratio);
                break;
            }
            if (c.anchor_sim + kProvisionalScoreEpsilon < pending_body_anchor_) {
                clear_body_provisional(
                    "anchor_degraded", "continuation_qualify",
                    c.source, c.det_index, c.reid_sim, c.anchor_sim,
                    center_dist, box_iou, size_ratio);
                break;
            }
            if (!body_provisional_scores_stable(c.reid_sim, c.anchor_sim)) break;
            c.qualified_pending = true;
            c.pending_center_dist = center_dist;
            c.pending_box_iou = box_iou;
            c.pending_size_ratio = size_ratio;
            pending_ci = ci;
            if (ptz_identity_recovery && kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[IDENTITY_PROVISIONAL] f=%d action=continue hyp=%d r=%.2f A=%.2f"
                    " prevR=%.2f prevA=%.2f gate=identity_only",
                    frame_count, c.body_hyp_id, c.reid_sim, c.anchor_sim,
                    pending_body_reid_, pending_body_anchor_);
                trace_push(line);
            }
            break;
        }
        // 仅改变活跃且合格 provisional 的仲裁顺序；无 pending 场景完全不变。
        if (pending_ci > 0) {
            std::rotate(candidates.begin(), candidates.begin() + pending_ci,
                        candidates.begin() + pending_ci + 1);
        }
    }

    if (kMatchTrace && close_det_count >= 2 && !candidates.empty()) {
        const Candidate& a = candidates[0];
        const Candidate* b = candidates.size() > 1 ? &candidates[1] : nullptr;
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[CAND2] f=%d a=%d/%d r=%.2f A=%.2f i=%.2f h=%.2f t=%.2f"
            " b=%d/%d r=%.2f A=%.2f i=%.2f h=%.2f t=%.2f",
            frame_count, a.source, a.det_index, a.reid_sim, a.anchor_sim,
            a.iou, a.head_match, a.total,
            b ? b->source : -1, b ? b->det_index : -1,
            b ? b->reid_sim : 0.f, b ? b->anchor_sim : 0.f,
            b ? b->iou : 0.f, b ? b->head_match : 0.f, b ? b->total : 0.f);
        trace_push(line);
    }

    // 直证所需的 ReID 次优差：只比较本帧确实完成 ReID 推理的候选；未推理候选的
    // reid_sim 为 0，不能将其当成真实的低分竞争者。单一已评估候选没有次优者，
    // 仍由共存/头部/运动等硬否决保护。
    auto reid_margin_to_runner = [&](const Candidate& candidate, bool& has_runner) -> float {
        float runner = -1.f;
        for (const auto& other : candidates) {
            if (&other == &candidate || !other.reid_evaluated || other.emb.empty()) continue;
            runner = std::max(runner, other.reid_sim);
        }
        has_runner = runner >= 0.f;
        return has_runner ? candidate.reid_sim - runner : FLT_MAX;
    };

    auto reid_direct_threshold = [&]() -> float {
        switch (visibility_state_) {
            case VisibilityState::HALF:
            case VisibilityState::UPPER:
                return kReidDirectConfirmPartial;
            case VisibilityState::HEAD_ONLY:
                return FLT_MAX;  // 仅头部不把 body ReID 作为身份直证
            case VisibilityState::FULL:
            case VisibilityState::MOSTLY_FULL:
            default:
                return kReidDirectConfirmFull;
        }
    };

    auto has_strong_reid_direct = [&](const Candidate& candidate) -> bool {
        if (!candidate.reid_evaluated) return false;
        bool has_runner = false;
        float margin = reid_margin_to_runner(candidate, has_runner);
        int evaluated_count = 0;
        for (const auto& cc : candidates)
            if (cc.reid_evaluated && !cc.emb.empty()) ++evaluated_count;
        // 全图多人搜索中若 NPU 只成功返回了一个 embedding，不能把“其他人尚未知”
        // 当作无限大的领先差；需等下一帧或依赖极强 anchor/人脸。
        if (body_global_search && candidates.size() > 1 && evaluated_count < 2
            && candidate.anchor_sim < kAnchorDirectConfirm)
            return false;
        return candidate.reid_sim >= reid_direct_threshold()
            && (!has_runner || margin >= kReidDirectMargin)
            && (candidate.anchor_sim >= kReidDirectAnchorFloor
                || !trackers[main_trk_idx]->has_anchor());
    };

    // ──────────────────────────────────────────────────────
    // STEP 2：多条件门控选择最佳候选
    // ──────────────────────────────────────────────────────
    MainMatchResult best;
    best.matched = false;
    bool diagnostic_spatial_hold = false;
    int diagnostic_spatial_hold_source = -1;
    int diagnostic_spatial_hold_index = -1;
    // 若赢家通过“强主身份 vs 二级轨迹身份”冲突纠错，最终真正提交后再隔离该
    // 二级轨迹。先只记索引，避免候选随后被其它门控拒绝时误伤合法他人轨迹。
    int best_ownership_repair_trk = -1;
    int best_ownership_repair_src = -1;
    int best_ownership_repair_det = -1;
    std::vector<int> coexist_repair_trk(candidates.size(), -1);

    // ── 遮挡期单候选保护 ──
    // 深度遮挡（overlap>=2）且只有 1 个候选 → 该候选极可能是
    // 站在前面的遮挡者，真正的主目标被挡在后面不可见。
    // 此时匹配 = 必然匹配错 → 拒绝匹配，让 KF 纯滑行。
    bool occ_single_reject = false;
    if (occlusion_state_ == OcclusionState::OCCLUDED
        && overlap_count >= 2
        // 全图池里可能还有很多无关远端人体；“深遮挡只剩一个候选”必须仍按
        // 原局部搜索池判断，否则扩大候选集合会意外解除这道遮挡者保护。
        && proximity.match_candidates.rows <= 1) {
        occ_single_reject = true;
        null_sink << "[OCC_PROTECT] single candidate during deep occlusion, skip match"
                  << std::endl;
    }

    // 门控阈值
    //
    // 设计原则：
    //   CLEAR  → 宽松，保证正常跟踪不丢人
    //   danger → 收紧身份门槛 + 遮挡者排斥 + 歧义检测
    {
        float anchor_veto_thresh = in_danger ? 0.50f : 0.40f;
        // body_shape 硬否决已移除：旧"全正向量余弦"对真人恒 ~0.99 → 永不否决冒充者，
        //   只在姿态被误测时（常发生于真目标半遮挡）才跌破阈值 → 误否决真目标。
        //   现 body_shape 仅作软性 tie-breaker（见融合权重 w.shape），不再设硬门槛。

        // ── 人脸锁期间收紧身份门槛（#1）──
        // 刚确认过人脸 → 身份已知 → 提高 anchor 底线，让同衣冒充者更难蒙混过关。
        if (face_lock_active && in_danger) {
            anchor_veto_thresh += kFaceLockVetoBoost;
            null_sink << "[FACE_LOCK] active, raise veto: anchor>="
                      << anchor_veto_thresh << std::endl;
        }

        // ── 同衣运动否决：预计算参考中心 + 各候选到参考点距离 ──
        //   参考中心：主目标丢失期用引导中心（预测轨迹点），否则用 KF 预测框中心。
        float mv_ref_cx, mv_ref_cy;
        int   main_tsu_mv = trackers[main_trk_idx]->get_time_since_update();
        if (main_tsu_mv > 0 && lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
            mv_ref_cx = lead_cx_; mv_ref_cy = lead_cy_;
        } else {
            mv_ref_cx = (trk_box.at<float>(0,0) + trk_box.at<float>(0,2)) * 0.5f;
            mv_ref_cy = (trk_box.at<float>(0,1) + trk_box.at<float>(0,3)) * 0.5f;
        }
        float mv_main_w = trk_box.at<float>(0,2) - trk_box.at<float>(0,0);
        float mv_main_h = trk_box.at<float>(0,3) - trk_box.at<float>(0,1);
        float mv_main_diag = std::sqrt(mv_main_w*mv_main_w + mv_main_h*mv_main_h);
        auto cand_pred_dist = [&](const Candidate& cc) -> float {
            cv::Mat b = near_box.row(cc.near_idx).colRange(0, 4);
            float bx = (b.at<float>(0,0) + b.at<float>(0,2)) * 0.5f;
            float by = (b.at<float>(0,1) + b.at<float>(0,3)) * 0.5f;
            return std::sqrt((bx-mv_ref_cx)*(bx-mv_ref_cx) + (by-mv_ref_cy)*(by-mv_ref_cy));
        };
        // ── 共存排除预计算（确定性反夺锁 + 二级轨迹所有权纠错）──
        //   正常情况：候选压在一条长期共存、仍新鲜的确认他人轨迹上 → 硬否决。
        //   例外不是“高 anchor 就放行”，而是更窄的身份矛盾：候选已满足强 ReID 直证，
        //   同时其现有 embedding 明显更像主 anchor 而不像该二级轨迹保存的身份，或该二级
        //   轨迹自身已主目标化。此时说明纯 IoU 二级关联在交错中换了所有权；继续位置否决
        //   会形成“主候选→二级轨迹→下一帧继续否决主候选”的永久自锁。
        std::vector<char> coexist_identity_vetoed(candidates.size(), 0);
        std::vector<char> coexist_spatial_conflicted(candidates.size(), 0);
        std::vector<int> coexist_spatial_other(candidates.size(), -1);
        // 外观排除/所有权纠错的主目标基线（循环外取一次）。
        cv::Mat prim_anchor_emb = trackers[main_trk_idx]->get_anchor_emb();
        for (size_t ci2 = 0; ci2 < candidates.size(); ++ci2) {
            const Candidate& cc = candidates[ci2];
            cv::Mat cb = near_box.row(cc.near_idx).colRange(0, 4);
            for (int sti = 0; sti < (int)trackers.size(); ++sti) {
                auto& st = trackers[sti];
                if (st->get_is_main()) continue;
                if (st->quarantined_) continue;   // B3：隔离轨迹（疑影子）无否决权
                // P1 分级确认：full=coexist≥kCoexistConfirm（可单凭位置/单臂否决，行为同旧）；
                //   provisional=coexist∈[kCoexistProvisional,kCoexistConfirm)（刚进画面的路人）——
                //   否决权受限，须"位置 AND 外观"双证。堵住确认前的头 ~10 帧空窗（同衣者夺锁高发段），
                //   双证 + 污染护栏使误伤有界（自身碎轨与主 anchor 相近 → 被污染护栏挡下）。
                bool full_confirm = st->coexist_with_main >= kCoexistConfirm;
                if (st->coexist_with_main < kCoexistProvisional) continue;  // 连暂定门槛都不到 → 无否决权
                if (st->get_time_since_update() > kCoexistVetoMaxTsu) continue;
                // B6：新鲜度加墙钟上限——慢帧下"2 帧"可达 748ms，轨迹位置早已陈旧，
                //   幽灵框会误伤真目标。帧数条件 AND 墙钟条件。
                if (st->last_update_ms_ >= 0
                    && (now_match_ms - st->last_update_ms_) > kCoexistVetoMaxMs) continue;
                float pos_iou = Utils::iou_single(st->get_state(), cb);

                cv::Mat se = st->get_emb();
                bool has_sec_identity = !cc.emb.empty() && !se.empty();
                float sec_sim = has_sec_identity
                              ? (float)cc.emb.dot(se) : -1.f;
                float sec_main_sim = (!se.empty() && !prim_anchor_emb.empty())
                                   ? (float)se.dot(prim_anchor_emb) : -1.f;
                bool sec_mainlike = sec_main_sim >= kSecPollutionSim;
                bool strong_main_claim = has_strong_reid_direct(cc)
                                      && cc.anchor_sim >= kReacqAnchorConfident;
                bool main_beats_sec = has_sec_identity
                                   && (cc.anchor_sim - sec_sim) >= kSecOwnershipMainMargin;
                bool no_sec_identity_but_dual_strong = se.empty()
                                                    && cc.anchor_sim >= kAnchorDirectConfirm;
                bool ownership_reversal = full_confirm
                                       && pos_iou > kCoexistVetoIou
                                       && strong_main_claim
                                       && (main_beats_sec || sec_mainlike
                                           || no_sec_identity_but_dual_strong);

                if (ownership_reversal) {
                    if (coexist_repair_trk[ci2] < 0)
                        coexist_repair_trk[ci2] = sti;
                    null_sink << "[COEXIST_OWNERSHIP] candidate is strong main identity; other id="
                                 << st->get_id() << " lost veto authority"
                                 << " iou=" << pos_iou << " reid=" << cc.reid_sim
                                 << " anchor=" << cc.anchor_sim << " sec=" << sec_sim
                                 << " sec_main=" << sec_main_sim << std::endl;
                    if (kMatchTrace) {
                        char line[kTraceLineLen];
                        std::snprintf(line, sizeof(line),
                            "[COEXIST_GATE] f=%d action=ownership_challenge src=%d idx=%d"
                            " other=%d coexist=%d iou=%.2f r=%.2f A=%.2f sec=%.2f secMain=%.2f",
                            frame_count, cc.source, cc.det_index, st->get_id(),
                            st->coexist_with_main, pos_iou, cc.reid_sim, cc.anchor_sim,
                            sec_sim, sec_main_sim);
                        trace_push(line);
                        trace_event_pending_ = true;
                    }
                    continue;
                }

                // 二级轨迹自身已与主 anchor 高度相似时，它很可能早已吸入主目标；即使当前
                // 候选证据尚不足以立即纠错，也不能让这条污染轨迹充当“他人”否决源。
                if (sec_mainlike) {
                    if (kMatchTrace && full_confirm && pos_iou > kCoexistVetoIou) {
                        char line[kTraceLineLen];
                        std::snprintf(line, sizeof(line),
                            "[COEXIST_GATE] f=%d action=skip_polluted src=%d idx=%d"
                            " other=%d coexist=%d iou=%.2f secMain=%.2f",
                            frame_count, cc.source, cc.det_index, st->get_id(),
                            st->coexist_with_main, pos_iou, sec_main_sim);
                        trace_push(line);
                    }
                    continue;
                }

                // ── 臂 A：瞬时 IoU ── 无 GMC 时这里的 secondary 尚未经过本帧
                // detection correction，只是 prediction-only。它只能产生 spatial conflict，
                // 不能伪装成 confirmed-other 身份负证据。
                if (!ptz_identity_recovery && full_confirm && pos_iou > kCoexistVetoIou) {
                    if (gmc_enabled_) {
                        coexist_identity_vetoed[ci2] = 1;
                    } else {
                        coexist_spatial_conflicted[ci2] = 1;
                        coexist_spatial_other[ci2] = st->get_id();
                    }
                    if (kMatchTrace) {
                        char line[kTraceLineLen];
                        std::snprintf(line, sizeof(line),
                            "[COEXIST_GATE] f=%d action=%s evidence=%s"
                            " otherObservation=predicted src=%d idx=%d"
                            " other=%d coexist=%d iou=%.2f r=%.2f A=%.2f sec=%.2f secMain=%.2f",
                            frame_count,
                            gmc_enabled_ ? "reject_pos_gmc" : "hold_spatial",
                            gmc_enabled_ ? "identity_negative" : "spatial_conflict",
                            cc.source, cc.det_index, st->get_id(),
                            st->coexist_with_main, pos_iou, cc.reid_sim, cc.anchor_sim,
                            sec_sim, sec_main_sim);
                        trace_push(line);
                    }
                    if (gmc_enabled_) break;
                }
                // ── 臂 B：外观排除（带同衣相对余量护栏 + 污染护栏）──
                //   候选"明显更像某他人、而不像主目标" → 就是那个人 → 否决。
                //   仅对已算嵌入的候选(do_reid) × 已注册嵌入的他人轨迹生效。
                if (cc.emb.empty() || se.empty()) continue;
                if (sec_sim < kSecExclSimMin) continue;                 // 不够像该他人
                if (sec_sim - cc.anchor_sim < kSecExclMargin) continue; // 未"明显更像他人"（同衣护栏）
                // P1：暂定确认轨迹的外观排除须叠加位置佐证（站在他人处 且 明显更像他人）→ 误伤极低；
                //   full 确认轨迹保持原行为（臂 B 位置无关，此条对 full 恒为假、不改变行为）。
                if (!full_confirm && pos_iou <= kCoexistVetoIou) continue;
                // 颜色直方图佐证已移除（整体弃用，弊大于利）：排除改由 emb 相似度 + 同衣相对余量
                //   + 污染护栏 + 位置(暂定档)综合判定，不再要求颜色二次确认。
                coexist_identity_vetoed[ci2] = 1;
                null_sink << "[SEC_EXCL] cand k=" << cc.near_idx
                             << (full_confirm ? " (confirmed)" : " (provisional)")
                             << " vetoed by other id=" << st->get_id()
                             << " sec_sim=" << sec_sim << " prim=" << cc.anchor_sim
                             << " iou=" << pos_iou << std::endl;
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[COEXIST_GATE] f=%d action=reject_identity src=%d idx=%d"
                        " other=%d coexist=%d iou=%.2f r=%.2f A=%.2f sec=%.2f secMain=%.2f",
                        frame_count, cc.source, cc.det_index, st->get_id(),
                        st->coexist_with_main, pos_iou, cc.reid_sim, cc.anchor_sim,
                        sec_sim, sec_main_sim);
                    trace_push(line);
                }
                break;
            }
        }

        // same-cloth 相对距离的比较基准不得包含已由真实 secondary identity
        // 明确排除的候选；否则 known-other 会反向制造真目标的 motion conflict。
        float mv_min_dist = FLT_MAX;
        for (size_t ci2 = 0; ci2 < candidates.size(); ++ci2) {
            if (candidates[ci2].reid_evaluated
                && !coexist_identity_vetoed[ci2]) {
                mv_min_dist = std::min(mv_min_dist,
                                       cand_pred_dist(candidates[ci2]));
            }
        }

        // ── M1（强化）：运动一致性否决 —— 见 .h ──
        //   已去掉 anchor 逃逸：本判别正是为同衣（anchor 不可信）而设；且真主目标不会
        //   落在"快速移动的非主轨迹"上（其检测归主轨迹，不入非主轨迹）→ 不会误伤；
        //   人脸仍可在 STEP 3 翻盘。新增方向判别：
        //     主目标静止 → 否决"明显移动"的候选；
        //     主目标移动 → 否决"运动方向与主目标明显相悖"的候选（同/横向不否决）。
        //   速度/方向取自 KF 状态（GMC 补偿，px/sec），仅速度显著时才信任方向。
        //
        // ⚠ 修复：原先用双阈值 main_stationary(<40) / main_moving(>75) 二选一触发，
        //   主目标速度落在 [40,75] 死区时两者皆 false → M1 整体不触发（站立主目标
        //   抖动进入死区即失去保护，正是"静止主目标被身后移动者夺锁"的成因之一）。
        //   改为单阈值：主目标"未明显移动"(<=mv_move_thr) 即按静止处理（否决任何
        //   明显移动的候选）；仅"明显移动"时才放宽为方向判别。消除死区。
        float mv_move_thr = kCandMovingPxSec;                    // "明显移动"下限(px/s)
        float     main_kspeed = trackers[main_trk_idx]->get_kf_speed_per_sec();
        cv::Vec2f main_kvel   = trackers[main_trk_idx]->get_kf_velocity_per_sec();
        bool main_clearly_moving = (main_kspeed > mv_move_thr);
        std::vector<char> motion_vetoed(candidates.size(), 0);
        // ── GMC 绑定：M1 的速度/方向取自 KF 状态，显式假设 GMC 已把相机运动补偿掉。
        //    GMC 关闭时 KF 速度含相机回扫分量（站立主目标在云台扫动下"看似在动"、
        //    静止背景轨迹也"在动"）→ 静止-移动判别与方向判别均失真，会误否决真目标
        //    或漏否决冒充者。故 GMC 关闭时整体不启用 M1，motion_vetoed 全 0，身份保护
        //    交给与相机运动无关的 共存否决（瞬时 IoU 压在确认他人轨迹）/头部否决/anchor/人脸。
        if (!ptz_identity_recovery && gmc_enabled_) {
            for (size_t ci2 = 0; ci2 < candidates.size(); ++ci2) {
                const Candidate& cc = candidates[ci2];
                cv::Mat cb = near_box.row(cc.near_idx).colRange(0, 4);
                for (auto& st : trackers) {
                    if (st->get_is_main()) continue;
                    if (st->get_time_since_update() > kCoexistVetoMaxTsu) continue;
                    float tsp = st->get_kf_speed_per_sec();
                    if (tsp <= mv_move_thr) continue;                          // 候选轨迹须明显移动
                    if (Utils::iou_single(st->get_state(), cb) <= kMotionVetoAssocIou) continue;
                    if (!main_clearly_moving) {
                        motion_vetoed[ci2] = 1; break;                         // 主目标未明显移动 ≠ 移动检测
                    } else {
                        cv::Vec2f tv = st->get_kf_velocity_per_sec();          // 主目标移动：方向相悖才否决
                        float cosang = (main_kvel[0]*tv[0] + main_kvel[1]*tv[1])
                                     / (main_kspeed * tsp + 1e-6f);
                        if (cosang < kMotionDirCosVeto) { motion_vetoed[ci2] = 1; break; }
                    }
                }
            }
        }

        // ── 分离期消歧（S1）：遮挡后重现时防"干净的遮挡者夺锁"。见 .h kSepTraj* 说明 ──
        //   轨迹参考 = KF 纯预测框中心（遮挡期沿遮挡前速度 pure-predict，已含 pre_occ 运动意图）；
        //   reid_sim 危险期已用遮挡前快照 pre_occ_emb_ 计算(match_emb) → 外观分本身即基于干净参考。
        bool sep_mode = !ptz_identity_recovery && (occlusion_state_ == OcclusionState::RECOVERING
                         || (occlusion_state_ == OcclusionState::OCCLUDED && overlap_count <= 1))
                        && candidates.size() >= 2;
        // S1 轨迹参考：遮挡期改用预测浮现点（遮挡者远侧），否则 KF 纯预测框中心（原行为）。
        //   浮现点基于遮挡者几何、非脆弱速度外推 → 与 S1 初衷一致，只是把参考从"进入点(贴遮挡者)"
        //   移到"目标将浮现处" → 分离消歧不再偏向遮挡者。
        float sep_ref_cx = emergence_valid_ ? emergence_cx_
                         : (trk_box.at<float>(0, 0) + trk_box.at<float>(0, 2)) * 0.5f;
        float sep_ref_cy = emergence_valid_ ? emergence_cy_
                         : (trk_box.at<float>(0, 1) + trk_box.at<float>(0, 3)) * 0.5f;
        auto sep_traj_dist = [&](const Candidate& cc) -> float {
            cv::Mat b = near_box.row(cc.near_idx).colRange(0, 4);
            float bx = (b.at<float>(0, 0) + b.at<float>(0, 2)) * 0.5f;
            float by = (b.at<float>(0, 1) + b.at<float>(0, 3)) * 0.5f;
            return std::sqrt((bx - sep_ref_cx) * (bx - sep_ref_cx)
                           + (by - sep_ref_cy) * (by - sep_ref_cy)) / std::max(1.f, mv_main_diag);
        };
        if (sep_mode)
            null_sink << "[SEP] separation disambiguation active, cands=" << candidates.size() << std::endl;
        if (sep_mode && kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[SEP_REFERENCE] f=%d source=%s center=(%.1f,%.1f)"
                " permission=hold_only candidates=%zu",
                frame_count, emergence_valid_ ? "emergence" : "kf",
                sep_ref_cx, sep_ref_cy, candidates.size());
            trace_push(line);
        }

        std::vector<char> trajectory_spatial_conflicted(candidates.size(), 0);
        if (!ptz_identity_recovery && !gmc_enabled_ && reid_ambiguous && mv_min_dist < FLT_MAX) {
            for (size_t ci2 = 0; ci2 < candidates.size(); ++ci2) {
                if (!candidates[ci2].reid_evaluated
                    || coexist_identity_vetoed[ci2]) continue;
                const float cdist = cand_pred_dist(candidates[ci2]);
                trajectory_spatial_conflicted[ci2] =
                    cdist > kMotionVetoFactor * mv_min_dist
                    && (cdist - mv_min_dist)
                       > kMotionVetoMinGapDiag * mv_main_diag;
            }
        }

        // 无 GMC 的空间冲突只作用于最终 tentative BODY winner。首个已通过既有
        // 身份/安全门的冲突候选被保留为 HOLD winner，后续候选仍完成身份门评估；
        // 只有严格更强的 BODY ReID direct 可以以身份而非 motion 取代它。
        bool spatial_winner_hold = false;
        bool spatial_winner_strong_reid = false;
        int spatial_winner_source = -1, spatial_winner_index = -1;
        float spatial_winner_reid = -1.f, spatial_winner_anchor = -1.f;

        for (auto& c : candidates) {
            if (!c.reid_evaluated) continue;
            const bool qualified_pending = c.qualified_pending
                && pending_body_hyp_id_ >= 0
                && c.body_hyp_id == pending_body_hyp_id_;
            // ── 单候选保护：默认跳过；强 ReID/anchor 可绕过以减少穿人卡顿 ──
            bool strong_reid_direct = has_strong_reid_direct(c);
            bool strong_anchor_direct = c.anchor_sim >= kAnchorDirectConfirm;
            bool strong_appearance_direct = strong_reid_direct || strong_anchor_direct;

            if (occ_single_reject && !strong_appearance_direct) break;
            if (occ_single_reject && strong_appearance_direct) {
                null_sink << "[OCC_PROTECT_BYPASS] strong appearance admits single candidate"
                             << " reid=" << c.reid_sim << " anchor=" << c.anchor_sim << std::endl;
            }

            // ── anchor 硬否决：底线身份检查 ──
            //   宽容档（防"转身/走远即丢"）：CLEAR + 无邻人 + 候选与 KF 预测框
            //   空间连续（IoU≥kVetoRelaxIou）→ 极可能是正在转身/走远的本人
            //   （ID switch 需要第二个人在场，孤立场景不可能发生）→ 底线降至
            //   kAnchorVetoRelaxed，让背面/侧面/远距外观不被逐帧否决。
            //   危险期 / 有邻人 / 空间不连续时仍用严格档。
            float eff_anchor_veto = anchor_veto_thresh;
            // §1 防误检夺锁：仅主目标近期有真实身体命中，或刚由严格头/脸连续性
            // 确认仍在场时，才允许在空间连续前提下放宽 anchor 硬门。
            //   持续丢失/滑行期(main_tsu 大)放宽 = 把停在预测处的误检/陌生人以 0.28 低门收编成
            //   "满置信命中"(coast_weight_=1.0 误导下游云台，且用误检喂预测器) → 云台追向空区。
            //   此期改走严格档；真身份复捕交由 人脸(STEP3)/头部连续/分离策略(sep_mode 另有轨迹门)。
            int64_t mp_blind_ms = get_body_blind_ms();
            bool body_present = (main_tsu_mv <= kRelaxMaxMissFrames)
                             && (mp_blind_ms >= 0 && mp_blind_ms <= kRelaxMaxMissMs);
            bool face_present = last_face_identity_ms_ >= 0
                             && (now_match_ms - last_face_identity_ms_) <= kPartObsRecentMs;
            bool head_present = last_head_continuity_ms_ >= 0
                             && (now_match_ms - last_head_continuity_ms_) <= kPartObsRecentMs;
            // 头部不是身份直证：多人交错时绝不能单凭 head recent 降低 anchor 门。
            // 仅无竞争、且当前人体候选自己的头与主头轨迹高度连续时，允许维持本人。
            bool safe_head_presence = head_present && close_det_count <= 1
                                   && c.has_head && c.head_match >= kHeadBodyResumeMin;
            bool main_present = body_present || face_present || safe_head_presence;
            bool low_vis = (visibility_state_ == VisibilityState::UPPER
                         || visibility_state_ == VisibilityState::HEAD_ONLY);
            // [N2] spatial_cont 第三臂：候选中心到统一参考点(lead 优先)的归一化距离够近。
            //   无 GMC 时云台移动帧 KF 预测框系统性滞后 → 纯 IoU 臂失效；候选又无关联头时
            //   危险期严格档会误杀刚露出的真目标。中心距臂用真实观测锚定的 lead，含自运动松弛。
            bool spatial_cont_center = false;
            {
                cv::Mat _cb = near_box.row(c.near_idx).colRange(0, 4);
                float _ccx = (_cb.at<float>(0,0) + _cb.at<float>(0,2)) * 0.5f;
                float _ccy = (_cb.at<float>(0,1) + _cb.at<float>(0,3)) * 0.5f;
                float _rcx, _rcy;
                if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) { _rcx = lead_cx_; _rcy = lead_cy_; }
                else {
                    _rcx = (trk_box.at<float>(0,0) + trk_box.at<float>(0,2)) * 0.5f;
                    _rcy = (trk_box.at<float>(0,1) + trk_box.at<float>(0,3)) * 0.5f;
                }
                float _tw = trk_box.at<float>(0,2) - trk_box.at<float>(0,0);
                float _th = trk_box.at<float>(0,3) - trk_box.at<float>(0,1);
                float _tdiag = std::sqrt(_tw*_tw + _th*_th);
                float _cd = std::sqrt((_ccx-_rcx)*(_ccx-_rcx) + (_ccy-_rcy)*(_ccy-_rcy));
                float _allow = _tdiag * (kSpatialContCenterDiag
                             + kSpatialContRateDiagPerSec * frame_dt_sec_) + ego_shift_mag_;
                spatial_cont_center = (_cd <= _allow);
            }
            bool spatial_cont = !ptz_identity_recovery && ((c.iou >= kVetoRelaxIou)
                             || (c.has_head && c.head_match >= kHeadMatchVetoMin)
                             || spatial_cont_center);
            bool  veto_relaxed = false;
            // (1) CLEAR 孤立 + 空间连续（IoU≥kVetoRelaxIou）→ 正在转身/走远的本人（无邻人 = 无 ID
            //     switch 风险）→ 宽容档 kAnchorVetoRelaxed，让背面/侧面/远距外观不被逐帧否决。
            if (!in_danger && close_det_count <= 1 && c.iou >= kVetoRelaxIou && main_present) {
                veto_relaxed = true;
                eff_anchor_veto = std::min(eff_anchor_veto, kAnchorVetoRelaxed);
            }
            // (2) 危险/半身 + 空间连续 → 降到 kAnchorVetoDanger（新 ReID 模型对半身/遮挡真目标只给
            //     ~0.2，宽容档 0.28 仍会误否决 → 真目标被删、只剩干净遮挡者 = 必切/丢失，重捕也追不上）。
            //     身份改由 空间连续 + 相对排序(歧义门，新模型判别力强、相对可靠) + 共存/头部否决 + 人脸
            //     承担；绝对外观弱时不主导。仍要求 main_present（§1：持续丢失期不放宽，防停在预测处的
            //     误检/陌生人被低门收编）。远离预测的陌生人 spatial_cont=false → 仍走严格档，blast 有界。
            //   （UPPER/HEAD_ONLY 时融合已把 reid 权重降到 0.25/0.10，却仍用同一退化 embedding 做
            //     anchor 硬否决 = 自相矛盾；此处一并修正，同时覆盖危险期半身真目标。）
            if ((in_danger || low_vis) && spatial_cont && main_present) {
                veto_relaxed = true;
                const float danger_floor = close_det_count >= 2
                                         ? kAnchorVetoDangerCrowded : kAnchorVetoDanger;
                eff_anchor_veto = std::min(eff_anchor_veto, danger_floor);
            }
            // ── 朝向放宽（转身/光照期，仅"几何连续的本人"，有界窗口内）──
            //   转身背面 anchor 天然低；候选肩落在主目标预测处(shoulder_cont 高)=正在转身的本人。
            //   即便有邻人(严格档)也放宽底线，避免逐帧否决真目标 → 切他人。超时(orient_hold_ok=false)
            //   自动回落严格档，几何只维持不提交；位置不连续的陌生人(shoulder_cont 低)不放宽。
            if (orient_hold_ok && shoulder_geom_valid_
                && c.has_shoulder && c.shoulder_cont >= kShoulderContIncumbent) {
                veto_relaxed = true;
                eff_anchor_veto = std::min(eff_anchor_veto, kAnchorVetoRelaxTurned);
                if (kMatchTrace) {   // [N9] → trace 环
                    char _ol[kTraceLineLen];
                    std::snprintf(_ol, sizeof(_ol),
                        "[ORIENT_HOLD] f=%d anchor=%.2f scont=%.2f floor=%.2f",
                        frame_count, c.anchor_sim, c.shoulder_cont, eff_anchor_veto);
                    trace_push(_ol);
                    trace_event_pending_ = true;
                }
            }
            // ── S1 分离期排名不硬删：贴合遮挡前轨迹的候选(极可能是刚露出的真目标，外观退化致 anchor 天然低)
            //   放宽 anchor 硬门，避免其被删后只剩干净遮挡者 = 必切。远离轨迹者不放宽 → 陌生人不获益、blast 有界。
            if (sep_mode && sep_traj_dist(c) <= kSepTrajGateDiag) {
                veto_relaxed = true;
                eff_anchor_veto = std::min(eff_anchor_veto, kAnchorVetoDanger);  // 分离重现=半身外观弱 → 危险档
            }
            // ── 人脸锁硬保持：身份已由人脸确定 → 该"空间延续 confirmed 轨迹"候选的 anchor 硬门降到地板，
            //   新模型半身外观弱(~0.2)也不该否决它（配合上方 FACE_HOLD 决定性加成 → 任何非人脸特征夺不走锁）。
            //   仍受 coexist/头部 否决约束（已确认他人 / 头部明显错位者不被硬保持）。
            if (c.face_hold)
                eff_anchor_veto = std::min(eff_anchor_veto, kFaceHoldAnchorFloor);
            // 新 ReID 强直证：当前 embedding 与候选高度一致时，不再让陈旧/单视角 anchor 画廊
            // 把真目标误杀。仍要求 anchor 至少有低基线（或无 anchor），且后续共存/头部硬否决照常生效。
            if (strong_reid_direct) {
                veto_relaxed = true;
                eff_anchor_veto = std::min(eff_anchor_veto, std::max(0.f, c.anchor_sim - 1e-3f));
            }
            // 观测：低可见度放宽实际改变了结果（严格档会否决、宽容档放行该候选过 anchor 门）
            //   → 打点，便于现场用日志核对修复是否在该 ID switch 帧生效（std::cout，与 [MATCH] 同源）。
            if (kMatchTrace && low_vis && spatial_cont
                && c.anchor_sim < anchor_veto_thresh
                && c.anchor_sim >= eff_anchor_veto) {   // [N9] → trace 环
                char _ll[kTraceLineLen];
                std::snprintf(_ll, sizeof(_ll),
                    "[LOWVIS_ADMIT] f=%d anchor=%.2f strict=%.2f iou=%.2f head=%.2f src=%d idx=%d",
                    frame_count, c.anchor_sim, anchor_veto_thresh, c.iou, c.head_match,
                    c.source, c.det_index);
                trace_push(_ll);
                trace_event_pending_ = true;
            }
            if (c.anchor_sim < eff_anchor_veto) {
                null_sink << "[VETO] anchor=" << c.anchor_sim
                          << " < " << eff_anchor_veto
                          << (veto_relaxed ? (low_vis ? " (lowvis-relaxed)" : " (relaxed)") : "")
                          << std::endl;
                const bool reliable_full_body = c.reid_evaluated
                    && trackers[main_trk_idx]->has_anchor()
                    && (visibility_state_ == VisibilityState::FULL
                        || visibility_state_ == VisibilityState::MOSTLY_FULL);
                const bool strict_identity_reject = reliable_full_body
                    && !veto_relaxed
                    && std::fabs(eff_anchor_veto - anchor_veto_thresh) <= 1e-5f;
                if (strict_identity_reject) {
                    mark_body_identity_evidence(
                        near_box.row(c.near_idx).colRange(0, 4),
                        IdentityEvidence::NEGATIVE,
                        "strict_full_body_identity_reject");
                    if (qualified_pending)
                        clear_body_provisional(
                            "strict_body_identity_negative", "anchor_gate",
                            c.source, c.det_index, c.reid_sim, c.anchor_sim,
                            c.pending_center_dist, c.pending_box_iou,
                            c.pending_size_ratio);
                }
                continue;
            }

            // ── body_shape 硬否决已移除（弱生物特征 + 度量噪声，仅作软性 tie-breaker）──

            // ── 头部硬否决（遮挡期防 ID switch 的关键）──
            //   危险期 + 头部预测足够新鲜 + 候选确有头部 + 头部匹配分极低
            //   → 该候选头部明显不在主目标头部应在的位置 = 他人（遮挡者）→ 否决。
            //   用标准化中心距分（head_match）替代裸 IoU：容忍较大但合理的头部位移
            //   （veto 触发距离 ≈ 2.2 头部尺寸），避免帧间位移大时误否决真目标。
            //   头部缺失（has_head=false）不否决：缺失≠身份不符。
            //   头部关联在两人交错时可能短暂串到另一人体，因此它不能推翻已经满足
            //   “绝对 ReID + 次优分差 + anchor 地板”的人体身份直证。强 ReID 放行后仍
            //   受 coexist/motion 等他人轨迹硬否决约束；普通/弱外观候选仍执行头部否决。
            if (!ptz_identity_recovery && in_danger && head_track_fresh
                && head_tsu <= kHeadVetoMaxAge
                && c.has_head && c.head_match < kHeadMatchVetoMin) {
                if (!strong_reid_direct) {
                    null_sink << "[HEAD_VETO] head_match=" << c.head_match
                              << " < " << kHeadMatchVetoMin << " (tsu=" << head_tsu << ")" << std::endl;
                    if (kMatchTrace) {
                        char line[kTraceLineLen];
                        std::snprintf(line, sizeof(line),
                            "[HEAD_GATE] f=%d action=reject src=%d idx=%d h=%.2f tsu=%d r=%.2f A=%.2f",
                            frame_count, c.source, c.det_index, c.head_match, head_tsu,
                            c.reid_sim, c.anchor_sim);
                        trace_push(line);
                    }
                    if (qualified_pending)
                        clear_body_provisional(
                            "geometry_jump", "head_gate",
                            c.source, c.det_index, c.reid_sim, c.anchor_sim,
                            c.pending_center_dist, c.pending_box_iou,
                            c.pending_size_ratio);
                    continue;
                }
                null_sink << "[HEAD_VETO_BYPASS] strong ReID identity overrides noisy head geometry"
                             << " reid=" << c.reid_sim << " head=" << c.head_match << std::endl;
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[HEAD_GATE] f=%d action=reid_bypass src=%d idx=%d h=%.2f tsu=%d r=%.2f A=%.2f",
                        frame_count, c.source, c.det_index, c.head_match, head_tsu,
                        c.reid_sim, c.anchor_sim);
                    trace_push(line);
                }
            }

            // ── 共存排除 + M1 静止-移动否决（见上方预计算注释）──
            {
                int ci = (int)(&c - candidates.data());
                if (coexist_identity_vetoed[ci]) {
                    null_sink << "[COEXIST_VETO] candidate on known-other track"
                              << " (anchor=" << c.anchor_sim << ")" << std::endl;
                    mark_body_identity_evidence(
                        near_box.row(c.near_idx).colRange(0, 4),
                        IdentityEvidence::NEGATIVE, "confirmed_other_identity");
                    note_person_identity_ambiguity(
                        kPersonRiskKnownOther, close_det_count, false);
                    person_identity_risk_active = true;
                    if (qualified_pending)
                        clear_body_provisional(
                            "confirmed_other_identity", "coexist_identity",
                            c.source, c.det_index, c.reid_sim, c.anchor_sim,
                            c.pending_center_dist, c.pending_box_iou,
                            c.pending_size_ratio);
                    continue;
                }
                if (!ptz_identity_recovery && motion_vetoed[ci]) {
                    null_sink << "[MOTION_VETO] motion inconsistent with main "
                              << (main_clearly_moving ? "(direction)" : "(main~still)")
                              << " anchor=" << c.anchor_sim << std::endl;
                    if (qualified_pending)
                        clear_body_provisional(
                            "geometry_jump", "gmc_motion_gate",
                            c.source, c.det_index, c.reid_sim, c.anchor_sim,
                            c.pending_center_dist, c.pending_box_iou,
                            c.pending_size_ratio);
                    continue;
                }
            }

            // GMC 开启时保留原 same-cloth trajectory hard veto；无 GMC 的同一判据
            // 已在上方只标成 spatial conflict，不能在这里 continue 后让另一人获胜。
            if (!ptz_identity_recovery && gmc_enabled_ && reid_ambiguous) {
                float cdist = cand_pred_dist(c);
                if (cdist > kMotionVetoFactor * mv_min_dist
                    && (cdist - mv_min_dist) > kMotionVetoMinGapDiag * mv_main_diag) {
                    null_sink << "[MOTION_VETO] dist=" << cdist << " min=" << mv_min_dist
                              << " diag=" << mv_main_diag << " (ambiguous spatial outlier)" << std::endl;
                    if (qualified_pending)
                        clear_body_provisional(
                            "geometry_jump", "gmc_trajectory_gate",
                            c.source, c.det_index, c.reid_sim, c.anchor_sim,
                            c.pending_center_dist, c.pending_box_iou,
                            c.pending_size_ratio);
                    continue;
                }
            }

            // ── 歧义检测 ──
            //   与"已被共存排除"的候选不构成歧义——它是确认的他人，分数再接近
            //   也不该让真目标被拒。否则分离后（同衣遮挡者 vs 重现的主目标）
            //   两者分数永远接近 → 永远 AMBIGUOUS → 永远无法重捕（实测病灶）。
            // 歧义门锚定到"真正的赢家"（第一个通过所有否决的候选），而非固定的
            // candidates[0]：若原始最高分候选被 anchor/head/coexist/motion 否决并 continue，
            // 赢家会是排序更靠后的候选，此时仍需对它做近分歧义检查。runner 从 ci+1 起找，
            // 且跳过 coexist/motion 否决者（确认的他人分数再接近也不该阻断真目标重捕）。
            if (candidates.size() >= 2) {
                int ci = (int)(&c - &candidates[0]);
                int runner = -1;
                for (int q = ci + 1; q < (int)candidates.size(); ++q) {
                    if (candidates[q].reid_evaluated
                        && !coexist_identity_vetoed[q] && !motion_vetoed[q]) {
                        runner = q;
                        break;
                    }
                }
                if (runner >= 0) {
                    float gap = c.total - candidates[runner].total;
                    float gap_thresh = in_danger ? kAmbiguousGapDanger : kAmbiguousGapClear;
                    // strong_reid_direct 已验证相对所有“已评估 ReID”候选的最佳次优差，
                    // 不能只和 total 排名紧随者比较，否则会遗漏一个 total 较低但 ReID 更近的候选。
                    bool strong_reid_margin = strong_reid_direct;
                    bool strong_anchor_margin =
                        strong_anchor_direct
                        && (c.anchor_sim - candidates[runner].anchor_sim) >= kReidDirectMargin;
                    if (gap < gap_thresh && !strong_reid_margin
                        && !strong_anchor_margin && !qualified_pending) {
                        null_sink << "[AMBIGUOUS] gap=" << gap
                                  << " < " << gap_thresh << ", reject" << std::endl;
                        break;
                    } else if (gap < gap_thresh) {
                        null_sink << "[AMBIGUOUS_BYPASS] strong appearance margin admits"
                                     << " gap=" << gap
                                     << " reid_margin=" << (c.reid_sim - candidates[runner].reid_sim)
                                     << " anchor_margin=" << (c.anchor_sim - candidates[runner].anchor_sim)
                                     << " provisional=" << qualified_pending
                                     << std::endl;
                    }
                }
            }

            // ── S1 分离期轨迹保持：外观赢家 c 明显偏离遮挡前轨迹、而另有候选更贴轨迹、且 c 身份优势不决定性
            //   → c 极可能是干净的遮挡者、真目标是那个贴轨迹但退化的候选 → 本帧不提交，主目标 coast（保持），
            //   等真目标更完整 / 人脸(STEP3)再决。有界：分离窗口(RECOVERING≤kRecoveryMs)内；出窗即恢复常规提交。
            //   仅当"更贴轨迹者"非 coexist/motion 否决者（已确认他人不参与，避免拿遮挡者当真目标锚）。
            if (sep_mode) {
                int best_traj = -1; float best_td = FLT_MAX;
                for (size_t q = 0; q < candidates.size(); ++q) {
                    if (!candidates[q].reid_evaluated
                        || coexist_identity_vetoed[q] || motion_vetoed[q]) continue;
                    float td = sep_traj_dist(candidates[q]);
                    if (td < best_td) { best_td = td; best_traj = (int)q; }
                }
                if (best_traj >= 0 && &candidates[best_traj] != &c) {
                    float traj_gap = sep_traj_dist(c) - best_td;              // c 比最贴轨迹者远多少（对角线归一）
                    float app_lead = c.anchor_sim - candidates[best_traj].anchor_sim;
                    float reid_lead = c.reid_sim - candidates[best_traj].reid_sim;
                    bool strong_sep_direct =
                        strong_appearance_direct
                        && (app_lead >= kSepAppDominant || reid_lead >= kReidDirectMargin);
                    if (traj_gap > kSepTrajMargin && app_lead < kSepAppDominant
                        && !strong_sep_direct && !qualified_pending) {
                        null_sink << "[SEP_HOLD] appearance-winner off-trajectory traj_gap="
                                     << traj_gap << " app_lead=" << app_lead
                                     << " -> coast (hold)" << std::endl;
                        if (kMatchTrace) {
                            char line[kTraceLineLen];
                            std::snprintf(line, sizeof(line),
                                "[SEP_DECISION] f=%d action=hold source=%s"
                                " winner=%d/%d trajGap=%.3f appLead=%.3f reidLead=%.3f",
                                frame_count, emergence_valid_ ? "emergence" : "kf",
                                c.source, c.det_index, traj_gap, app_lead, reid_lead);
                            trace_push(line);
                        }
                        break;
                    } else if (traj_gap > kSepTrajMargin
                               && (strong_sep_direct || qualified_pending)) {
                        null_sink << "[SEP_BYPASS] strong appearance admits off-trajectory candidate"
                                     << " traj_gap=" << traj_gap
                                     << " app_lead=" << app_lead
                                     << " reid_lead=" << reid_lead << std::endl;
                        if (kMatchTrace) {
                            char line[kTraceLineLen];
                            std::snprintf(line, sizeof(line),
                                "[SEP_DECISION] f=%d action=%s source=%s"
                                " winner=%d/%d trajGap=%.3f appLead=%.3f reidLead=%.3f",
                                frame_count,
                                qualified_pending ? "provisional_bypass" : "identity_bypass",
                                emergence_valid_ ? "emergence" : "kf",
                                c.source, c.det_index, traj_gap, app_lead, reid_lead);
                            trace_push(line);
                        }
                    }
                }
            }

            const int candidate_ci = (int)(&c - candidates.data());
            const bool coexist_spatial =
                coexist_spatial_conflicted[candidate_ci] != 0;
            const bool trajectory_spatial =
                trajectory_spatial_conflicted[candidate_ci] != 0;
            const bool candidate_spatial_conflict =
                !ptz_identity_recovery && !gmc_enabled_ && (coexist_spatial || trajectory_spatial);
            if (candidate_spatial_conflict && !qualified_pending) {
                if (!spatial_winner_hold
                    || (strong_reid_direct && !spatial_winner_strong_reid)) {
                    spatial_winner_hold = true;
                    spatial_winner_strong_reid = strong_reid_direct;
                    spatial_winner_source = c.source;
                    spatial_winner_index = c.det_index;
                    spatial_winner_reid = c.reid_sim;
                    spatial_winner_anchor = c.anchor_sim;
                }
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[MOTION_CONFLICT] f=%d action=defer_winner_hold"
                        " src=%d idx=%d reason=%s other=%d identityNegative=0"
                        " strongBody=%d nextCandidatePrevented=1",
                        frame_count, c.source, c.det_index,
                        coexist_spatial && trajectory_spatial
                            ? "coexist_predicted+same_cloth_trajectory"
                            : coexist_spatial ? "coexist_predicted"
                                              : "same_cloth_trajectory",
                        coexist_spatial_other[candidate_ci],
                        strong_reid_direct ? 1 : 0);
                    trace_push(line);
                    trace_event_pending_ = true;
                }
                continue;
            }
            if (candidate_spatial_conflict && qualified_pending && kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[MOTION_CONFLICT] f=%d action=provisional_override"
                    " src=%d idx=%d hyp=%d reason=%s identityNegative=0",
                    frame_count, c.source, c.det_index, c.body_hyp_id,
                    coexist_spatial && trajectory_spatial
                        ? "coexist_predicted+same_cloth_trajectory"
                        : coexist_spatial ? "coexist_predicted"
                                          : "same_cloth_trajectory");
                trace_push(line);
                trace_event_pending_ = true;
            }

            if (spatial_winner_hold) {
                // 后续候选不能因为前一候选被 motion 冲突而自动获胜。只有现有
                // strong ReID direct（已对全部已评估候选验证 margin）能证明它具有
                // 严格更强 BODY identity；否则继续完成评估但保持原 winner HOLD。
                if (!(strong_reid_direct && !spatial_winner_strong_reid)) {
                    if (kMatchTrace) {
                        char line[kTraceLineLen];
                        std::snprintf(line, sizeof(line),
                            "[MOTION_CONFLICT] f=%d action=keep_winner_hold"
                            " held=%d/%d evaluated=%d/%d strongBody=%d"
                            " nextCandidateAccepted=0",
                            frame_count, spatial_winner_source,
                            spatial_winner_index, c.source, c.det_index,
                            strong_reid_direct ? 1 : 0);
                        trace_push(line);
                    }
                    continue;
                }
                if (kMatchTrace) {
                    bool override_has_runner = false;
                    const float override_margin =
                        reid_margin_to_runner(c, override_has_runner);
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[MOTION_CONFLICT] f=%d action=body_identity_override"
                        " held=%d/%d winner=%d/%d reason=strong_reid_direct",
                        frame_count, spatial_winner_source,
                        spatial_winner_index, c.source, c.det_index);
                    trace_push(line);
                    std::snprintf(line, sizeof(line),
                        "[MOTION_OVERRIDE_ID] f=%d held=%d/%d heldR=%.2f heldA=%.2f"
                        " winner=%d/%d r=%.2f A=%.2f threshold=%.2f"
                        " margin=%.2f required=%.2f hasRunner=%d",
                        frame_count, spatial_winner_source, spatial_winner_index,
                        spatial_winner_reid, spatial_winner_anchor,
                        c.source, c.det_index, c.reid_sim, c.anchor_sim,
                        reid_direct_threshold(), override_margin,
                        kReidDirectMargin, override_has_runner ? 1 : 0);
                    trace_push(line);
                    trace_event_pending_ = true;
                }
                spatial_winner_hold = false;
            }

            best.matched = true;
            best.source = c.source;
            best.index = c.det_index;
            best.reid_sim = c.reid_sim;
            best.anchor_sim = c.anchor_sim;
            best.total_score = c.total;
            best.head_match = c.head_match;
            best.has_head = c.has_head;
            best.strong_reid_direct = strong_reid_direct;
            best.from_global_body_scan = c.global_explore;
            best.body_hyp_id = c.body_hyp_id;
            best.provisional_continuation = qualified_pending;
            best.provisional_weak_spatial = qualified_pending
                                          && candidate_spatial_conflict;
            best.provisional_center_dist = c.pending_center_dist;
            best.provisional_box_iou = c.pending_box_iou;
            best.provisional_size_ratio = c.pending_size_ratio;
            best.emb = c.emb;   // C2：赢家特征随结果携带，下游（入廊/emb 更新）复用免重算
            int ci = (int)(&c - candidates.data());
            best_ownership_repair_trk = coexist_repair_trk[ci];
            best_ownership_repair_src = c.source;
            best_ownership_repair_det = c.det_index;
            break;
        }

        if (spatial_winner_hold && !best.matched && kMatchTrace) {
            diagnostic_spatial_hold = true;
            diagnostic_spatial_hold_source = spatial_winner_source;
            diagnostic_spatial_hold_index = spatial_winner_index;
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[MOTION_CONFLICT] f=%d action=final_hold held=%d/%d"
                " identityNegative=0 alternateAccepted=0",
                frame_count, spatial_winner_source, spatial_winner_index);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }

    // ── OCC 身体提交门（须在人脸调度前执行）──
    // best.matched 从此处开始必须表示“该身体候选可真实更新主 KF”。低信任候选先改成
    // unmatched，当前帧才能立即进入人脸兜底；不能等到 KF 更新层才偷偷 update(empty)。
    if (best.matched && !ptz_identity_recovery
        && occlusion_state_ == OcclusionState::OCCLUDED && !best.from_face) {
        float commit_iou = 0.f;
        bool has_commit_box = false;
        const cv::Mat& commit_src = (best.source == 0) ? dets_one : dets_second;
        if ((best.source == 0 || best.source == 1)
            && best.index >= 0 && best.index < commit_src.rows) {
            cv::Mat commit_box = commit_src.row(best.index).colRange(0, 4);
            commit_iou = Utils::iou_single(trk_box, commit_box);
            has_commit_box = true;
        }

        int main_tsu_commit = trackers[main_trk_idx]->get_time_since_update();
        int64_t blind_ms_commit = get_body_blind_ms();
        bool front_visible_commit = (visibility_state_ == VisibilityState::FULL
                                  || visibility_state_ == VisibilityState::MOSTLY_FULL)
                                 && main_tsu_commit <= kFrontFollowMaxTsu
                                 && blind_ms_commit >= 0
                                 && blind_ms_commit <= kFrontFollowMaxMs;
        bool commit_trusted = has_commit_box
                           && (best.strong_reid_direct
                               || best.anchor_sim >= kAnchorDirectConfirm
                               || (best.anchor_sim >= kReacqAnchorConfident
                                   && (commit_iou >= kVetoRelaxIou
                                       || front_visible_commit)));
        if (!commit_trusted) {
            null_sink << "[COMMIT_GATE] OCC low-trust candidate rejected before commit"
                         << " src=" << best.source << " idx=" << best.index
                         << " reid=" << best.reid_sim << " anchor=" << best.anchor_sim
                         << " iou=" << commit_iou << std::endl;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[COMMIT_GATE] f=%d action=reject src=%d idx=%d r=%.2f A=%.2f"
                    " iou=%.2f front=%d blind=%lldms",
                    frame_count, best.source, best.index, best.reid_sim, best.anchor_sim,
                    commit_iou, front_visible_commit ? 1 : 0, (long long)blind_ms_commit);
                trace_push(line);
                trace_event_pending_ = true;
            }
            best = MainMatchResult{};
        }
    }

    // ──────────────────────────────────────────────────────
    // STEP 3：人脸识别
    //   触发条件：歧义 / 未匹配 / 周期验证 / alert 模式加速
    // ──────────────────────────────────────────────────────
    int face_interval = id_switch_alert_ ? 5 : face_recognition_every_n_frames;
    // 性能：危险期人脸原本逐帧（in_danger 直接触发），face 推理是大头(2×/候选)。
    //   改为每 kFaceDangerEveryN 帧；未匹配（丢失风险最高）仍立即触发，周期验证照旧。
    // 低可见度危险期（UPPER/HEAD_ONLY：仅头肩可见 → 体 ReID/骨骼权重≈0，运动/IoU 无法区分近旁冒充者）
    //   → 逐帧人脸：脸是此状态唯一可靠的同衣免疫身份证据。逐帧使 STEP3 人脸覆盖持续锚定主目标，身后
    //   走过的人无法夺锁；且人脸确认的重捕即时提交（face_direct 不受 provisional 闸约束 → 提升响应）。
    bool low_vis_danger = in_danger
        && (visibility_state_ == VisibilityState::UPPER
            || visibility_state_ == VisibilityState::HEAD_ONLY);
    bool need_face = ptz_identity_recovery || (!best.matched)
                   || low_vis_danger
                   || (in_danger && (frame_count % kFaceDangerEveryN == 0))
                   || (frame_count % face_interval == 0);

    // 主目标未匹配时必须给 face-only 留一个 slot。OCC/恢复或长盲中即使已有弱身体候选，只要它
    // 没有强 ReID/anchor/人脸身份，也预留该 slot 做全帧脸仲裁：否则旧位置的路人身体
    // 会先提交，update 外层看到 got_main=true 后根本不会进入人脸找回。
    int64_t face_arb_blind_ms = get_blind_ms();
    bool face_identity_recovery_context = ptz_identity_recovery || in_danger
                                       || face_arb_blind_ms < 0
                                       || face_arb_blind_ms >= kReacqProbationMs
                                       || id_switch_alert_
                                       || pending_from_sweep_;
    bool weak_body_identity_for_face = best.matched
                                    && !best.from_face
                                    && !best.strong_reid_direct
                                    && best.anchor_sim < kAnchorDirectConfirm;
    bool suspicious_body_for_face = weak_body_identity_for_face
                                  && face_identity_recovery_context;
    // 多人墙场景中，错误 BODY 可能凭局部连续性或较高 anchor 先被提交，使 update()
    // 外层 got_main=true，从而完全跳过第四位置的人脸恢复。不能再依赖 OCC/long-blind
    // 才扫描：状态若被错误 BODY 持续刷新，会永远保持 CLEAR。只要原始池中存在尺寸
    // 合格的脸，就让共享 standalone 分支持续做有界全帧身份观察（普通帧 1、优先帧 3）。
    int viable_recovery_faces = 0;
    for (const cv::Rect& f : recovery_faces_)
        if ((float)(f.height - f.y) >= kFaceRecogMinFacePx)
            ++viable_recovery_faces;
    bool face_template_ready = face_registered_
                            && face_recognizer.has_face_template();
    bool face_recovery_ready = face_template_ready
                            && last_main_bw_ > 0.f
                            && last_main_bh_ > 0.f
                            && viable_recovery_faces > 0;
    bool best_has_viable_associated_face = false;
    bool best_face_tracks_last_identity = false;
    bool best_body_face_geometry_ok = false;
    int best_face_plausible_owner_count = 0;
    if (best.matched && (best.source == 0 || best.source == 1)) {
        const auto& body_faces = (best.source == 0)
                               ? det_groups.dets_one_face
                               : det_groups.dets_second_face;
        const cv::Mat& body_dets = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < (int)body_faces.size()) {
            const cv::Mat& bf = body_faces[best.index];
            for (int f = 0; f < bf.rows; ++f) {
                if (bf.at<float>(f, 3) - bf.at<float>(f, 1)
                    >= kFaceRecogMinFacePx) {
                    best_has_viable_associated_face = true;
                    if (best.index < body_dets.rows) {
                        cv::Rect assoc_face(
                            (int)std::lround(bf.at<float>(f, 0)),
                            (int)std::lround(bf.at<float>(f, 1)),
                            (int)std::lround(bf.at<float>(f, 2)),
                            (int)std::lround(bf.at<float>(f, 3)));
                        best_body_face_geometry_ok =
                            face_body_geometry_consistent(
                                assoc_face,
                                body_dets.row(best.index).colRange(0, 4));
                        auto count_face_owners = [&](const cv::Mat& dets) {
                            for (int i = 0; i < dets.rows; ++i) {
                                if (face_body_geometry_consistent(
                                        assoc_face, dets.row(i).colRange(0, 4)))
                                    ++best_face_plausible_owner_count;
                            }
                        };
                        count_face_owners(dets_one);
                        count_face_owners(dets_second);
                    }
                    if (last_confirmed_face_ms_ >= 0
                        && now_match_ms - last_confirmed_face_ms_
                           <= kFaceRecoveryRefMaxAgeMs
                        && last_confirmed_face_box_.width > last_confirmed_face_box_.x
                        && last_confirmed_face_box_.height > last_confirmed_face_box_.y) {
                        float cx = (bf.at<float>(f, 0) + bf.at<float>(f, 2)) * 0.5f;
                        float cy = (bf.at<float>(f, 1) + bf.at<float>(f, 3)) * 0.5f;
                        float rcx = ((float)last_confirmed_face_box_.x
                                   + (float)last_confirmed_face_box_.width) * 0.5f;
                        float rcy = ((float)last_confirmed_face_box_.y
                                   + (float)last_confirmed_face_box_.height) * 0.5f;
                        float fw = std::max(1.f, bf.at<float>(f, 2)
                                                - bf.at<float>(f, 0));
                        float fh = std::max(1.f, bf.at<float>(f, 3)
                                                - bf.at<float>(f, 1));
                        float rw = std::max(1.f, (float)(last_confirmed_face_box_.width
                                                      - last_confirmed_face_box_.x));
                        float rh = std::max(1.f, (float)(last_confirmed_face_box_.height
                                                      - last_confirmed_face_box_.y));
                        float fdiag = std::hypot(fw, fh);
                        float rdiag = std::hypot(rw, rh);
                        float ratio = fdiag / std::max(1.f, rdiag);
                        best_face_tracks_last_identity =
                            ratio >= kFaceTrackSizeRatioMin
                            && ratio <= kFaceTrackSizeRatioMax
                            && std::hypot(cx - rcx, cy - rcy)
                               <= kFaceTrackGateDiag * std::max(fdiag, rdiag);
                    }
                    break;
                }
            }
        }
    }
    bool recent_face_identity = last_face_identity_ms_ >= 0
                             && now_match_ms - last_face_identity_ms_
                                <= kFaceRecoveryRefMaxAgeMs;
    bool recent_body_face_identity = last_body_face_identity_ms_ >= 0
                                  && now_match_ms - last_body_face_identity_ms_
                                     <= kFaceRecoveryRefMaxAgeMs;
    // 单人、CLEAR、当前 BODY 自带刚确认过的唯一人脸时沿用周期验证，避免健康跟踪
    // 每帧都多跑 11.5ms。这里必须是“该 BODY 本身近期经 FaceReco 确认”，不能只
    // 复用 face-only 身份时间；否则错误 BODY 包住真脸后会关闭全局扫描。除此之外
    // （多人脸、脸未绑定当前 BODY、身份已过期、几何异常、危险/长盲）都持续扫描
    // 原始脸池，保证任意位置的主目标脸不会被 BODY 控制流跳过。
    bool healthy_single_face_body = best.matched
                                 && !face_identity_recovery_context
                                 && viable_recovery_faces == 1
                                 && best_has_viable_associated_face
                                 && best_face_tracks_last_identity
                                 && best_body_face_geometry_ok
                                 && recent_face_identity
                                 && recent_body_face_identity
                                 && face_recovery_fail_streak_ == 0;
    bool global_face_watch = face_recovery_ready && !healthy_single_face_body;
    if (kMatchTrace && !recovery_faces_.empty()) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_WATCH] f=%d ready=%d active=%d healthy1=%d flag=%d template=%d"
            " history=%d viable=%d best=%d/%d assoc=%d track=%d geom=%d"
            " bodyId=%d budget=%d context=%d",
            frame_count, face_recovery_ready ? 1 : 0,
            global_face_watch ? 1 : 0, healthy_single_face_body ? 1 : 0,
            face_registered_ ? 1 : 0,
            face_recognizer.has_face_template() ? 1 : 0,
            (last_main_bw_ > 0.f && last_main_bh_ > 0.f) ? 1 : 0,
            viable_recovery_faces, best.source, best.index,
            best_has_viable_associated_face ? 1 : 0,
            best_face_tracks_last_identity ? 1 : 0,
            best_body_face_geometry_ok ? 1 : 0,
            recent_body_face_identity ? 1 : 0, face_model_budget_,
            face_identity_recovery_context ? 1 : 0);
        trace_push(line);
    }
    bool body_needs_face_arbiter = ptz_identity_recovery || (best.matched
                                && !best.from_face
                                && global_face_watch);
    preserve_face_search_state_ = body_needs_face_arbiter;
    bool strong_body_identity = best.strong_reid_direct
                             || best.anchor_sim >= kAnchorDirectConfirm
                             || (recent_body_face_identity
                                 && best_face_tracks_last_identity);
    bool body_face_owner_clear = best_has_viable_associated_face
                              && best_face_tracks_last_identity
                              && best_body_face_geometry_ok
                              && (best_face_plausible_owner_count == 1
                                  || strong_body_identity);
    prefer_body_geometry_output_ = body_needs_face_arbiter
                                && (strong_body_identity
                                    || body_face_owner_clear);
    if (kMatchTrace && body_needs_face_arbiter) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[BOX_ARB] f=%d action=%s body=%d/%d strong=%d assoc=%d"
            " track=%d geom=%d owners=%d",
            frame_count,
            prefer_body_geometry_output_ ? "prefer_body" : "allow_part_override",
            best.source, best.index, strong_body_identity ? 1 : 0,
            best_has_viable_associated_face ? 1 : 0,
            best_face_tracks_last_identity ? 1 : 0,
            best_body_face_geometry_ok ? 1 : 0,
            best_face_plausible_owner_count);
        trace_push(line);
    }
    bool reserve_standalone_face = face_recovery_ready
                                && (!best.matched || body_needs_face_arbiter);
    int standalone_reserve = reserve_standalone_face ? 1 : 0;
    if (face_priority_frame_ && reserve_standalone_face)
        standalone_reserve = face_model_budget_;
    int body_face_budget = std::max(0, face_model_budget_ - standalone_reserve);

    if (need_face && near_box.rows > 0 && body_face_budget > 0
        && face_recognizer.has_face_template()) {
        cv::Mat face_matched_one = cv::Mat::zeros(0, 2, CV_32S);
        cv::Mat face_matched_second = cv::Mat::zeros(0, 2, CV_32S);

        // 注意：绝不能把当前 body 匹配作为"种子"传入（旧实现如此）——
        // face_recognition_match 在识别失败（got_match=false）时会把种子原样
        // 返回，下方仅凭 rows>0 判定 → "身体匹配"被伪装成"人脸确认"：
        //   from_face=true 骗过 C-identity/teleport 身份门、人脸锁无脸自续、
        //   emb 以 0.7 大步刷新、画廊以人脸级质量入样；danger 期逐帧触发时
        //   等于持续给所有人脸防线喂假信号（跟错人时直接确认错误目标）。
        // 同时周期性人脸"拉回主目标"被种子顶替，形同虚设。
        // 传空种子后：返回非空 = 真正完成了一次人脸识别匹配。
        // 性能：CLEAR 期人脸最多对前 kFaceMaxCand 个最近候选跑（near_box 行按距主
        //   目标升序，见 collect_nearby_dets 排序）；危险期保持全量候选（判别力优先，
        //   频次已由 kFaceDangerEveryN 降低）。远处目标的歧义由下方全画面人脸扫描兜底。
        int face_box_n = in_danger ? std::min(near_box.rows, kFaceDangerMaxCand)
                                   : std::min(near_box.rows, kFaceMaxCand);
        face_box_n = std::min(face_box_n, body_face_budget);
        Face_Match face_result = face_recognition_match(
            near_box, face_box_n, det_groups,
            face_matched_one, face_matched_second, 0, img);

        int face_src = -1, face_idx = -1;
        if (face_result.matched_one.rows > 0) {
            face_src = 0; face_idx = face_result.matched_one.at<int>(0, 1);
        } else if (face_result.matched_second.rows > 0) {
            face_src = 1; face_idx = face_result.matched_second.at<int>(0, 1);
        }

        // ── 疑惑期全画面人脸扫描（near_box 未命中时的第二级，见 .h 注释）──
        //   错锁时真主目标常在 near_box 半径（2×对角线）之外 → 第一级永远
        //   看不到他的脸。疑惑期把全画面"可识别尺寸"的带脸检测轮询送识别。
        //   B1(b)：扫描命中不再即时硬覆盖（远距小脸误识 = 瞬移 + 模板污染的单点
        //   故障）——改走下方 C-identity 暂定闸（kFaceSweepConfirmFrames 帧同假设确认）。
        //   暂定假设活跃期间（pending_from_sweep_）逐帧重扫以复验同一假设。
        bool face_hit_from_sweep = false;
        // 陈旧假设保险：连续数帧未能复验（目标消失/被遮）→ 停止逐帧重扫，
        // 回落 kFaceSweepPeriod 周期（防无限期 +kFaceSweepMaxFaces×2 NPU/帧）。
        if (pending_from_sweep_
            && frame_count - pending_sweep_frame_ > kFaceSweepConfirmFrames + 2)
            pending_from_sweep_ = false;
        if (face_idx < 0 && !reserve_standalone_face
            && ((frame_count % kFaceSweepPeriod == 0) || pending_from_sweep_)
            && best.anchor_sim < kFaceSweepDoubtAnchor
            && !best.from_face
            && !face_lock_active) {
            struct SweepCand { int src; int idx; };
            std::vector<SweepCand> pool;
            auto in_near = [&](int src, int idx) -> bool {
                // near_box 在人体全图搜索期包含所有人体；第一级实际上只处理了
                // face_box_n 个候选。只排除真正送过第一级的人脸，否则会把远端脸
                // 错当成“已覆盖”，令 FACE_SWEEP 失效。
                for (int k = 0; k < face_box_n; ++k)
                    if ((int)near_box.at<float>(k, 4) == src
                        && (int)near_box.at<float>(k, 5) == idx) return true;
                return false;
            };
            auto gather = [&](const std::vector<cv::Mat>& faces_vec, int src) {
                for (int i = 0; i < (int)faces_vec.size(); ++i) {
                    const cv::Mat& fb = faces_vec[i];
                    if (fb.empty() || fb.rows == 0) continue;
                    bool viable = false;   // 任一关联脸达到可识别尺寸（xyxy：col3-col1=脸高）
                    for (int f = 0; f < fb.rows && !viable; ++f)
                        if (fb.at<float>(f, 3) - fb.at<float>(f, 1) >= kFaceSweepMinFacePx)
                            viable = true;
                    if (!viable) continue;
                    if (in_near(src, i)) continue;   // 第一级已覆盖，避免重复推理
                    pool.push_back({ src, i });
                }
            };
            gather(det_groups.dets_one_face, 0);
            gather(det_groups.dets_second_face, 1);

            if (!pool.empty()) {
                int take = std::min((int)pool.size(), kFaceSweepMaxFaces);
                cv::Mat sweep_box(take, 6, CV_32F);
                for (int t = 0; t < take; ++t) {
                    const SweepCand& sc = pool[(face_sweep_rotor_ + t) % (int)pool.size()];
                    const cv::Mat& sdets = (sc.src == 0) ? dets_one : dets_second;
                    sdets.row(sc.idx).colRange(0, 4).copyTo(sweep_box.row(t).colRange(0, 4));
                    sweep_box.at<float>(t, 4) = (float)sc.src;
                    sweep_box.at<float>(t, 5) = (float)sc.idx;
                }
                face_sweep_rotor_ = (face_sweep_rotor_ + take) % (int)pool.size();

                Verification_Result sv = face_recognition_verification(
                    sweep_box, take, det_groups, 0, img,
                    cv::Mat::zeros(0, 2, CV_32S), cv::Mat::zeros(0, 2, CV_32S));
                if (sv.got_match) {
                    if (sv.matches_main_one.rows > 0) {
                        face_src = 0; face_idx = sv.matches_main_one.at<int>(0, 1);
                    } else if (sv.matches_main_second.rows > 0) {
                        face_src = 1; face_idx = sv.matches_main_second.at<int>(0, 1);
                    }
                    if (face_idx >= 0) {
                        face_hit_from_sweep = true;
                        null_sink << "[FACE_SWEEP] main recovered by full-frame sweep"
                                  << " src=" << face_src << " idx=" << face_idx
                                  << " frame=" << frame_count << std::endl;
                    }
                }
            }
        }

        if (face_idx >= 0) {
            // ── 人脸覆盖：锚定到人脸所在检测 ──
            //   B1(b)：来自全画面扫描的命中只改写匹配假设（best.*），信任放大类
            //   副作用（人脸锁 / emb 刷新 / 入廊 / faceVerified / 解除 alert）一律
            //   延后——待暂定闸确认、锁物理移过去后，近场人脸确认自然补上这些。
            // 人脸覆盖若恰好落在已做 ReID 的候选上，复用该 embedding；不为人脸确认
            // 额外补跑 ReID，避免复杂帧突破 10 FPS 模型预算。
            cv::Mat reused_face_emb;
            const Candidate* face_cand = nullptr;
            for (size_t face_ci = 0; face_ci < candidates.size(); ++face_ci) {
                const auto& c = candidates[face_ci];
                if (c.source == face_src && c.det_index == face_idx) {
                    face_cand = &c;
                    if (!c.emb.empty()) reused_face_emb = c.emb;
                    if (coexist_repair_trk[face_ci] >= 0) {
                        best_ownership_repair_trk = coexist_repair_trk[face_ci];
                        best_ownership_repair_src = c.source;
                        best_ownership_repair_det = c.det_index;
                    }
                    break;
                }
            }

            best.matched   = true;
            best.source    = face_src;
            best.index     = face_idx;
            best.from_face = true;
            best.face_from_sweep = face_hit_from_sweep;
            best.from_global_body_scan = false;
            best.emb       = reused_face_emb;
            best.strong_reid_direct = false;
            if (diagnostic_spatial_hold && kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[MOTION_CONFLICT] f=%d action=face_identity_override"
                    " held=%d/%d winner=%d/%d reason=face_reco_positive"
                    " bodyStrongReidRequired=0",
                    frame_count, diagnostic_spatial_hold_source,
                    diagnostic_spatial_hold_index, face_src, face_idx);
                trace_push(line);
                trace_event_pending_ = true;
            }
            if (face_cand) {
                best.reid_sim = face_cand->reid_sim;
                best.anchor_sim = face_cand->anchor_sim;
                best.total_score = face_cand->total;
                best.head_match = face_cand->head_match;
                best.has_head = face_cand->has_head;
                best.body_hyp_id = face_cand->body_hyp_id;
            } else {
                best.reid_sim = best.anchor_sim = best.total_score = 0.f;
                best.head_match = 0.f;
                best.has_head = false;
            }

            const cv::Mat& fsrc_dets = (face_src == 0) ? dets_one : dets_second;

            if (!face_hit_from_sweep) {
                // ── 近场人脸确认（第一级）：即时建立/刷新人脸锁 + 信任副作用 ──
                face_locked_          = true;
                last_face_lock_frame_ = frame_count;
                last_face_lock_ms_    = now_match_ms;   // B6：TTL 改墙钟
                last_face_identity_ms_ = now_match_ms;
                last_body_face_identity_ms_ = now_match_ms;

                if (face_idx < fsrc_dets.rows) {
                    cv::Mat conf_box = fsrc_dets.row(face_idx).colRange(0, 4).clone();
                    face_lock_box_ = conf_box.clone();

                    // 保存真实脸框而不是人体框，供目标随后躲到障碍物后时作为首选找脸参考。
                    // matchPersonFaces 已保证每个人体最多一张上部居中的脸。
                    const auto& body_faces = (face_src == 0)
                                           ? det_groups.dets_one_face
                                           : det_groups.dets_second_face;
                    if (face_idx >= 0 && face_idx < (int)body_faces.size()
                        && !body_faces[face_idx].empty()
                        && body_faces[face_idx].rows > 0) {
                        const cv::Mat& fb = body_faces[face_idx];
                        last_confirmed_face_box_ = cv::Rect(
                            (int)std::lround(fb.at<float>(0, 0)),
                            (int)std::lround(fb.at<float>(0, 1)),
                            (int)std::lround(fb.at<float>(0, 2)),
                            (int)std::lround(fb.at<float>(0, 3)));
                        last_confirmed_face_ms_ = now_match_ms;
                        last_confirmed_face_frame_ = frame_count;
                        face_recovery_rotor_ = 0;
                        face_recovery_fail_streak_ = 0;
                        face_recovery_hypotheses_.clear();
                        next_face_recovery_hyp_id_ = 1;
                        face_global_pending_ = false;
                        face_global_pending_streak_ = 0;
                        face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
                        face_global_pending_ms_ = -1;
                    }

                    // ── 人脸确认 → 用本帧已有的 ReID 特征刷新主目标 emb ──
                    //   未入本帧 ReID 预算的人脸只建立人脸锁；embedding 留给后续常规帧，
                    //   不在此补跑模型，保证复杂帧耗时有界。
                    if (proximity.close_det_count <= 2) {
                        cv::Mat conf_emb = best.emb;
                        if (!conf_emb.empty()) {
                            trackers[main_trk_idx]->update_emb(conf_emb, kFaceConfirmEmbAlpha);
                            best.emb = conf_emb;   // C2：下游（入廊/emb 更新）复用，免重算
                            null_sink << "[FACE_LOCK] emb refreshed by face confirmation" << std::endl;

                            // ── 人脸确认 = 最强身份证据 → 入锚点画廊（自愈注册污染）──
                            //   质量随可见度提升。B1(c)：封顶 0.9 —— q≈1.0 的样本"永生"
                            //   （淘汰需更高 q），一次误确认即永久污染画廊；封顶后仍可被
                            //   高置信全身样本（q 上限 1.0）淘汰，保留自愈通道。
                            float q = std::min(0.9f, 0.6f + 0.4f * visible_ratio_ema_);
                            // Q1：人脸确认 = 身份确定 → 安全的多可见度采样点（vis 标注当前可见比例，
                            //   低可见度时填充 LOW/MID 带参考，且不会污染画廊）。
                            trackers[main_trk_idx]->add_anchor_sample(conf_emb, q, visible_ratio_ema_);
                            null_sink << "[ANCHOR] gallery += face (q=" << q << " vis=" << visible_ratio_ema_
                                      << " size=" << trackers[main_trk_idx]->get_anchor_gallery_size()
                                      << ")" << std::endl;
                        }
                    }
                }

                // ── 标记该 tracker 已通过人脸验证 ──
                trackers[main_trk_idx]->faceVerified = true;

                // ── 人脸确认 → 解除 alert / 嫌疑 ──
                if (id_switch_alert_) {
                    id_switch_alert_ = false;
                    suspect_streak_  = 0;
                    alert_start_ms_  = 0;
                    null_sink << "[ALERT_CLEARED] face confirmed main target" << std::endl;
                }
                null_sink << "[FACE_LOCK] locked src=" << face_src
                          << " idx=" << face_idx << " frame=" << frame_count << std::endl;
            }
        }
    }

    // 全状态原始人脸观察：只消费上面预留的共享 slot；人脸优先帧最多使用 3 个，
    // 普通帧仍为 1 个。该分支存在的前提就是“当前 BODY 尚未被本帧人脸身份解释”。
    // 因此一旦全局原始脸命中主目标，身份必须覆盖 BODY：先撤销身体提交，让 update()
    // 外层的 try_face_only_continuity 从帧内缓存复用结果，并统一执行全局阈值/两帧
    // 确认、脸框重构、KF 重定位和 FACE 输出。不能再按“脸落在当前 BODY 内”执行
    // confirm_same_body——多人重叠时宽大错误框恰恰会吞住真脸，导致识别成功却拉不回。
    if (body_needs_face_arbiter && !best.from_face
        && !recovery_faces_.empty()
        && (face_model_budget_ > 0 || !face_inference_cache_.empty())) {
        float rcx = (trk_box.at<float>(0, 0) + trk_box.at<float>(0, 2)) * 0.5f;
        float rcy = (trk_box.at<float>(0, 1) + trk_box.at<float>(0, 3)) * 0.5f;
        if (last_confirmed_face_ms_ >= 0
            && now_match_ms - last_confirmed_face_ms_ <= kFaceRecoveryRefMaxAgeMs) {
            rcx = ((float)last_confirmed_face_box_.x
                 + (float)last_confirmed_face_box_.width) * 0.5f;
            rcy = ((float)last_confirmed_face_box_.y
                 + (float)last_confirmed_face_box_.height) * 0.5f;
        }
        float tw = std::max(1.f, trk_box.at<float>(0, 2) - trk_box.at<float>(0, 0));
        float th = std::max(1.f, trk_box.at<float>(0, 3) - trk_box.at<float>(0, 1));
        float face_gate = kFaceOnlyGateDiag * std::hypot(tw, th);
        cv::Point2f face_ref(rcx, rcy);
        DetectionGroups empty_groups;
        // 正常情况下上方已为本分支预留至少一个 slot。若其它人体人脸路径刚好
        // 消耗完预算，仍允许遍历本帧缓存；不会新增 FaceKps/FaceReco 调用。
        int arb_budget = face_model_budget_ > 0
                       ? std::min(kFaceOnlyMaxCand, face_model_budget_)
                       : (int)recovery_faces_.size();
        Verification_Result arb = face_recognition_verification(
            cv::Mat::zeros(0, 6, CV_32F), 0, empty_groups, 0, img,
            cv::Mat::zeros(0, 2, CV_32S), cv::Mat::zeros(0, 2, CV_32S),
            ptz_identity_recovery ? nullptr : &face_ref,
            ptz_identity_recovery ? 0.f : face_gate, arb_budget,
            nullptr, ptz_identity_recovery);
        if (arb.got_match && arb.matched_standalone) {
            bool current_body_geom_ok = false;
            int plausible_owner_count = 0;
            int owner_src = -1, owner_idx = -1;
            float owner_cost = FLT_MAX;
            cv::Mat reconstructed_face_body;
            if (arb.standalone_face_idx >= 0
                && arb.standalone_face_idx < (int)recovery_faces_.size()) {
                const cv::Rect& af = recovery_faces_[arb.standalone_face_idx];
                reconstructed_face_body = reconstruct_body_from_face(af);

                auto inspect_owners = [&](const cv::Mat& body_dets, int src) {
                    for (int i = 0; i < body_dets.rows; ++i) {
                        float cost = FLT_MAX;
                        if (!face_body_geometry_consistent(
                                af, body_dets.row(i).colRange(0, 4), &cost))
                            continue;
                        ++plausible_owner_count;
                        if (cost < owner_cost) {
                            owner_cost = cost;
                            owner_src = src;
                            owner_idx = i;
                        }
                    }
                };
                inspect_owners(dets_one, 0);
                inspect_owners(dets_second, 1);

                const cv::Mat& current_body_dets =
                    (best.source == 0) ? dets_one : dets_second;
                if ((best.source == 0 || best.source == 1)
                    && best.index >= 0 && best.index < current_body_dets.rows) {
                    current_body_geom_ok = face_body_geometry_consistent(
                        af, current_body_dets.row(best.index).colRange(0, 4));
                }
            }

            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_OWNER] f=%d face=%d body=%d/%d geom=%d"
                    " owners=%d bestOwner=%d/%d cost=%.2f recon=%d",
                    frame_count, arb.standalone_face_idx,
                    best.source, best.index,
                    current_body_geom_ok ? 1 : 0, plausible_owner_count,
                    owner_src, owner_idx,
                    std::isfinite(owner_cost) ? owner_cost : -1.f,
                    reconstructed_face_body.empty() ? 0 : 1);
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[FACE_ARBITER] f=%d action=route_face_only body=%d/%d"
                    " face=%d sim=%.2f global=%d weak=%d budget=%d",
                    frame_count, best.source, best.index,
                    arb.standalone_face_idx, arb.standalone_sim,
                    arb.standalone_global ? 1 : 0,
                    weak_body_identity_for_face ? 1 : 0, arb_budget);
                trace_push(line);
                trace_event_pending_ = true;
            }

            // 无论该脸是否落在当前人体框内，都先按脸定位。人体框的“包含关系”
            // 不是身份依据；目标从人墙后露出时，错误人体框可能恰好包住这张脸。
            // 外层会复用同帧缓存，不增加模型推理，并在身份确认后重定位 KF。
            clear_body_provisional("face_identity_relocated", "face_arbiter");
            best = MainMatchResult{};
        } else {
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[FACE_ARBITER] f=%d action=no_face_match body=%d/%d budget=%d",
                    frame_count, best.source, best.index, arb_budget);
                trace_push(line);
            }
            // 弱 BODY 在全帧人脸尚未覆盖完之前宁可 HOLD，避免三人墙中的路人先抢走
            // ID；强 ReID/anchor BODY 则仍保留原提交能力。下次仲裁会轮到另一张物理脸。
            if (suspicious_body_for_face && !best.provisional_continuation) {
                best = MainMatchResult{};
                // update() 外层的 try_face_only_continuity 会复用本帧缓存并统一累计失败。
            } else {
                ++face_recovery_fail_streak_;
            }
        }
    }
    // ──────────────────────────────────────────────────────
    // Face 仍是最终 BODY measurement 之前的强身份安全门。owner 使用
    // recovery BODY/Face 映射，不假设 detector source/index 就是 owner 下标。
    // Face UNKNOWN 不阻断 exact + stable continuation；Face NEGATIVE 仍硬拒绝。
    if (best.matched && best.from_face && pending_body_hyp_id_ >= 0) {
        clear_body_provisional("face_identity_relocated", "face_identity");
    } else if (best.matched && best.provisional_continuation
               && (best.source == 0 || best.source == 1)) {
        const cv::Mat& body_src = best.source == 0 ? dets_one : dets_second;
        if (best.index >= 0 && best.index < body_src.rows) {
            std::string identity_reason;
            int owner_person = -1;
            const IdentityEvidence body_evidence = body_identity_evidence_for_box(
                body_src.row(best.index).colRange(0, 4),
                &identity_reason, &owner_person);
            if (body_evidence == IdentityEvidence::NEGATIVE) {
                const char* reset_reason =
                    identity_reason == "associated_face_mismatch"
                        ? "associated_face_mismatch"
                        : identity_reason == "confirmed_other_identity"
                            ? "confirmed_other_identity"
                            : "strict_body_identity_negative";
                clear_body_provisional(
                    reset_reason, "face_body_identity",
                    best.source, best.index, best.reid_sim, best.anchor_sim,
                    best.provisional_center_dist, best.provisional_box_iou,
                    best.provisional_size_ratio);
                best = MainMatchResult{};
            }
        }
    }
    if (best.matched && best.provisional_continuation) {
        trace_body_provisional(
            "continue", best.body_hyp_id, pending_body_hyp_id_,
            best.source, best.index, best.reid_sim, best.anchor_sim,
            pending_body_reid_, pending_body_anchor_,
            reacq_defer_count_, kReacqMaxDefer,
            best.provisional_center_dist, best.provisional_box_iou,
            best.provisional_size_ratio, "qualified", "face_safety_passed",
            best.provisional_weak_spatial ? "weak_spatial_only" : "none");
    }

    // C-identity：长盲 coast 后重捕的身份复核
    //   场景：目标被家具长时间遮挡（纯预测盲 coast），重新出现时
    //   极可能在画面"全新位置"，此时若轻率接受一个弱匹配，会把
    //   云台引向错误目标并自我强化。故：盲时长 blind_ms ≥ kReacqProbationMs 时，
    //   要求强身份证据（人脸 / 高 anchor / 高 reid）才接受重捕；
    //   证据不足则延迟接受（best.matched=false → 路由到 unmatched →
    //   继续纯预测 coast），延迟上限 kReacqMaxDefer 帧防止永久丢失。
    //   说明：仅用"检测派生信号"判定（不依赖 visibility，后者在盲 coast
    //   期间锁定在冻结 KF 框上不可信）；高 anchor/reid 隐含足够可见度。
    //   头/脸连续性与真实身体命中都会刷新 last_real_obs_ms_，故 blind_ms 只在"真盲"时增长，
    //   正是此门控想要的信号（[N1] 墙钟版，取代原 predictor.get_miss_count / main_blind_frames_）。
    if (best.matched) {
        // [N1] 盲跟改墙钟：blind_ms = now − last_real_obs_ms_（<0 视为从未观测=长盲）。
        //   帧计数在 27~374ms 帧间隔波动下语义漂移 14×，故弃用。
        int64_t blind_ms  = get_body_blind_ms();
        float   blind_sec = (blind_ms < 0) ? 0.f : (float)blind_ms / 1000.f;
        bool    long_blind = (blind_ms < 0) || (blind_ms >= kReacqProbationMs);

        // ── 空间触发臂（teleport gate）：短盲期内的中心跳变复核 ──
        //   长盲已无条件要求身份，无需再算距离；人脸匹配天然豁免（在 id_confirmed 中）。
        //   参考点用引导中心（每帧刷新、coast 期跟随外推），无效时退回 KF 框中心。
        //   预算 = 主框对角线 × (基础 + 每丢失帧追加)，宽松到正常运动/云台回扫绝不触发，
        //   只拦"瞬移"——即检测闪烁期路人凭冻结 KF 框 iou 夺锁的经典错配。
        // best 检测框中心 / 与 KF 预测框 IoU / 主框对角线（连续性 + 各触发臂共用，恒计算）
        float bcx = 0.f, bcy = 0.f, best_iou = 0.f;
        float tw = trk_box.at<float>(0, 2) - trk_box.at<float>(0, 0);
        float th = trk_box.at<float>(0, 3) - trk_box.at<float>(0, 1);
        float tdiag = std::sqrt(tw * tw + th * th);
        bool has_bb = false;
        cv::Mat best_body_box;
        {
            const cv::Mat& bsrc = (best.source == 0) ? dets_one : dets_second;
            if (best.index >= 0 && best.index < bsrc.rows) {
                cv::Mat bb = bsrc.row(best.index).colRange(0, 4);
                best_body_box = bb.clone();
                bcx = (bb.at<float>(0, 0) + bb.at<float>(0, 2)) * 0.5f;
                bcy = (bb.at<float>(0, 1) + bb.at<float>(0, 3)) * 0.5f;
                best_iou = Utils::iou_single(trk_box, bb);
                has_bb = true;
            }
        }

        // 空间触发臂（teleport，仅短盲有意义）：中心相对引导/KF 参考点跳变超预算
        bool teleport = false;
        float jump_d = 0.f, jump_allow = 0.f;
        if (!long_blind && has_bb) {
            float rcx, rcy;
            if (lead_cx_ >= 0.f && lead_cy_ >= 0.f) { rcx = lead_cx_; rcy = lead_cy_; }
            else {
                rcx = (trk_box.at<float>(0, 0) + trk_box.at<float>(0, 2)) * 0.5f;
                rcy = (trk_box.at<float>(0, 1) + trk_box.at<float>(0, 3)) * 0.5f;
            }
            jump_d = std::sqrt((bcx - rcx) * (bcx - rcx) + (bcy - rcy) * (bcy - rcy));
            // [N1] teleport 预算帧率无关：对角线 ×(基础 + 每秒追加×盲秒[上限]) + 自运动松弛(px)。
            //   [N10] ego_shift_mag_ = 本帧预测表观位移，作为额外松弛（云台自身运动不该触发瞬移复核）。
            float blind_sec_cap = std::min(blind_sec, (float)kTeleportElapsedCapMs / 1000.f);
            jump_allow = tdiag * (kTeleportBaseDiagPerSec * frame_dt_sec_
                                + kTeleportRateDiagPerSec * blind_sec_cap)
                       + ego_shift_mag_;
            teleport = (jump_d > jump_allow);
        }

        // 头部不连续臂（原独立 M2/VIS_GATE 折叠进来）：头部新鲜但身体框与 KF 预测几乎不
        //   重叠 → 主目标不可能瞬移到那里，必是他人（同衣骗过 anchor/reid/teleport）。
        bool head_discont = head_track_fresh && head_tsu <= kHeadVetoMaxAge
                          && has_bb && best_iou < kUncertainVisIou;

        // B7-③ 软盲臂：头部维持（无身份证据）已持续较久 → 本次身体重捕视同 long_blind
        //   （堵死"头部维持 on_matched 清 miss_count → long_blind 永不武装"的漏洞）。
        bool body_head_continuous = best.has_head
                                 && best.head_match >= kHeadBodyResumeMin;
        bool head_only_long = head_only_since_ms_ >= 0
                           && (now_ms() - head_only_since_ms_) >= kHeadOnlySuspectMs
                           && !body_head_continuous;

        // ── 统一暂定提交闸（高惊奇度重捕：分离瞬间 / 远处重现）──
        //   四类高惊奇：long_blind（盲久）/ teleport（中心跳变）/ head_discont（头在但体框
        //   错位）/ head_only_long（头部维持后的身体重捕）。B1(b)：扫描人脸也走此闸。
        //   不立即落定，作为"暂定假设"持有；仅当【同一假设】连续 N 帧稳定、或
        //   近场人脸确认 才提交。暂定期 best.matched=false → 主目标 coast（KF 冻结不被
        //   污染、云台保持）。
        bool sweep_face_hyp = best.from_face && best.face_from_sweep;
        if (long_blind || teleport || head_discont || head_only_long || sweep_face_hyp) {
            // 扫描人脸假设：人脸本身是强证据，同假设 kFaceSweepConfirmFrames(2) 帧即提交；
            // 其余高惊奇 BODY 维持 kReacqMaxDefer。
            int need = sweep_face_hyp ? kFaceSweepConfirmFrames : kReacqMaxDefer;
            const bool body_hyp = long_blind && !best.from_face
                               && has_bb && best.body_hyp_id >= 0;
            if (body_hyp && pending_body_hyp_id_ >= 0
                && best.body_hyp_id != pending_body_hyp_id_) {
                clear_body_provisional(
                    "hyp_changed", "c_identity",
                    best.source, best.index, best.reid_sim, best.anchor_sim,
                    -1.f, -1.f, -1.f, best.body_hyp_id);
            }

            bool same_hyp = false;
            if (body_hyp) {
                same_hyp = best.provisional_continuation
                        && pending_body_hyp_id_ == best.body_hyp_id;
                if (pending_body_hyp_id_ < 0) {
                    pending_body_hyp_id_ = best.body_hyp_id;
                    pending_body_x1_ = best_body_box.at<float>(0, 0);
                    pending_body_y1_ = best_body_box.at<float>(0, 1);
                    pending_body_x2_ = best_body_box.at<float>(0, 2);
                    pending_body_y2_ = best_body_box.at<float>(0, 3);
                    pending_body_reid_ = best.reid_sim;
                    pending_body_anchor_ = best.anchor_sim;
                    pending_body_start_frame_ = frame_count;
                    pending_body_start_ms_ = now_match_ms;
                    reacq_defer_count_ = 0;
                    trace_body_provisional(
                        "start", pending_body_hyp_id_, -1,
                        best.source, best.index, best.reid_sim, best.anchor_sim,
                        pending_body_reid_, pending_body_anchor_, 0, need,
                        0.f, 1.f, 1.f, "high_surprise_body", "c_identity");
                }
            } else {
                // 非 BODY provisional 保留原有中心连续性语义。
                same_hyp = pending_active_ && has_bb
                    && std::sqrt((bcx - pending_cx_) * (bcx - pending_cx_)
                               + (bcy - pending_cy_) * (bcy - pending_cy_))
                       <= kProvisionalPosTolFactor * tdiag;
                if (!same_hyp) reacq_defer_count_ = 0;
            }
            if (has_bb) {
                pending_cx_ = bcx; pending_cy_ = bcy; pending_active_ = true;
                pending_src_ = best.source; pending_idx_ = best.index;   // [N4] 暂定假设对应检测
            }
            pending_from_sweep_ = sweep_face_hyp;   // 下帧扫描逐帧重试，复验同一假设
            if (sweep_face_hyp) pending_sweep_frame_ = frame_count;

            // 身份直证：近场人脸确认即时落定；新 ReID/anchor 达强直证阈值时也即时落定。
            // 扫描人脸仍需同假设连续确认，防远距小脸单点误识。
            bool face_direct = best.from_face && !best.face_from_sweep;
            // 全图探索会把一次比较扩展到更多人，异人高分尾部被撞中的概率随人数增加。
            // 普通强 ReID 命中保留为候选，但不能单帧绕过同假设复验；极强 anchor 仍可即时提交。
            bool strong_reid_direct = best.strong_reid_direct
                                   && (!best.from_global_body_scan
                                       || best.anchor_sim >= kAnchorDirectConfirm);
            bool strong_anchor_direct = best.anchor_sim >= kAnchorDirectConfirm;
            // 新 ReID 可作为强身份直证：只要前面的共存/头部等硬否决已通过，就不再因为
            // long_blind/head_discont/head_only 机械等待 1~2 帧，避免主目标穿人后可见却卡顿。
            bool id_confirmed = face_direct
                             || best.provisional_continuation
                             || strong_reid_direct
                             || strong_anchor_direct
                             || (teleport && !long_blind && !head_discont && !head_only_long
                                 && best.anchor_sim >= kReacqAnchorConfident);
            if (!id_confirmed && reacq_defer_count_ < need) {
                reacq_defer_count_++;
                null_sink << "[PROVISIONAL] hold #" << reacq_defer_count_ << "/" << need
                          << " long_blind=" << long_blind << " teleport=" << teleport
                          << " head_discont=" << head_discont
                          << " head_only=" << head_only_long
                          << " sweep_face=" << sweep_face_hyp
                          << " jump=" << jump_d << " iou=" << best_iou
                          << " anchor=" << best.anchor_sim << " reid=" << best.reid_sim
                          << " -> coast (KF frozen)" << std::endl;
                best.matched = false;   // 暂定不提交 → 主目标 coast（KF 冻结、云台保持）
                trace_event_pending_ = true;   // [N9] 暂定持有是关键事件 → 本帧触发文件 flush
            } else {
                if (!face_direct && reacq_defer_count_ >= need)
                    null_sink << "[PROVISIONAL] same-hyp sustained " << reacq_defer_count_
                              << " frames -> commit" << std::endl;
                if (!best.provisional_continuation) {
                    reacq_defer_count_ = 0;
                    pending_active_ = false;
                    pending_src_ = pending_idx_ = -1;   // [N4]
                }
                pending_from_sweep_ = false;
            }
        } else {
            if (pending_body_hyp_id_ >= 0 && best.body_hyp_id >= 0
                && best.body_hyp_id != pending_body_hyp_id_)
                clear_body_provisional(
                    "hyp_changed", "normal_body_commit",
                    best.source, best.index, best.reid_sim, best.anchor_sim,
                    -1.f, -1.f, -1.f, best.body_hyp_id);
            if (pending_body_hyp_id_ < 0) {
                reacq_defer_count_ = 0;
                pending_active_ = false;
                pending_src_ = pending_idx_ = -1;   // [N4]
            }
            pending_from_sweep_ = false;
        }
    }

    // 注：原独立 M2/VIS_GATE（头部新鲜但体框空间不连续 → 延迟）已折叠进上方统一暂定提交闸
    //     的 head_discont 触发臂，复用同一"同假设连续 K 帧"确认，不再单独门控
    //     （避免两闸各自递增/清零 reacq_defer_count_ 造成的计数语义混乱）。

    // ── 最终 measurement 可靠性层 ────────────────────────────────
    // 候选"通过原匹配门"与"可作为主目标 measurement"不是同一语义。
    // 危险/多人/低可见度/alert 中，缺少当前帧强身份证据的 BODY 仅保留为
    // UNCERTAIN 假设：继续参与下一帧 ReID/Face 搜索，但不得更新主 KF、feature
    // 或真实观测时钟。外层会并行尝试有限 short prediction 驱动 PTZ。
    bool has_evaluated_body_hypothesis = false;
    for (const auto& c : candidates) {
        if (c.reid_evaluated) {
            has_evaluated_body_hypothesis = true;
            break;
        }
    }

    if (best.matched) {
        const bool low_visibility_risk =
            visibility_state_ == VisibilityState::UPPER
            || visibility_state_ == VisibilityState::HEAD_ONLY;
        const bool strong_identity = best.from_face
                                  || best.provisional_continuation
                                  || best.strong_reid_direct
                                  || best.anchor_sim >= kAnchorDirectConfirm;
        const bool identity_risk = in_danger || close_det_count >= 2
                                || person_identity_risk_active
                                || id_switch_alert_ || low_visibility_risk;
        const bool isolated_head_support = !in_danger
                                        && !person_identity_risk_active
                                        && !id_switch_alert_
                                        && close_det_count <= 1
                                        && best.has_head
                                        && best.head_match >= kHeadBodyResumeMin;
        const bool reliable_measurement = !identity_risk
                                       || strong_identity
                                       || isolated_head_support;

        if (reliable_measurement) {
            best.reliability = MeasurementReliability::RELIABLE;
            frame_measurement_reliability_ = MeasurementReliability::RELIABLE;
            const cv::Mat& positive_src = best.source == 0 ? dets_one : dets_second;
            if ((best.source == 0 || best.source == 1)
                && best.index >= 0 && best.index < positive_src.rows) {
                mark_body_identity_evidence(
                    positive_src.row(best.index).colRange(0, 4),
                    IdentityEvidence::POSITIVE,
                    strong_identity ? "strong_identity" : "reliable_body");
            }
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[MEASUREMENT] f=%d class=RELIABLE src=%d idx=%d"
                    " hyp=%d provisional=%d risk=%d personRisk=%d strong=%d head=%d r=%.2f A=%.2f"
                    " kf_update=1 timer_reset=1",
                    frame_count, best.source, best.index, best.body_hyp_id,
                    best.provisional_continuation ? 1 : 0, identity_risk ? 1 : 0,
                    person_identity_risk_active ? 1 : 0,
                    strong_identity ? 1 : 0, isolated_head_support ? 1 : 0,
                    best.reid_sim, best.anchor_sim);
                trace_push(line);
                std::snprintf(line, sizeof(line),
                    "[IDENTITY_RECOVERY] f=%d source=BODY"
                    " sceneRiskStillActive=%d strong=%d src=%d idx=%d",
                    frame_count, person_identity_risk_active ? 1 : 0,
                    strong_identity ? 1 : 0, best.source, best.index);
                trace_push(line);
            }
        } else {
            const int uncertain_src = best.source;
            const int uncertain_idx = best.index;
            const float uncertain_reid = best.reid_sim;
            const float uncertain_anchor = best.anchor_sim;
            const cv::Mat& uncertain_src_dets =
                (uncertain_src == 0) ? dets_one : dets_second;
            if ((uncertain_src == 0 || uncertain_src == 1)
                && uncertain_idx >= 0 && uncertain_idx < uncertain_src_dets.rows) {
                const cv::Mat ub = uncertain_src_dets.row(uncertain_idx).colRange(0, 4);
                mark_body_identity_evidence(
                    ub, IdentityEvidence::UNKNOWN,
                    "insufficient_identity_evidence");
                pending_cx_ = 0.5f * (ub.at<float>(0, 0) + ub.at<float>(0, 2));
                pending_cy_ = 0.5f * (ub.at<float>(0, 1) + ub.at<float>(0, 3));
                pending_active_ = true;
                pending_src_ = uncertain_src;
                pending_idx_ = uncertain_idx;
            }
            frame_measurement_reliability_ = MeasurementReliability::UNCERTAIN;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[MEASUREMENT] f=%d class=UNCERTAIN action=hold src=%d idx=%d"
                    " hyp=%d reason=%s personRisk=%d r=%.2f A=%.2f"
                    " kf_update=0 feature_update=0 timer_reset=0",
                    frame_count, uncertain_src, uncertain_idx, best.body_hyp_id,
                    person_identity_risk_active
                        ? "person_occlusion_without_strong_identity"
                        : "identity_risk_without_strong_evidence",
                    person_identity_risk_active ? 1 : 0,
                    uncertain_reid, uncertain_anchor);
                trace_push(line);
                trace_event_pending_ = true;
            }
            best = MainMatchResult{};
            best.reliability = MeasurementReliability::UNCERTAIN;
        }
    } else if (has_evaluated_body_hypothesis || pending_active_
               || pending_from_sweep_ || face_global_pending_) {
        frame_measurement_reliability_ = MeasurementReliability::UNCERTAIN;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[MEASUREMENT] f=%d class=UNCERTAIN action=search"
                " candidates=%d pending=%d bodyHyp=%d facePending=%d"
                " kf_update=0 feature_update=0 timer_reset=0",
                frame_count, (int)candidates.size(), pending_active_ ? 1 : 0,
                pending_body_hyp_id_,
                face_global_pending_ ? 1 : 0);
            trace_push(line);
        }
    }

    // PTZ 实际滑动后的第一帧可靠 BODY 不能拿旧 KF 做普通 update：它属于新图像
    // 坐标系的重捕。立即输出真实观测，同时清空旧空间先验并开始两帧 feature warm-up。
    if (best.matched
        && frame_measurement_reliability_ == MeasurementReliability::RELIABLE
        && (best.source == 0 || best.source == 1)) {
        const cv::Mat& rebase_src = best.source == 0 ? dets_one : dets_second;
        if (best.index >= 0 && best.index < rebase_src.rows) {
            if (ptz_identity_recovery) {
                finish_ptz_blind_reacquisition(
                    main_trk_idx, rebase_src.row(best.index).colRange(0, 4));
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[IDENTITY_COMMIT] f=%d action=reacquire src=%d idx=%d hyp=%d"
                        " r=%.2f A=%.2f feature_freeze=1",
                        frame_count, best.source, best.index, best.body_hyp_id,
                        best.reid_sim, best.anchor_sim);
                    trace_push(line);
                }
            } else if (ptz_reacq_body_streak_ > 0 && ptz_reacq_body_streak_ < 3) {
                ++ptz_reacq_body_streak_;
            }
            update_ptz_blind_anchor(rebase_src.row(best.index).colRange(0, 4), now_match_ms);
        }
    }

    // 强主身份候选最终提交后，才修复与它冲突的二级轨迹。隔离会立即剥夺否决权；
    // coexist 清零要求该轨迹若重新找到真正的他人，必须重新积累同帧共存证据。
    if (best.matched
        && best_ownership_repair_trk >= 0
        && best_ownership_repair_trk < (int)trackers.size()
        && best.source == best_ownership_repair_src
        && best.index == best_ownership_repair_det) {
        auto& shadow = trackers[best_ownership_repair_trk];
        if (!shadow->get_is_main()) {
            int shadow_id = shadow->get_id();
            int old_coexist = shadow->coexist_with_main;
            shadow->quarantined_ = true;
            shadow->quarantine_clear_streak_ = 0;
            shadow->coexist_with_main = 0;
            secondary_frame_observations_.erase(shadow_id);
            relative_motion_history_.erase(shadow_id);
            // 若它正是 OCC onset 记录的遮挡者，其几何身份也已随 IoU 换人而失效；
            // 继续拿它计算 anti_occ / 浮现点会反向惩罚刚确认的主目标。
            if (occluder_tracker_id_ == shadow_id) {
                occluder_tracker_id_ = -1;
                emergence_valid_ = false;
                emergence_update_ms_ = -1;
            }
            null_sink << "[COEXIST_REPAIR] quarantined ownership-swapped id="
                         << shadow_id << " old_coexist=" << old_coexist << std::endl;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[COEXIST_REPAIR] f=%d other=%d oldCoexist=%d main_src=%d main_idx=%d",
                    frame_count, shadow_id, old_coexist, best.source, best.index);
                trace_push(line);
                trace_event_pending_ = true;
            }
        }
    }

    // 真正提交的非人脸身体观测才会把本次 OCC 窗口标为“吃过 body”；被最终门拒绝的
    // preliminary 候选不再污染该 sticky 状态。
    bool committed_during_occ = best.matched
                             && occlusion_state_ == OcclusionState::OCCLUDED;
    if (committed_during_occ && !best.from_face) {
        occ_kf_clean_ = false;
        null_sink << "[OCC_COMMIT] trusted body observation committed" << std::endl;
    }

    // 当前帧已有强身份真观测且场景已经分离时，不必等下一帧 main_tsu 回流才能退出 OCC。
    // 仅改变状态推进时机；候选仍须先通过全部身份/共存/暂定/提交门。
    bool direct_identity_commit = best.matched
                               && (best.from_face
                                   || best.strong_reid_direct
                                   || best.anchor_sim >= kAnchorDirectConfirm);
    if (committed_during_occ && overlap_count <= 1 && direct_identity_commit) {
        occlusion_state_ = OcclusionState::RECOVERING;
        recovery_start_ms_ = now_match_ms;
        separation_streak_ = 0;
        null_sink << "[STATE] OCCLUDED -> RECOVERING (direct identity commit)"
                     << std::endl;
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[STATE] f=%d OCC->RECOV reason=direct_identity overlap=%d src=%d idx=%d",
                frame_count, overlap_count, best.source, best.index);
            trace_push(line);
            trace_event_pending_ = true;
        }
    }

    // 朝向/光照基线也只能由最终提交的身体观测更新；preliminary 低信任候选不得留下副作用。
    if (best.matched && !best.from_face) {
        prev_incumbent_anchor_ = (prev_incumbent_anchor_ < 0.f)
                               ? best.anchor_sim
                               : 0.5f * prev_incumbent_anchor_ + 0.5f * best.anchor_sim;
    }

    // 只用可信真实观测移动下一轮全图人体搜索圆心。正常孤立 CLEAR 命中可持续跟随；
    // 多人/危险期则必须有强 ReID、强 anchor 或人脸身份，弱 BODY 不得把圆心带到路人身上。
    const bool trusted_body_search_obs = best.matched
        && (best.from_face || best.strong_reid_direct
            || best.anchor_sim >= kAnchorDirectConfirm
            || (occlusion_state_ == OcclusionState::CLEAR
                && proximity.close_det_count <= 1 && !id_switch_alert_));
    if (trusted_body_search_obs && (best.source == 0 || best.source == 1)) {
        const cv::Mat& trusted_src = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < trusted_src.rows)
            update_body_reid_search_anchor(
                trusted_src.row(best.index).colRange(0, 4));
    }

    // 质量监控只能观察最终提交结果；被 provisional/commit gate 持有的候选不是主目标命中。
    update_quality_monitor(best, proximity.close_det_count);

    null_sink << "FINAL: matched=" << best.matched
              << " src=" << best.source
              << " idx=" << best.index
              << " reid=" << best.reid_sim
              << " anchor=" << best.anchor_sim
              << " total=" << best.total_score
              << " face=" << best.from_face
              << " alert=" << id_switch_alert_ << std::endl;

    // ── [MATCH] 每帧主目标决策摘要（聚焦单行）──
    //   放在所有会改写 best.matched 的门控（REACQ-GATE / VIS_GATE）之后 → 反映最终决策。
    if (kMatchTrace) {
        const char* st = (occlusion_state_ == OcclusionState::CLEAR)    ? "CLEAR"
                       : (occlusion_state_ == OcclusionState::OCCLUDED) ? "OCC"
                                                                        : "RECOV";
        const char* vs = (visibility_state_ == VisibilityState::FULL)        ? "FULL"
                       : (visibility_state_ == VisibilityState::MOSTLY_FULL) ? "MFUL"
                       : (visibility_state_ == VisibilityState::HALF)        ? "HALF"
                       : (visibility_state_ == VisibilityState::UPPER)       ? "UPPR"
                                                                             : "HEAD";
        const char* meas =
            frame_measurement_reliability_ == MeasurementReliability::RELIABLE ? "REL"
          : frame_measurement_reliability_ == MeasurementReliability::UNCERTAIN ? "UNC"
                                                                                 : "NONE";
        float top0 = candidates.empty()      ? 0.f : candidates[0].total;
        float top1 = (candidates.size() >= 2) ? candidates[1].total : 0.f;
        // [N9] 写入完整缓冲 trace 时间线（不再每帧 UART 直出）。
        char _ml[kTraceLineLen];
        std::snprintf(_ml, sizeof(_ml),
            "[MATCH] f=%d st=%s vis=%s/%.2f near=%d ovl=%d close=%d head=%s(tsu=%d)"
            " | matched=%d meas=%s personRisk=%d src=%d idx=%d hyp=%d provisional=%d"
            " face=%d reid=%.2f anc=%.2f total=%.2f"
            " top2=(%.2f,%.2f) alert=%d blind=%lldms",
            frame_count, st, vs, visible_ratio_ema_, near_box.rows,
            overlap_count, close_det_count, (head_track_fresh ? "fresh" : "stale"), head_tsu,
            (best.matched ? 1 : 0), meas, person_identity_risk_active ? 1 : 0,
            best.source, best.index, best.body_hyp_id,
            best.provisional_continuation ? 1 : 0, (best.from_face ? 1 : 0),
            best.reid_sim, best.anchor_sim, best.total_score, top0, top1,
            (id_switch_alert_ ? 1 : 0), (long long)get_blind_ms());
        trace_push(_ml);
    }

    // ──────────────────────────────────────────────────────
    // 非人脸高置信样本入锚点画廊（次强证据，比人脸确认更保守）
    //   条件：本帧已接受匹配（经 C-identity 复核）+ 非人脸 + 全身可见 +
    //         无贴近人 + anchor 置信 + 周期限制（控推理开销）。
    //   目的：注册样本糟糕时，让后续高质量全身样本逐步淘汰它（自愈）。
    //   保守理由：anchor_sim=max(画廊) 会放宽身份门，过度添加会侵蚀防 ID-switch。
    //   放在 C-identity 之后：仅对最终接受的帧入廊（被延迟的弱重捕不入廊）。
    // ──────────────────────────────────────────────────────
    if (best.matched && !best.from_face
        && !(ptz_reacq_body_streak_ > 0 && ptz_reacq_body_streak_ < 3)
        && visibility_state_ == VisibilityState::FULL
        && proximity.close_det_count <= 1
        && best.anchor_sim >= kAnchorAddAnchorMin
        && (frame_count % kAnchorAddPeriod == 0)) {
        const cv::Mat& add_src = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < add_src.rows) {
            // 画廊维护不是实时身份判定，不得突破本帧 ReID 预算；没有可复用特征则延后。
            cv::Mat add_emb = best.emb;
            if (!add_emb.empty()) {
                float q = 0.5f + 0.5f * visible_ratio_ema_;
                trackers[main_trk_idx]->add_anchor_sample(add_emb, q, visible_ratio_ema_);  // Q1：vis 标注（此处 FULL 门 → HIGH 带）
                null_sink << "[ANCHOR] gallery += high-conf (q=" << q
                          << " anchor=" << best.anchor_sim
                          << " size=" << trackers[main_trk_idx]->get_anchor_gallery_size()
                          << ")" << std::endl;
            }
        }
    }

    // ──────────────────────────────────────────────────────
    // 时空连续性认证的"新视角"样本入廊（解决"画廊学不到背面"死锁）
    //   高置信入廊要求 anchor≥0.70 → 背面/侧面（0.3~0.6）永远进不去；
    //   人脸入廊需要正脸 → 背面同样没有。结果：画廊只有正面视角，目标一转身
    //   anchor_sim 即跌破否决线 →"转身即丢"。此处身份确定性来自时空连续而非
    //   外观：CLEAR + 无邻人（ID switch 需第二人在场，孤立场景不可能）+ 长连续
    //   命中 + 无嫌疑/警报。质量 kViewAddQuality=0.55：可淘汰注册样本（0.5），
    //   永远低于人脸认证样本（0.6+），不会反向侵蚀画廊。
    // ──────────────────────────────────────────────────────
    if (best.matched && !best.from_face
        && !(ptz_reacq_body_streak_ > 0 && ptz_reacq_body_streak_ < 3)
        && occlusion_state_ == OcclusionState::CLEAR
        && proximity.close_det_count <= 1
        && !id_switch_alert_ && suspect_streak_ == 0
        && best.anchor_sim <  kAnchorAddAnchorMin           // ≥0.70 已由上一块覆盖
        && best.anchor_sim >= kAnchorVetoRelaxed + 0.05f    // 完全陌生的外观不收
        && visible_ratio_ema_ >= 0.60f
        && trackers[main_trk_idx]->get_hit_streak() >= kViewAddMinStreak
        && (frame_count % kAnchorAddPeriod == 0)) {
        const cv::Mat& vsrc = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < vsrc.rows) {
            // 新视角入廊同样只复用本帧已有特征，禁止额外 ReID 推理。
            cv::Mat vemb = best.emb;
            if (!vemb.empty()) {
                trackers[main_trk_idx]->add_anchor_sample(vemb, kViewAddQuality, visible_ratio_ema_);  // Q1：vis 标注（此处 ≥0.60 门 → HIGH 带）
                null_sink << "[ANCHOR] gallery += new-view (anchor=" << best.anchor_sim
                          << " streak=" << trackers[main_trk_idx]->get_hit_streak()
                          << " size=" << trackers[main_trk_idx]->get_anchor_gallery_size()
                          << ")" << std::endl;
            }
        }
    }

    // ──────────────────────────────────────────────────────
    // 全身高度保持值更新 + 部分遮挡框补全（防 PTZ 误抬，见 .h 注释）
    // ──────────────────────────────────────────────────────
    if (best.matched && occlusion_state_ == OcclusionState::CLEAR
        && (visibility_state_ == VisibilityState::FULL
            || visibility_state_ == VisibilityState::MOSTLY_FULL)) {
        // 仅安全情境更新保持值：遮挡/半身帧不污染
        const cv::Mat& hsrc = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < hsrc.rows) {
            float hh = hsrc.at<float>(best.index, 3) - hsrc.at<float>(best.index, 1);
            if (hh > 1.f) {
                main_h_hold_ = (main_h_hold_ <= 0.f)
                             ? hh
                             : (1.f - kBoxCompleteAlpha) * main_h_hold_
                               + kBoxCompleteAlpha * hh;
                main_h_hold_ms_ = now_match_ms;   // B8：保持值新鲜度戳
            }
        }
    }

    // B8：补全三护栏——① 保持值须新鲜（危险期中途坐下/走远 → 陈旧站姿高会过度拉伸）；
    //     ② 纵横比上限（宽是"自下而上遮挡"时仍真实的观测 → 兼作尺度自适应）；
    //     ③ 亚军护栏：缺失区的"遮挡证据"若本身是与赢家近分的候选（疑真主目标/同衣
    //       歧义者）→ 不作证据（别把主框画到他身上）。失败方向安全：框保持截断。
    if (best.matched && in_danger && main_h_hold_ > 1.f
        && (now_match_ms - main_h_hold_ms_) <= kBoxCompleteHoldMaxMs) {
        // 注意：msrc 与 det_groups 共享底层数据，原地补全使 KF 更新/输出行/
        // add_other_det 去重/预测器 on_matched 全部看到同一个补全后的框
        // （也顺带避免半身框触发预测器的"焦距突变"误判清窗）。
        cv::Mat msrc = (best.source == 0) ? dets_one : dets_second;
        if (best.index >= 0 && best.index < msrc.rows) {
            float* r = msrc.ptr<float>(best.index);
            float h_obs = r[3] - r[1];
            float w_obs = r[2] - r[0];
            if (h_obs > 1.f && w_obs > 1.f && h_obs < main_h_hold_ * kBoxCompleteTrig) {
                float y2_full = std::min(r[1] + main_h_hold_, (float)img_h - 1.f);
                // ② 纵横比上限：补全高 ≤ kBoxCompleteMaxAspect × 观测宽
                y2_full = std::min(y2_full, r[1] + kBoxCompleteMaxAspect * w_obs);
                if (y2_full > r[3] + 2.f) {
                    // 遮挡证据：缺失的下半区必须与他人检测重叠（排除走远变小）
                    cv::Mat gap = (cv::Mat_<float>(1, 4) << r[0], r[3], r[2], y2_full);
                    bool evid = false;
                    // ③ 亚军护栏：该检测是否为与赢家近分的已评分候选
                    auto is_runner_up = [&](int src, int i) -> bool {
                        for (const auto& rc : candidates)
                            if (rc.source == src && rc.det_index == i
                                && rc.total >= best.total_score - kAmbiguousGapDanger)
                                return true;
                        return false;
                    };
                    auto chk = [&](const cv::Mat& dd, int src) {
                        for (int i = 0; i < dd.rows && !evid; ++i) {
                            if (src == best.source && i == best.index) continue;
                            if (Utils::iou_single(gap, dd.row(i).colRange(0, 4))
                                    > kBoxCompleteEvidIou
                                && !is_runner_up(src, i))
                                evid = true;
                        }
                    };
                    chk(dets_one, 0);
                    chk(dets_second, 1);
                    if (evid) {
                        null_sink << "[BOX_COMPLETE] h_obs=" << h_obs
                                  << " -> hold=" << main_h_hold_
                                  << " (y2 " << r[3] << " -> " << y2_full << ")"
                                  << std::endl;
                        r[3] = y2_full;
                    }
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────
    // STEP 4：构造 matches，更新 tracker
    // ──────────────────────────────────────────────────────
    cv::Mat matches_one = cv::Mat::zeros(0, 2, CV_32S);
    cv::Mat matches_second = cv::Mat::zeros(0, 2, CV_32S);
    std::vector<int> unmatched_trks;

    if (best.matched) {
        if (best.source == 0)
            matches_one = (cv::Mat_<int>(1, 2) << main_trk_idx, best.index);
        else
            matches_second = (cv::Mat_<int>(1, 2) << main_trk_idx, best.index);
    } else {
        unmatched_trks.push_back(main_trk_idx);
    }

    // ──────────────────────────────────────────────────────
    // STEP 5：Embedding 更新决策
    // ──────────────────────────────────────────────────────
    bool allow_emb_update = should_update_embedding(best, proximity.close_det_count);
    if (kMatchTrace) {
        const char* feature_reason = allow_emb_update ? "eligible"
            : !best.matched ? "no_reliable_measurement"
            : (ptz_reacq_body_streak_ > 0 && ptz_reacq_body_streak_ < 3)
                ? "ptz_reacq_warmup"
            : occlusion_state_ != OcclusionState::CLEAR ? "occlusion_or_recovery"
            : id_switch_alert_ ? "id_switch_alert"
            : proximity.close_det_count > 1 ? "crowded"
            : (visibility_state_ != VisibilityState::FULL
               && visibility_state_ != VisibilityState::MOSTLY_FULL)
                ? "partial_visibility" : "anchor_floor";
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FEATURE_GATE] f=%d update=%d reason=%s matched=%d meas=%d"
            " close=%d state=%d vis=%d A=%.2f",
            frame_count, allow_emb_update ? 1 : 0, feature_reason,
            best.matched ? 1 : 0, (int)best.reliability,
            proximity.close_det_count, (int)occlusion_state_,
            (int)visibility_state_, best.anchor_sim);
        trace_push(line);
    }

    std::vector<int> picked_idx;
    if (allow_emb_update && (frame_count % 15 == 0))
        picked_idx.push_back(0);

    // ── 更新主目标头部 KF（与 body KF 同帧更新）──
    //   匹配到检测且该检测关联了头部 → 校正；否则纯预测推进（头部被遮挡）。
    {
        // S2b：与 body KF 同款遮挡保护。遮挡期非人脸 off-trajectory 匹配若吸遮挡者的头 →
        //   头部预测漂到遮挡者 → 分离期真目标 head_match 偏低被 [HEAD_VETO] 误删 → 反助遮挡者夺锁；
        //   且会污染"头→体"几何。故遮挡期低信任匹配 → 头部 KF 亦纯预测、不学几何（与 S2 一致的信任判据）。
        bool occ_head_protect = false;
        if (occlusion_state_ == OcclusionState::OCCLUDED && best.matched && !best.from_face) {
            const cv::Mat& body_src = (best.source == 0) ? dets_one : dets_second;
            float iou_pred = (best.index >= 0 && best.index < body_src.rows)
                           ? Utils::iou_single(trk_box, body_src.row(best.index).colRange(0, 4)) : 0.f;
            // issue 2：同 S2——连续在场+全身可见的前景目标即便与陈旧头预测 IoU 低也跟随(否则头 KF 亦冻结)。
            int     main_tsu_hd = trackers[main_trk_idx]->get_time_since_update();
            int64_t blind_ms_hd = get_body_blind_ms();
            bool front_visible_hd = (visibility_state_ == VisibilityState::FULL
                                  || visibility_state_ == VisibilityState::MOSTLY_FULL)
                                 && main_tsu_hd <= kFrontFollowMaxTsu
                                 && blind_ms_hd >= 0 && blind_ms_hd <= kFrontFollowMaxMs;
            bool strong_reid_hd = best.strong_reid_direct;
            bool strong_anchor_hd = best.anchor_sim >= kAnchorDirectConfirm;
            occ_head_protect = !((best.anchor_sim >= kReacqAnchorConfident
                                  && (iou_pred >= kVetoRelaxIou || front_visible_hd))
                                 || strong_reid_hd
                                 || strong_anchor_hd);
            if (occ_head_protect)
                null_sink << "[KF_PROTECT] OCCLUDED low-trust -> head KF pure-predict" << std::endl;
        }
        std::optional<cv::Mat> head_obs;
        if (best.matched && !occ_head_protect) {
            const cv::Mat& head_src = (best.source == 0) ? det_groups.dets_one_head
                                                         : det_groups.dets_second_head;
            if (best.index >= 0 && best.index < head_src.rows) {
                cv::Mat hb = head_src.row(best.index).colRange(0, 4);
                if (cv::countNonZero(hb) > 0) {
                    head_obs = hb.clone();
                    // 头部连续性（A）：仅可靠、无遮挡且人体未触边时学习完整
                    // “头→体”几何。近距裁剪框或家具后的半身框不能污染完整比例。
                    const cv::Mat& body_src = (best.source == 0) ? dets_one : dets_second;
                    if (best.index < body_src.rows) {
                        cv::Mat bb = body_src.row(best.index).colRange(0, 4);
                        const float margin = 3.f;
                        const bool reliable_geom =
                            occlusion_state_ == OcclusionState::CLEAR
                            && proximity.close_det_count <= 1
                            && !id_switch_alert_
                            && (visibility_state_ == VisibilityState::FULL
                                || visibility_state_ == VisibilityState::MOSTLY_FULL)
                            && bb.at<float>(0, 0) > margin
                            && bb.at<float>(0, 1) > margin
                            && bb.at<float>(0, 2) < (float)img_w - 1.f - margin
                            && bb.at<float>(0, 3) < (float)img_h - 1.f - margin;
                        if (reliable_geom)
                            learn_head_body_geom(bb, hb);
                    }
                }
            }
        }
        trackers[main_trk_idx]->update_head(head_obs);
        if (head_obs.has_value())
            head_ego_dx_ = head_ego_dy_ = 0.f;
    }

    // 可靠完整人体 + 唯一关联脸：学习脸→完整人体比例。只用干净单人 CLEAR 帧，
    // 且人体不得触碰画面边界；近距离裁剪框不能反向污染完整比例。
    if (best.matched && occlusion_state_ == OcclusionState::CLEAR
        && proximity.close_det_count <= 1 && !id_switch_alert_
        && (visibility_state_ == VisibilityState::FULL
            || visibility_state_ == VisibilityState::MOSTLY_FULL)) {
        const cv::Mat& body_src = (best.source == 0) ? dets_one : dets_second;
        const auto& face_src = (best.source == 0)
                             ? det_groups.dets_one_face
                             : det_groups.dets_second_face;
        if (best.index >= 0 && best.index < body_src.rows
            && best.index < (int)face_src.size()
            && face_src[best.index].rows == 1
            && face_src[best.index].cols >= 5
            && face_src[best.index].at<float>(0, 4) >= 0.50f) {
            cv::Mat bb = body_src.row(best.index).colRange(0, 4);
            const float bx1 = bb.at<float>(0, 0), by1 = bb.at<float>(0, 1);
            const float bx2 = bb.at<float>(0, 2), by2 = bb.at<float>(0, 3);
            const float margin = 3.f;
            if (bx1 > margin && by1 > margin
                && bx2 < (float)img_w - 1.f - margin
                && by2 < (float)img_h - 1.f - margin) {
                learn_face_body_geom(
                    bb, face_src[best.index].row(0).colRange(0, 4));
            }
        }
    }

    // 所有身份、共存、OCC、provisional 和人脸门均已完成；只允许最终 BODY winner
    // 提交 Pose。歧义阶段已有结果则复用，CLEAR 到刷新周期才购买一次 refresh。
    if (best.matched && (best.source == 0 || best.source == 1)) {
        const cv::Mat& pose_src = best.source == 0 ? dets_one : dets_second;
        if (best.index >= 0 && best.index < pose_src.rows) {
            cv::Mat winner_box = pose_src.row(best.index).colRange(0, 4);
            PoseCacheEntry* pose_entry = find_cached_pose(best.source, best.index);
            if (pose_entry == nullptr
                && occlusion_state_ == OcclusionState::CLEAR
                && (last_committed_pose_frame_ < 0
                    || frame_count - last_committed_pose_frame_ >= kPoseInferEveryN)) {
                pose_entry = request_pose(img, winner_box, best.source, best.index,
                                          PoseReason::TARGET_REFRESH);
            }
            if (pose_entry && pose_entry->status == PoseRequestStatus::AVAILABLE)
                commit_pose(pose_entry->pose, winner_box, img,
                            pose_budget_used_ >= 2 ? "ambiguity_winner" : "target_refresh");
            const int64_t obs_ms = now_ms();
            last_real_obs_ms_ = obs_ms;
            last_body_observation_ms_ = obs_ms;
            if (kMatchTrace) {
                char line[kTraceLineLen];
                std::snprintf(line, sizeof(line),
                    "[OBS_CLOCK] f=%d kind=BODY any=%lld body=%lld",
                    frame_count, (long long)last_real_obs_ms_,
                    (long long)last_body_observation_ms_);
                trace_push(line);
            }
        }
    }

    update_trackers_unified(
        matches_one, matches_second, unmatched_trks,
        dets_one, dets_second, picked_idx, best, img);

    // ── 非主目标轻量轨迹维护（occluder 识别 / 共存排除 / id=2,3,4… 输出）──
    update_secondary_tracks(det_groups, best, img);
    // ── 非主轨迹 ReID 轮询刷新（不占候选 Pose 预算）──
    update_secondary_features(det_groups, img);

    // ── 延迟/重新人脸注册（稳定跟踪时自动提升人脸模板质量）──
    //   传入本帧匹配结果作身份安全闸（详见函数头注释）
    try_deferred_face_register(img, det_groups, best, proximity.close_det_count);

    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[POSE_FRAME] f=%d count=%d cache=%zu state=%d winner=%d/%d matched=%d",
            frame_count, pose_budget_used_, pose_cache_.size(),
            (int)occlusion_state_, best.source, best.index, best.matched ? 1 : 0);
        trace_push(line);
    }

    return generate_final_results();
}

// ============================================================
// 判断这一帧是否允许更新主目标 embedding
// 核心防线：anchor 验证 + 邻近检测 + 质量监控
// ============================================================
bool LightTracker::should_update_embedding(
    const MainMatchResult& match_result,
    int close_det_count)
{
    if (!match_result.matched) return false;

    // PTZ 重捕后先让新坐标系内的真实 BODY 连续稳定，再允许模板自适应。
    if (ptz_reacq_body_streak_ > 0 && ptz_reacq_body_streak_ < 3) return false;

    // 遮挡/恢复期间：完全冻结 embedding（防止被交错的人污染）
    if (occlusion_state_ != OcclusionState::CLEAR) return false;

    // ID switch 警报期间：完全冻结 embedding
    if (id_switch_alert_) return false;

    // P2：有任何邻人(close_det_count≥2)一律冻结模板。旧逻辑保留了一个"双邻人 + reid>0.80 &&
    //   anchor>0.70 && total>0.65 即放行"的逃逸——但同衣冒充者贴近时 reid/anchor 恰恰同样高，
    //   该逃逸在最危险的同衣场景把模板 EMA 拉向冒充者 → 冒充者随后过 anchor 门（自强化夺锁）。
    //   模板自适应只在"确证独处"时进行；多视角/转身鲁棒性由 anchor 画廊(add_anchor_sample)另行承担。
    if (close_det_count > 1) return false;

    // 半身/上身裁剪的 emb 已退化（含遮挡物像素或缺躯干），用它 EMA 会把模板拉向"仅上身"表征
    //   → 同样冻结。注意转身不降可见度，正面全身转身期仍可适应，不损 reid 抗转身能力。
    if (visibility_state_ != VisibilityState::FULL
        && visibility_state_ != VisibilityState::MOSTLY_FULL) return false;

    // 独处 + 躯干基本完整：放宽 anchor 下限到 kEmbAdaptAnchorFloor，让 emb 在转身/走远的
    //   渐变外观中小步适应（EMA alpha + 每 15 帧一次，本身即慢速通道）。
    return match_result.anchor_sim >= kEmbAdaptAnchorFloor;
}


// ============================================================
// 非主目标轻量轨迹维护（纯运动/IoU 关联，不存 emb）
//   ① 复活 occluder 识别与 anti_occ（原先 trackers 从无非主轨迹，
//      遮挡者排斥一直是死代码）；
//   ② 共存排除依据：与主目标同帧可见的轨迹不可能是主目标本人；
//   ③ 稳定轨迹经 generate_final_results 自然输出 id=2,3,4…。
// ============================================================
void LightTracker::update_secondary_tracks(const DetectionGroups& det_groups,
                                           const MainMatchResult& best,
                                           const cv::Mat& img)
{
    (void)img;
    cv::Mat reliable_main_detection;
    if (best.matched && best.reliability == MeasurementReliability::RELIABLE
        && (best.source == 0 || best.source == 1)) {
        const cv::Mat& main_dets = best.source == 0
                                 ? det_groups.dets_one : det_groups.dets_second;
        if (best.index >= 0 && best.index < main_dets.rows)
            reliable_main_detection =
                main_dets.row(best.index).colRange(0, 4).clone();
    }

    // ── 收集本帧检测（排除被主目标占用的那一个）──
    struct SecDet {
        cv::Mat box4;
        float score;
        bool high;
        int source;
        int index;
    };
    std::vector<SecDet> dets;
    auto collect = [&](const cv::Mat& m, int src, bool high) {
        for (int i = 0; i < m.rows; ++i) {
            if (best.matched && best.source == src && best.index == i)
                continue;
            // [N4] 暂定假设（provisional-commit）对应的检测从二级收集中排除：否则被延迟提交的
            //   主目标检测会写进"他人"轨迹/emb → 反过来成为否决真目标的负证据（自证冒充者）。
            if (pending_active_ && src == pending_src_ && i == pending_idx_)
                continue;
            dets.push_back({ m.row(i).colRange(0, 4).clone(),
                             m.at<float>(i, 4), high, src, i });
        }
    };
    collect(det_groups.dets_one,    0, true);
    collect(det_groups.dets_second, 1, false);

    // ── 非主轨迹及其当前状态框（KF 状态已含 GMC 补偿）──
    std::vector<int>     sec_idx;
    std::vector<cv::Mat> sec_box;
    cv::Mat main_box;
    for (int t = 0; t < (int)trackers.size(); ++t) {
        if (trackers[t]->get_is_main()) { main_box = trackers[t]->get_state(); continue; }
        sec_idx.push_back(t);
        sec_box.push_back(trackers[t]->get_state());
    }

    // ── 贪心 IoU 关联：每轮取全局最大 IoU 配对 ──
    //   B9 颜色封禁矩阵已随颜色直方图弃用一并移除 → 纯 IoU 贪心关联。
    std::vector<char> det_used(dets.size(), 0);
    std::vector<char> trk_hit(sec_idx.size(), 0);
    for (;;) {
        float best_iou = kSecTrkIou;
        int bi = -1, bj = -1;
        for (int i = 0; i < (int)sec_idx.size(); ++i) {
            if (trk_hit[i]) continue;
            for (int j = 0; j < (int)dets.size(); ++j) {
                if (det_used[j]) continue;
                float iou = Utils::iou_single(sec_box[i], dets[j].box4);
                if (iou > best_iou) { best_iou = iou; bi = i; bj = j; }
            }
        }
        if (bi < 0) break;
        auto& trk = trackers[sec_idx[bi]];

        // 只记录 association 可辨识度，不改变全局最大 IoU 贪心结果。row 表示
        // “该 tracker 是否不知道该选哪个 detection”；column 表示“该 detection
        // 是否同时可被另一 tracker 合理拥有”。runner 使用本轮尚未占用的集合。
        float row_second_iou = 0.f;
        float column_second_iou = 0.f;
        int row_candidate_count = 0;
        int column_candidate_count = 0;
        if (kMatchTrace) {
            for (int j = 0; j < (int)dets.size(); ++j) {
                if (det_used[j] || j == bj) continue;
                const float iou = Utils::iou_single(sec_box[bi], dets[j].box4);
                row_second_iou = std::max(row_second_iou, iou);
                if (iou >= kSecTrkIou) ++row_candidate_count;
            }
            for (int i = 0; i < (int)sec_idx.size(); ++i) {
                if (trk_hit[i] || i == bi) continue;
                const float iou = Utils::iou_single(sec_box[i], dets[bj].box4);
                column_second_iou = std::max(column_second_iou, iou);
                if (iou >= kSecTrkIou) ++column_candidate_count;
            }
        }

        // A-relative competition 诊断：比较选中 detection 与其它尚未占用 detection
        // 对上一条已接受相对 observation 的归一化几何距离。它不参与 association，
        // 也不决定 history 是否写入，仅供 replay 判断多个 secondary 是否同样合理。
        float relative_selected_cost = -1.f;
        float relative_second_cost = -1.f;
        int64_t relative_previous_dt_ms = -1;
        if (kMatchTrace) {
            auto relative_it = relative_motion_history_.find(trk->get_id());
            if (!reliable_main_detection.empty()
                && relative_it != relative_motion_history_.end()
                && !relative_it->second.empty()) {
                const RelativeCenterObservation& previous = relative_it->second.back();
                const float main_x1 = reliable_main_detection.at<float>(0, 0);
                const float main_y1 = reliable_main_detection.at<float>(0, 1);
                const float main_x2 = reliable_main_detection.at<float>(0, 2);
                const float main_y2 = reliable_main_detection.at<float>(0, 3);
                const float main_w = std::max(1.f, main_x2 - main_x1);
                const float main_h = std::max(1.f, main_y2 - main_y1);
                const float main_cx = 0.5f * (main_x1 + main_x2);
                const float main_cy = 0.5f * (main_y1 + main_y2);
                auto relative_cost = [&](const cv::Mat& box) {
                    const float cx = 0.5f * (box.at<float>(0, 0)
                                           + box.at<float>(0, 2));
                    const float cy = 0.5f * (box.at<float>(0, 1)
                                           + box.at<float>(0, 3));
                    const float rx = (cx - main_cx) / main_w;
                    const float ry = (cy - main_cy) / main_h;
                    return std::hypot(rx - previous.normalized_x,
                                      ry - previous.normalized_y);
                };
                relative_selected_cost = relative_cost(dets[bj].box4);
                float runner = FLT_MAX;
                for (int j = 0; j < (int)dets.size(); ++j) {
                    if (det_used[j] || j == bj) continue;
                    runner = std::min(runner, relative_cost(dets[j].box4));
                }
                if (runner < FLT_MAX) relative_second_cost = runner;
                relative_previous_dt_ms = current_frame_timestamp_ms_
                                        - previous.timestamp_ms;
            }
        }
        // ── B9 颜色门已移除（颜色直方图整体弃用，弊大于利）──
        //   原用颜色相关性禁止跨人贪心配对（防交错时 id 互换）。移除后二级轨迹关联纯按 IoU 贪心；
        //   跨人互换风险由 emb 外观排除 / 共存否决 / 隔离(quarantine) 承担（二级轨迹互换不影响主目标身份）。
        trk_hit[bi] = 1;
        det_used[bj] = 1;
        std::optional<cv::Mat> ob = dets[bj].box4;
        trk->update(ob);
        trk->last_det_score_ = dets[bj].score;   // 注册/刷新优先级用
        trk->last_update_ms_ = now_ms();         // B6：共存排除墙钟新鲜度用
        // ── B3 隔离转正：连续 kQuarantineClearFrames 帧自有检测明显离开主框
        //   → 自证独立他人（影子的"自有检测"就是主目标检测，做不到）。──
        if (trk->quarantined_) {
            bool clear_of_main = main_box.empty()
                || Utils::iou_single(main_box, dets[bj].box4) < kQuarantineClearIou;
            trk->quarantine_clear_streak_ = clear_of_main
                ? trk->quarantine_clear_streak_ + 1 : 0;
            if (trk->quarantine_clear_streak_ >= kQuarantineClearFrames) {
                trk->quarantined_ = false;
                null_sink << "[QUARANTINE] cleared id=" << trk->get_id()
                             << " (" << kQuarantineClearFrames
                             << " frames clear of main)" << std::endl;
            }
        }
        SecondaryFrameObservation frame_obs;
        frame_obs.frame_id = frame_count;
        frame_obs.timestamp_ms = current_frame_timestamp_ms_;
        frame_obs.box = dets[bj].box4.clone();
        frame_obs.detection_source = dets[bj].source;
        frame_obs.detection_index = dets[bj].index;
        frame_obs.association_iou = best_iou;
        frame_obs.row_second_iou = row_second_iou;
        frame_obs.column_second_iou = column_second_iou;
        frame_obs.row_candidate_count = row_candidate_count;
        frame_obs.column_candidate_count = column_candidate_count;
        frame_obs.relative_selected_cost = relative_selected_cost;
        frame_obs.relative_second_cost = relative_second_cost;
        frame_obs.relative_previous_dt_ms = relative_previous_dt_ms;
        secondary_frame_observations_[trk->get_id()] = frame_obs;

        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[SEC_ASSOC] f=%d other=%d det=%d/%d iou=%.3f"
                " row2=%.3f rowGap=%.3f rowAlt=%d"
                " col2=%.3f colGap=%.3f colAlt=%d",
                frame_count, trk->get_id(), dets[bj].source, dets[bj].index,
                best_iou, row_second_iou, best_iou - row_second_iou,
                row_candidate_count, column_second_iou,
                best_iou - column_second_iou, column_candidate_count);
            trace_push(line);
            std::snprintf(line, sizeof(line),
                "[SEC_ASSOC_REL] f=%d other=%d det=%d/%d selected=%.3f"
                " runner=%.3f margin=%.3f prevDt=%lldms diagnostic_only=1",
                frame_count, trk->get_id(), dets[bj].source, dets[bj].index,
                relative_selected_cost, relative_second_cost,
                relative_second_cost >= 0.f && relative_selected_cost >= 0.f
                    ? relative_second_cost - relative_selected_cost : -1.f,
                (long long)relative_previous_dt_ms);
            trace_push(line);
        }

        // quarantine 表示该二级轨迹当前不具备可靠人物所有权；旧的双真实相对历史
        // 随所有权一起失效，不能继续用于 occluder emergence。
        if (trk->quarantined_)
            relative_motion_history_.erase(trk->get_id());

        if (!reliable_main_detection.empty() && !trk->quarantined_) {
            record_relative_motion_observation(
                trk->get_id(), reliable_main_detection, dets[bj].box4);
            // onset 前历史只有两点时，本帧第三个双真实点可使方向首次有效。
            // 只在尚未冻结方向时尝试，OCC 生命周期内绝不逐帧改向。
            if (!gmc_enabled_ && occluder_tracker_id_ == trk->get_id()
                && occlusion_state_ != OcclusionState::CLEAR
                && emergence_dir_x_ == 0.f && emergence_dir_y_ == 0.f) {
                int samples = 0;
                float consistency = 0.f;
                float dx = 0.f, dy = 0.f;
                const bool valid = estimate_relative_emergence_direction(
                    trk->get_id(), dx, dy, &samples, &consistency);
                if (valid) {
                    emergence_dir_x_ = dx;
                    emergence_dir_y_ = dy;
                }
                if (kMatchTrace) {
                    char line[kTraceLineLen];
                    std::snprintf(line, sizeof(line),
                        "[EMERGE] f=%d other=%d source=%s permission=search_hold"
                        " samples=%d consistency=%.2f dir=(%.3f,%.3f)",
                        frame_count, trk->get_id(),
                        valid ? "relative_history" : "none",
                        samples, consistency, dx, dy);
                    trace_push(line);
                }
            }
        }
        // 共存确认：主目标本帧真实匹配 + 该轨迹同帧匹配到自己的检测
        // → 两人同时在场。只在"有自己检测"时累计；隔离轨迹不积累（B3：
        // 否决权须先转正）。主目标的影子轨迹（检测被主目标占用→永远关联
        // 不到）不会积累计数。
        if (best.matched && !trk->quarantined_ && trk->coexist_with_main < 100000)
            trk->coexist_with_main++;
    }

    // 未命中的非主轨迹：空更新（tsu 自然增长，超龄由 cleanup 清理）
    for (int i = 0; i < (int)sec_idx.size(); ++i) {
        if (!trk_hit[i]) {
            std::optional<cv::Mat> none;
            trackers[sec_idx[i]]->update(none);
        }
    }

    // ── 新建轨迹（仅高置信检测，数量受限）──
    int live = (int)sec_idx.size();
    for (int j = 0; j < (int)dets.size() && live < kSecTrkMax; ++j) {
        if (det_used[j] || !dets[j].high) continue;
        // B3 改造：原"防影子护栏"直接拒建 → 恰好把正面逼近的遮挡者也拒之门外
        // （遮挡者常在与主框强重叠时才首次出现检测）→ FSM onset 时
        // occluder_tracker_id_=-1、anti_occ 死、共存排除盲。现改为：照建，但戴
        // "隔离"标记 —— 轨迹存在（遮挡者识别/anti_occ 可见），否决权剥夺
        // （不积累 coexist、不作排除源、不输出、不占 ReID 预算），转正须自证
        // （见上方转正逻辑）。影子轨迹拿不到自有检测 → 永远转不了正 → 自然老死。
        bool quarantine_new = !main_box.empty()
            && Utils::iou_single(main_box, dets[j].box4) > kSecTrkNewMainIou;
        // id 回收：防止长时间运行后 tracker id 漫入 900+（add_other_det 展示区）。
        // A5 修复：回收基准 = 存活轨迹最大 id + 1（旧 set_count(1) 会与存活轨迹
        // id 冲撞，极端时新轨迹撞上 main_id → 输出行 id == main_id+1 → 云台拿错主框）。
        if (KalmanBoxTracker::count >= 800) {
            int mx = 0;
            for (auto& t : trackers) mx = std::max(mx, t->get_id());
            KalmanBoxTracker::set_count(mx + 1);
        }
        cv::Mat box5(1, 5, CV_32F);
        dets[j].box4.copyTo(box5.colRange(0, 4));
        box5.at<float>(0, 4) = dets[j].score;
        trackers.push_back(std::make_shared<KalmanBoxTracker>(
            box5, delta_t, cv::Mat(), false));
        trackers.back()->last_det_score_ = dets[j].score;   // 新轨迹：注册优先级用
        if (quarantine_new) {
            trackers.back()->quarantined_ = true;
            null_sink << "[QUARANTINE] new id=" << trackers.back()->get_id()
                         << " (born overlapping main)" << std::endl;
        }
        live++;
    }
}


// [N14] compute_color_hist / color_hist_sim 已删除（颜色直方图整体弃用，无调用点）。

// ── 非主轨迹外观特征刷新（见 .h 说明）────────────────────────────────
//   RTMPose 预算只服务主目标；非主轨迹仅按原预算刷新 ReID。
//   须在 update_secondary_tracks 之后调用（命中的非主轨迹此时 tsu==0、last_observation 为本帧检测框）。
void LightTracker::update_secondary_features(const DetectionGroups& det_groups, const cv::Mat& img) {
    if (img.empty()) return;
    (void)det_groups;   // 特征取自轨迹本帧观测（last_observation），无需再关联检测

    int   reg_pick = -1;  float reg_score = -1.f;   // 注册组：挑最高检测分
    int   ref_pick = -1;  int   ref_age   = -1;     // 刷新组：挑最陈旧

    for (int t = 0; t < (int)trackers.size(); ++t) {
        auto& trk = trackers[t];
        if (trk->get_is_main()) continue;
        if (trk->get_time_since_update() != 0) continue;    // 仅本帧真实命中的非主轨迹
        cv::Mat box = trk->get_last_observation();           // 本帧关联检测框（[1,4] xyxy）
        if (box.empty() || box.cols < 4
            || (box.at<float>(0, 2) - box.at<float>(0, 0)) < 4.f) continue;   // 框宽 = x2-x1

        // ReID 轮询候选登记：注册组（从未算过嵌入）优先，其次陈旧刷新组。
        //     B3：隔离轨迹不占 ReID 预算（疑影子，其裁剪≈主目标本人 → 白花推理
        //     且污染源；转正后自然进入轮询）。免费的骨骼/颜色仍照常刷新。
        if (trk->quarantined_) continue;
        if (trk->get_emb().empty()) {
            if (trk->last_det_score_ > reg_score) { reg_score = trk->last_det_score_; reg_pick = t; }
        } else {
            int age = frame_count - trk->get_emb_update_frame();
            if (age >= kSecEmbRefreshFrames && age > ref_age) { ref_age = age; ref_pick = t; }
        }
    }

    // (d) ReID 预算（每帧至多 kSecReidBudgetPerFrame，当前=1）：注册优先，其次刷新。
    //     [N12] 过载降级：dt 持续过高 → 二级 ReID 轮询整帧停（0 预算），把 NPU 让给主目标热路径。
    int budget = (overload_mode_ || !frame_allow_secondary_reid_)
               ? 0 : kSecReidBudgetPerFrame;
    auto embed_one = [&](int t) {
        if (t < 0 || budget <= 0) return;
        auto& trk = trackers[t];
        cv::Mat feat = compute_embedding(img, trk->get_last_observation());
        if (feat.empty()) return;
        // compute_embedding 返回已归一化特征；首样本直接置入（update_emb 对空 emb 为 clone），否则 EMA
        trk->update_emb(feat, trk->get_emb().empty() ? 1.0f : 0.5f);
        trk->set_emb_update_frame(frame_count);
        budget--;
    };
    embed_one(reg_pick);
    if (ref_pick != reg_pick) embed_one(ref_pick);
}

void LightTracker::update_trackers_unified(
    const cv::Mat& matches_one,
    const cv::Mat& matches_second,
    const std::vector<int>& unmatched_trks,
    const cv::Mat& dets_one,
    const cv::Mat& dets_second,
    const std::vector<int>& picked_idx,  // 非空 = 允许更新 emb
    const MainMatchResult& match_result,
    const cv::Mat& img) 
{
    // 所有 OCC 身份/空间信任判断已在构造 matches 之前的最终 COMMIT_GATE 完成。
    // 因此进入本函数的 match 必须真实更新 KF；这里不得再次把 matched 降级成
    // update(empty)，否则会重新制造 MATCH=1 / OUTPUT=NONE 的分裂语义。
    auto main_kf_update = [&](int trk_idx, const cv::Mat& det_box) {
        std::optional<cv::Mat> bbox = det_box;
        trackers[trk_idx]->update(bbox);
    };

    // 更新 KF 状态（位置）—— matches 仅包含最终已提交观测。
    for (int i = 0; i < matches_one.rows; ++i)
        main_kf_update(matches_one.at<int>(i, 0),
                       dets_one.row(matches_one.at<int>(i, 1)).colRange(0, 4));

    for (int i = 0; i < matches_second.rows; ++i)
        main_kf_update(matches_second.at<int>(i, 0),
                       dets_second.row(matches_second.at<int>(i, 1)).colRange(0, 4));

    // 只有走到这里才表示 BODY observation 已真正提交给主 KF。
    // continuation 身份成立与 measurement commit 分层，避免暂定期产生副作用。
    if (match_result.matched && match_result.provisional_continuation
        && pending_body_hyp_id_ >= 0) {
        const cv::Mat& committed_src = match_result.source == 0
                                     ? dets_one : dets_second;
        float center_dist = -1.f, box_iou = -1.f, size_ratio = -1.f;
        if ((match_result.source == 0 || match_result.source == 1)
            && match_result.index >= 0 && match_result.index < committed_src.rows) {
            body_provisional_geometry(
                committed_src.row(match_result.index).colRange(0, 4),
                center_dist, box_iou, size_ratio);
        }
        trace_body_provisional(
            "commit", match_result.body_hyp_id, pending_body_hyp_id_,
            match_result.source, match_result.index,
            match_result.reid_sim, match_result.anchor_sim,
            pending_body_reid_, pending_body_anchor_,
            reacq_defer_count_, kReacqMaxDefer,
            center_dist, box_iou, size_ratio,
            "measurement_reliable", "kf_update");
        clear_body_provisional(nullptr, nullptr);
        reacq_defer_count_ = 0;
        pending_active_ = false;
        pending_src_ = pending_idx_ = -1;
        pending_cx_ = pending_cy_ = -1.f;
    } else if (match_result.matched && pending_body_hyp_id_ >= 0) {
        clear_body_provisional(
            match_result.body_hyp_id == pending_body_hyp_id_
                ? "successful_commit" : "hyp_changed",
            "kf_update", match_result.source, match_result.index,
            match_result.reid_sim, match_result.anchor_sim,
            -1.f, -1.f, -1.f, match_result.body_hyp_id);
        reacq_defer_count_ = 0;
        pending_active_ = false;
        pending_src_ = pending_idx_ = -1;
        pending_cx_ = pending_cy_ = -1.f;
    }

    for (int trk_idx : unmatched_trks) {
        std::optional<cv::Mat> noting;
        trackers[trk_idx]->update(noting);
    }

    // ====== Embedding 更新：只在 picked_idx 非空时执行 ======
    if (!picked_idx.empty() && match_result.matched) {
        // 动态定位主目标 tracker
        int main_trk_idx = -1;
        if (matches_one.rows > 0)
            main_trk_idx = matches_one.at<int>(0, 0);
        else if (matches_second.rows > 0)
            main_trk_idx = matches_second.at<int>(0, 0);

        if (main_trk_idx >= 0 && main_trk_idx < (int)trackers.size()) {
            // 仅复用候选评分/人脸确认已得到的特征。人脸覆盖到未入 ReID 预算的
            // 候选时，本帧不补跑 ReID；身份由人脸锁保证，embedding 延后下一常规帧。
            cv::Mat new_emb = match_result.emb;
            if (new_emb.empty()) {
                null_sink << "[EMB_UPDATE] deferred (frame ReID budget)" << std::endl;
            } else {
                // 渐进更新：不直接替换，用 EMA 混合
                float alpha = 0.10f;
                if (match_result.anchor_sim > 0.75f) alpha = 0.15f;
                if (match_result.from_face) alpha = 0.30f;

                trackers[main_trk_idx]->update_emb(new_emb, alpha);

                if (match_result.from_face)
                    trackers[main_trk_idx]->set_confirmed_emb(new_emb);

                null_sink << "[EMB_UPDATE] reid=" << match_result.reid_sim
                          << " anchor=" << match_result.anchor_sim
                          << " alpha=" << alpha << std::endl;
            }
        }
    }

}


ProximityInfo LightTracker::collect_nearby_dets(
    const cv::Mat& main_trk,
    const cv::Mat& dets_one,
    const cv::Mat& dets_second)
{
    ProximityInfo result;
    result.close_det_count = 0;
    result.overlap_count   = 0;

    if (main_trk.empty() || main_trk.rows == 0) {
        result.match_candidates = cv::Mat::zeros(0, 6, CV_32F);
        result.all_candidates = cv::Mat::zeros(0, 6, CV_32F);
        return result;
    }

    if (dets_one.rows == 0 && dets_second.rows == 0) {
        result.match_candidates = cv::Mat::zeros(0, 6, CV_32F);
        result.all_candidates = cv::Mat::zeros(0, 6, CV_32F);
        return result;
    }

    const bool ptz_anchor_search = spatial_prior_invalid_
                                && !ptz_blind_anchor_box_.empty();
    const cv::Mat main_box = ptz_anchor_search
                           ? ptz_blind_anchor_box_.row(0)
                           : main_trk.row(0);
    float main_w  = main_box.at<float>(2) - main_box.at<float>(0);
    float main_h  = main_box.at<float>(3) - main_box.at<float>(1);
    float main_cx = main_box.at<float>(0) + main_w * 0.5f;
    float main_cy = main_box.at<float>(1) + main_h * 0.5f;

    // ════════════════════════════════════════════
    // 两个半径
    // ════════════════════════════════════════════

    // 紧邻半径：用于判断"有人贴着主目标" → 冻结 embedding
    float close_radius = 0.7f * std::sqrt(main_w * main_w + main_h * main_h);

    // 匹配半径：根据丢失帧数动态扩大
    int frames_lost = 0;
    for (auto& trk : trackers) {
        if (trk->get_is_main()) {
            frames_lost = trk->get_time_since_update();
            break;
        }
    }

    // ── 丢失期：把搜索门中心移到预测器引导中心（而非冻结的 KF 框中心）──
    // KF 的有限速度衰减会使长期纯预测逐步回归消失点附近；若以此为圆心扩圈，
    // 会在人已离开的旧位置搜索（"在全新位置找人"）。改用引导中心 →
    // 搜索门跟随轨迹移动到"即将出现的位置"。
    // 注意：仅改距离中心 main_cx/main_cy；overlap_count 仍用真实 KF 框算 IoU，
    // 不受影响（避免引导中心干扰遮挡触发）。
    if (!ptz_anchor_search && frames_lost > 0 && lead_cx_ >= 0.f && lead_cy_ >= 0.f) {
        main_cx = lead_cx_;
        main_cy = lead_cy_;
        null_sink << "[REACQ] lead-gate center=(" << main_cx << "," << main_cy
                  << ") frames_lost=" << frames_lost << std::endl;
    }

    float match_radius;
    if (frames_lost <= 3) {
        // 短期丢失：搜索范围 = 2倍框对角线
        match_radius = 4.0f * std::sqrt(main_w * main_w + main_h * main_h);
    } else if (frames_lost <= 10) {
        // 中期：扩大到 4 倍
        match_radius = 6.0f * std::sqrt(main_w * main_w + main_h * main_h);
    } else {
        // 长期：搜全图（用一个极大值）
        match_radius = std::sqrt((float)(img_w * img_w + img_h * img_h));
    }

    // ════════════════════════════════════════════
    // 收集所有检测框，计算距离
    // ════════════════════════════════════════════
    struct DetInfo {
        cv::Mat box;   // 4列
        float   dist;
        int     source;
        int     index;
    };
    std::vector<DetInfo> all_dets;

    auto collect = [&](const cv::Mat& dets, int source_id) {
        for (int i = 0; i < dets.rows; ++i) {
            cv::Mat det = dets.row(i).colRange(0, 4);
            float dw  = det.at<float>(2) - det.at<float>(0);
            float dh  = det.at<float>(3) - det.at<float>(1);
            float dcx = det.at<float>(0) + dw * 0.5f;
            float dcy = det.at<float>(1) + dh * 0.5f;
            float dist = std::sqrt((dcx - main_cx) * (dcx - main_cx) 
                                 + (dcy - main_cy) * (dcy - main_cy));
            all_dets.push_back({det.clone(), dist, source_id, i});
        }
    };

    if (!dets_one.empty())    collect(dets_one, 0);
    if (!dets_second.empty()) collect(dets_second, 1);

    // 按距离排序
    std::sort(all_dets.begin(), all_dets.end(),
              [](const DetInfo& a, const DetInfo& b) { return a.dist < b.dist; });

    // 全图候选不受动态搜索半径和 top-N 限制。这里只保存轻量 box/index，
    // 真正的 ReID 仍由后续固定 3/4 槽预算调度，不增加 Detector 或 ReID 峰值调用数。
    result.all_candidates = cv::Mat::zeros((int)all_dets.size(), 6, CV_32F);
    for (int i = 0; i < (int)all_dets.size(); ++i) {
        all_dets[i].box.copyTo(result.all_candidates.row(i).colRange(0, 4));
        result.all_candidates.at<float>(i, 4) = (float)all_dets[i].source;
        result.all_candidates.at<float>(i, 5) = (float)all_dets[i].index;
    }

    // ════════════════════════════════════════════
    // 分两路输出
    // ════════════════════════════════════════════
    
    // (1) close_det_count：在紧邻半径内的检测框数量（距离判定）
    for (auto& d : all_dets) {
        if (d.dist <= close_radius)
            result.close_det_count++;
        else
            break;  // 已排序，后面更远
    }

    // (1b) overlap_count：与主目标检测框有 IoU 重叠的数量（真正遮挡）
    // 区别于 close_det_count：
    //   - close = 离得近（人群中很多人都近）
    //   - overlap = 框重叠（只有真正遮挡/交错才会框重叠）
    const float overlap_iou_thresh = 0.25f;  // IoU > 25% 视为真正遮挡
    for (auto& d : all_dets) {
        float iou = Utils::iou_single(main_box, d.box);
        if (iou > overlap_iou_thresh)
            result.overlap_count++;
    }

    // (2) match_candidates：在匹配半径内的，最多取 top_n 个
    int top_n = 5;
    std::vector<DetInfo> match_pool;
    for (auto& d : all_dets) {
        if (d.dist <= match_radius)
            match_pool.push_back(d);
    }
    int actual_n = std::min((int)match_pool.size(), top_n);
    
    result.match_candidates = cv::Mat::zeros(actual_n, 6, CV_32F);
    for (int i = 0; i < actual_n; ++i) {
        match_pool[i].box.copyTo(result.match_candidates.row(i).colRange(0, 4));
        result.match_candidates.at<float>(i, 4) = (float)match_pool[i].source;
        result.match_candidates.at<float>(i, 5) = (float)match_pool[i].index;
    }

    null_sink << "[PROXIMITY] close=" << result.close_det_count
              << " overlap=" << result.overlap_count
              << " match_candidates=" << actual_n
              << " all_candidates=" << result.all_candidates.rows
              << " close_r=" << close_radius
              << " match_r=" << match_radius
              << " frames_lost=" << frames_lost << std::endl;
    if (ptz_anchor_search && kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[PTZ_RECOVERY] f=%d stage=candidate_admission anchor=(%.1f,%.1f)"
            " local=%d all=%d radius=%.1f",
            frame_count, main_cx, main_cy, actual_n, result.all_candidates.rows,
            match_radius);
        trace_push(line);
    }

    return result;
}

void LightTracker::update_smooth_center(const cv::Mat& trk_box) {
    float cx = (trk_box.at<float>(0) + trk_box.at<float>(2)) * 0.5f;
    float cy = (trk_box.at<float>(1) + trk_box.at<float>(3)) * 0.5f;
    
    if (smooth_cx_ < 0) {
        smooth_cx_ = cx;
        smooth_cy_ = cy;
    } else {
        smooth_cx_ = (1.f - center_ema_alpha_) * smooth_cx_ + center_ema_alpha_ * cx;
        smooth_cy_ = (1.f - center_ema_alpha_) * smooth_cy_ + center_ema_alpha_ * cy;
    }
}

void LightTracker::update_quality_monitor(const MainMatchResult& result,
                                          int close_det_count) {
    if (!result.matched) return;

    MatchQualityRecord rec;
    rec.reid_sim     = result.reid_sim;
    rec.anchor_sim   = result.anchor_sim;
    rec.total_score  = result.total_score;
    rec.from_face    = result.from_face;
    rec.frame_id     = frame_count;

    quality_history_.push_back(rec);
    if ((int)quality_history_.size() > kQualityWindowSize)
        quality_history_.pop_front();

    // ── 建立基线（前10帧稳定匹配的平均值）──
    if (stable_frame_count_ < 10 && result.anchor_sim > 0.65f) {
        stable_frame_count_++;
        float w = 1.0f / stable_frame_count_;
        baseline_reid_ = baseline_reid_ * (1.f - w) + result.reid_sim * w;
        baseline_anchor_sim_ = baseline_anchor_sim_ * (1.f - w) + result.anchor_sim * w;
        return;
    }

    if (stable_frame_count_ < 10) return;

    // ── 异常检测 ──
    // 信号1：anchor_sim 显著低于基线
    bool anchor_drop = (result.anchor_sim < baseline_anchor_sim_ - 0.12f)
                    && (result.anchor_sim < 0.60f);

    // 信号2：reid_sim 高但 anchor_sim 低 → embedding 被污染
    bool emb_divergence = (result.reid_sim > 0.70f)
                       && (result.anchor_sim < 0.50f)
                       && (result.reid_sim - result.anchor_sim > 0.20f);

    // 信号3：滑动窗口内 anchor_sim 持续走低
    bool trend_decline = false;
    if ((int)quality_history_.size() >= 10) {
        float recent_avg = 0.f, old_avg = 0.f;
        int n = (int)quality_history_.size();
        for (int i = n - 5; i < n; ++i)
            recent_avg += quality_history_[i].anchor_sim;
        recent_avg /= 5.0f;
        for (int i = 0; i < 5; ++i)
            old_avg += quality_history_[i].anchor_sim;
        old_avg /= 5.0f;
        if (old_avg - recent_avg > 0.10f)
            trend_decline = true;
    }

    bool suspect = anchor_drop || emb_divergence || trend_decline;

    // ── 孤立情境豁免（防"转身/走远"误报 ID switch）──
    // 三个嫌疑信号全部基于"外观偏离锚点"，但目标转身/走远时这正是预期现象；
    // 而真正的 ID switch 必须有第二个人在场（交错/遮挡中发生）。CLEAR + 无邻人
    // → 外观下滑判为视角/距离变化，不累计嫌疑（否则 3 帧后触发 alert，把 emb
    // 回滚到正面 anchor，反而加速丢失）。有邻人/危险期的监控灵敏度不变。
    if (suspect && occlusion_state_ == OcclusionState::CLEAR && close_det_count <= 1) {
        null_sink << "[SUSPECT-WAIVED] isolated view/distance change (anchor="
                  << result.anchor_sim << " reid=" << result.reid_sim << ")" << std::endl;
        suspect = false;
    }

    if (suspect) {
        suspect_streak_++;
        null_sink << "[SUSPECT] streak=" << suspect_streak_
                  << " anchor=" << result.anchor_sim
                  << " reid=" << result.reid_sim
                  << " baseline_anchor=" << baseline_anchor_sim_
                  << " drop=" << anchor_drop
                  << " diverge=" << emb_divergence
                  << " trend=" << trend_decline << std::endl;
    } else {
        if (suspect_streak_ > 0) suspect_streak_--;
    }

    // ── 触发 alert ──
    if (suspect_streak_ >= kSuspectThresh && !id_switch_alert_) {
        id_switch_alert_ = true;
        alert_frame_start_ = frame_count;
        alert_start_ms_ = now_ms();   // B6：超时改墙钟

        // 回滚 embedding 到 anchor
        for (auto& trk : trackers) {
            if (trk->get_is_main() && !trk->get_anchor_emb().empty()) {
                trk->update_emb(trk->get_anchor_emb(), 1.0f);
                null_sink << "[ALERT_TRIGGERED] embedding rolled back to anchor" << std::endl;
                break;
            }
        }
    }
}

LightTracker::PoseCacheEntry* LightTracker::find_cached_pose(int source, int index) {
    for (auto& entry : pose_cache_)
        if (entry.source == source && entry.index == index) return &entry;
    return nullptr;
}

LightTracker::PoseCacheEntry* LightTracker::request_pose(
    const cv::Mat& img, const cv::Mat& body_box, int source, int index,
    PoseReason reason) {
    if (PoseCacheEntry* cached = find_cached_pose(source, index)) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[POSE_REQ] f=%d reason=%s src=%d idx=%d status=CACHE cached=%d used=%d",
                frame_count, reason == PoseReason::AMBIGUITY_RESOLVE ? "AMBIG" : "REFRESH",
                source, index, (int)cached->status, pose_budget_used_);
            trace_push(line);
        }
        return cached;
    }
    PoseCacheEntry entry;
    entry.source = source;
    entry.index = index;
    entry.body_box = body_box.clone();
    if (pose_budget_used_ >= kPoseBudgetPerFrame) {
        entry.status = PoseRequestStatus::BUDGET_DENIED;
        pose_cache_.push_back(entry);
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[POSE_REQ] f=%d reason=%s src=%d idx=%d status=BUDGET used=%d",
                frame_count, reason == PoseReason::AMBIGUITY_RESOLVE ? "AMBIG" : "REFRESH",
                source, index, pose_budget_used_);
            trace_push(line);
        }
        return &pose_cache_.back();
    }
    ++pose_budget_used_;
    const int64_t start_ms = now_ms();
    const int ret = pose_estimator.run(img, body_box, entry.pose);
    entry.status = ret == 0 ? PoseRequestStatus::AVAILABLE : PoseRequestStatus::FAILED;
    pose_cache_.push_back(entry);
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[POSE_RUN] f=%d reason=%s src=%d idx=%d status=%s used=%d latency=%lldms kp=%d",
            frame_count, reason == PoseReason::AMBIGUITY_RESOLVE ? "AMBIG" : "REFRESH",
            source, index, ret == 0 ? "OK" : "FAIL", pose_budget_used_,
            (long long)(now_ms() - start_ms), entry.pose.valid_kp_count);
        trace_push(line);
    }
    return &pose_cache_.back();
}

void LightTracker::commit_pose(const PoseResult& pose, const cv::Mat& body_box,
                               const cv::Mat& img, const char* reason) {
    KalmanBoxTracker* main_trk = nullptr;
    for (auto& trk : trackers) if (trk->get_is_main()) { main_trk = trk.get(); break; }
    if (!main_trk || pose.valid_kp_count < 5) return;
    main_trk->set_keypoints(pose.keypoints);
    BodyProportionDescriptor bp = PoseMatch::extract_body_proportions(pose.keypoints, 0.4f);
    if (bp.valid) main_trk->set_body_proportions(bp);
    committed_pose_box_ = body_box.clone();
    last_committed_pose_frame_ = frame_count;
    last_committed_pose_ms_ = now_ms();
    assess_visibility(body_box, &pose, img.cols, img.rows);
    update_orientation_state(&pose, body_box);
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[POSE_COMMIT] f=%d reason=%s kp=%d vis=%d ratio=%.2f",
            frame_count, reason, pose.valid_kp_count,
            (int)visibility_state_, visible_ratio_ema_);
        trace_push(line);
    }
}

// ════════════════════════════════════════════════════════════
// 主目标身体可见度评估
//   关键点深度表（从头到脚递增）：可见到的最深部位 ≈ 露出多少躯干。
//   手肘/手腕排除（手臂会上下摆动，深度不稳定，不能代表躯干露出度）。
//   pose 缺失/结构关键点不足时退回几何（框宽高比）估计。
//   EMA 平滑 + 迟滞，避免状态在边界抖动。
// ════════════════════════════════════════════════════════════
LightTracker::VisibilityState LightTracker::assess_visibility(
    const cv::Mat& main_box,
    const PoseResult* pose,
    int frame_w, int frame_h)
{
    // main_box: [1,4] xyxy
    const float x1 = main_box.at<float>(0, 0);
    const float y1 = main_box.at<float>(0, 1);
    const float x2 = main_box.at<float>(0, 2);
    const float y2 = main_box.at<float>(0, 3);
    const float bw = std::max(1.f, x2 - x1);
    const float bh = std::max(1.f, y2 - y1);
    const float box_aspect = bh / bw;  // 站立全身 ~2.5-3.5；半身 ~1.2-1.5；上身 ~1.0
    (void)y1;  // 顶部坐标暂未使用（保留扩展：trunc_top）

    // ── 边缘截断判定（底部截断 → 看不到下半身/脚）──
    const float my = std::max(4.f, 0.01f * frame_h);
    const bool trunc_bottom = (y2 >= frame_h - my);
    // （trunc_top/left/right 暂未使用，保留扩展空间）

    // ── 关键点深度表（越往下越大；肘/腕返回 -1 表示排除）──
    auto kp_depth = [](int idx) -> float {
        switch (idx) {
            case NOSE: case LEFT_EYE: case RIGHT_EYE:
            case LEFT_EAR: case RIGHT_EAR:           return 0.10f;  // 头
            case LEFT_SHOULDER: case RIGHT_SHOULDER: return 0.28f;  // 肩
            case LEFT_HIP: case RIGHT_HIP:           return 0.52f;  // 髋
            case LEFT_KNEE: case RIGHT_KNEE:         return 0.78f;  // 膝
            case LEFT_ANKLE: case RIGHT_ANKLE:       return 1.00f;  // 踝
            default:                                 return -1.f;   // 肘/腕：排除
        }
    };

    // ── 瞬时可见比例（关键点优先）──
    float inst_ratio     = -1.f;
    int   struct_kp_count = 0;
    if (pose) {
        const float kConf = 0.35f;
        float max_depth = -1.f;
        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float d = kp_depth(k);
            if (d < 0.f) continue;                          // 排除肘/腕
            if (pose->keypoints[k].confidence < kConf) continue;
            ++struct_kp_count;
            if (d > max_depth) max_depth = d;
        }
        if (struct_kp_count >= 2 && max_depth > 0.f)
            inst_ratio = max_depth;
    }

    // ── 几何退回（pose 缺失或结构关键点不足）──
    if (inst_ratio < 0.f) {
        // 线性：aspect 2.8→1.0, 1.0→~0.18, ≤0.6→0.1（下钳）
        inst_ratio = (box_aspect - 0.6f) / 2.2f;
        inst_ratio = std::min(1.0f, std::max(0.1f, inst_ratio));
    }

    // ── 远处全身修正 ──
    // 框够高且底部未截断 → 人完整在画面内（脚也在），即使 pose 漏检下肢
    // （远/小目标关键点置信度低）。避免把"远处全身"误判为半身。
    if (!trunc_bottom && box_aspect >= 2.2f)
        inst_ratio = std::max(inst_ratio, 0.88f);

    // ── EMA 平滑 ──
    visible_ratio_ema_ = (1.f - kVisEmaAlpha) * visible_ratio_ema_
                       + kVisEmaAlpha * inst_ratio;

    // ── 量化为候选状态 ──
    auto quantize = [](float r) -> VisibilityState {
        if (r >= 0.85f) return VisibilityState::FULL;
        if (r >= 0.60f) return VisibilityState::MOSTLY_FULL;
        if (r >= 0.40f) return VisibilityState::HALF;
        if (r >= 0.20f) return VisibilityState::UPPER;
        return VisibilityState::HEAD_ONLY;
    };
    VisibilityState inst_state = quantize(visible_ratio_ema_);

    // ── [N7] 迟滞（需稳定 ≥ kVisHysteresisMs 墙钟才切换）──
    //   原帧计数在 27~374ms 帧间隔波动下语义漂移严重（快帧下迟滞过短、慢帧下过长）→ 改墙钟。
    const int64_t vis_now = now_ms();
    if (inst_state != visibility_state_) {
        if (inst_state == visibility_pending_) {
            if (vis_now - visibility_pending_since_ms_ >= kVisHysteresisMs) {
                static const char* kVisName[] = {
                    "FULL", "MOSTLY_FULL", "HALF", "UPPER", "HEAD_ONLY" };
                null_sink << "[VIS] " << kVisName[(int)visibility_state_]
                          << " -> " << kVisName[(int)inst_state]
                          << " (ratio=" << visible_ratio_ema_
                          << ", kp=" << struct_kp_count
                          << ", aspect=" << box_aspect << ")" << std::endl;
                visibility_state_   = inst_state;
                visibility_pending_ = inst_state;
            }
        } else {
            visibility_pending_          = inst_state;   // 新 pending → 重置墙钟起点
            visibility_pending_since_ms_ = vis_now;
        }
    } else {
        visibility_pending_          = visibility_state_;
        visibility_pending_since_ms_ = vis_now;
    }

    return visibility_state_;
}

// [N14] compute_candidate_pose_score 已删除（未被引用；compute_candidate_pose_detail 覆盖其功能）。

// ============================================================
// 分离版 Pose 打分：返回 OKS（位置）和 body_shape（身份）
// ============================================================
PoseScoreDetail LightTracker::compute_candidate_pose_detail(
    const PoseResult& cand_pose, const cv::Mat& candidate_bbox,
    const cv::Mat& predicted_box)
{
    PoseScoreDetail result;

    // 找主目标 tracker
    KalmanBoxTracker* main_trk = nullptr;
    for (auto& trk : trackers) {
        if (trk->get_is_main()) {
            main_trk = trk.get();
            break;
        }
    }
    if (!main_trk || !main_trk->get_has_pose()) return result;
    if (cand_pose.valid_kp_count < 5) return result;

    const float area = std::max(1.f,
        (candidate_bbox.at<float>(0, 2) - candidate_bbox.at<float>(0, 0))
      * (candidate_bbox.at<float>(0, 3) - candidate_bbox.at<float>(0, 1)));

    PoseKeypoint mapped_history[NUM_KEYPOINTS];
    std::memcpy(mapped_history, main_trk->get_keypoints(), sizeof(mapped_history));
    if (!committed_pose_box_.empty() && !predicted_box.empty()) {
        const float hx1 = committed_pose_box_.at<float>(0, 0);
        const float hy1 = committed_pose_box_.at<float>(0, 1);
        const float hw = std::max(1.f, committed_pose_box_.at<float>(0, 2) - hx1);
        const float hh = std::max(1.f, committed_pose_box_.at<float>(0, 3) - hy1);
        const float px1 = predicted_box.at<float>(0, 0);
        const float py1 = predicted_box.at<float>(0, 1);
        const float pw = std::max(1.f, predicted_box.at<float>(0, 2) - px1);
        const float ph = std::max(1.f, predicted_box.at<float>(0, 3) - py1);
        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            mapped_history[k].x = px1 + (mapped_history[k].x - hx1) * pw / hw;
            mapped_history[k].y = py1 + (mapped_history[k].y - hy1) * ph / hh;
        }
    }
    static constexpr int64_t kPoseOksMaxAgeMs = 600;
    const int64_t pose_age_ms = last_committed_pose_ms_ < 0
                              ? std::numeric_limits<int64_t>::max()
                              : now_ms() - last_committed_pose_ms_;
    if (pose_age_ms <= kPoseOksMaxAgeMs) {
        result.oks = PoseMatch::compute_oks(
            mapped_history, cand_pose.keypoints, area, 0.4f);
        result.has_oks = result.oks > 0.f;
        if (result.has_oks)
            result.oks *= std::max(0.f, 1.f - (float)pose_age_ms / kPoseOksMaxAgeMs);
    }

    // Body shape：骨骼比例余弦相似度（身份指纹）
    const BodyProportionDescriptor& anchor_bp = main_trk->get_anchor_body_proportions();
    {
        BodyProportionDescriptor cand_bp = PoseMatch::extract_body_proportions(
            cand_pose.keypoints, 0.4f);
        float bs = -1.f;
        // 优先全身骨骼比例（信息更全）；全身不可用（如桌遮挡髋不可见）→ 退上半身弱判别。
        if (anchor_bp.valid && cand_bp.valid)
            bs = PoseMatch::body_proportion_similarity(anchor_bp, cand_bp);
        if (bs < 0.f)
            bs = PoseMatch::upper_body_similarity(anchor_bp, cand_bp);
        if (bs >= 0.f) {              // -1 = 无可比维 → 不提供 body_shape 信号
            result.body_shape = bs;
            result.has_shape   = true;
        }
    }

    // 肩中点（供转身期肩部连续性信号；免费，复用已匹配的 cand_pose）
    {
        cv::Point2f smid; float sw = 0.f;
        if (extract_shoulder_geom(cand_pose.keypoints, 0.4f, smid, sw)) {
            result.shoulder_mid = smid;
            result.shoulder_w   = sw;
            result.has_shoulder = true;
        }
    }

    return result;
}


void LightTracker::reset() {
    clear_body_provisional("reset_target", "reset");
    ptz_blind_phase_ = PtzBlindPhase::IDLE;
    spatial_prior_invalid_ = false;
    ptz_blind_anchor_box_.release();
    ptz_blind_anchor_ms_ = -1;
    ptz_blind_explore_rotor_ = 0;
    ptz_reacq_body_streak_ = 0;
    KalmanBoxTracker::set_count(0);

    trackers.clear();
    frame_count = 0;
    main_track_unmatched_time = 0;
    main_id = -1;
    reset_flag = true;
    face_recognizer.reset();

    // 重置平滑中心
    smooth_cx_ = -1.f;
    smooth_cy_ = -1.f;

    // 重置丢失期引导中心 + 浮现点 + 盲跟时钟（避免跨目标残留）
    lead_cx_ = -1.f;
    lead_cy_ = -1.f;
    body_reid_hypotheses_.clear();
    next_body_reid_hyp_id_ = 1;
    body_reid_global_active_ = false;
    body_reid_scan_epoch_ = 0;
    body_reid_scan_epoch_frame_ = -1;
    body_reid_anchor_cx_ = body_reid_anchor_cy_ = -1.f;
    body_reid_anchor_diag_ = -1.f;
    body_reid_anchor_ms_ = -1;
    emergence_valid_ = false;
    emergence_cx_ = -1.f; emergence_cy_ = -1.f;
    emergence_dir_x_ = 0.f; emergence_dir_y_ = 0.f;
    emergence_update_ms_ = -1;   // [N8] 浮现点新鲜度时钟
    coast_weight_ = 0.0f;
    frame_output_source_ = OutputSource::NONE;
    clear_short_prediction("tracker_reset");
    clear_motion_history();
    secondary_frame_observations_.clear();
    relative_motion_history_.clear();
    pred_missing_real_frames_ = 0;
    pred_move_high_frames_ = 0;
    pred_move_medium_frames_ = 0;
    pred_hold_low_frames_ = 0;
    pred_output_none_frames_ = 0;
    frame_measurement_reliability_ = MeasurementReliability::NONE;
    part_output_box_valid_ = false;
    part_output_cx_ = part_output_cy_ = -1.f;
    part_output_w_ = part_output_h_ = -1.f;
    part_output_update_ms_ = -1;
    part_output_pending_since_ms_ = -1;
    last_returned_x1_ = last_returned_y1_ = -1.f;
    last_returned_x2_ = last_returned_y2_ = -1.f;
    last_returned_box_ms_ = -1;
    last_returned_source_ = OutputSource::NONE;
    closeup_output_active_ = false;
    closeup_top_y_ = -1.f;
    closeup_last_head_cy_ = -1.f;
    closeup_update_ms_ = -1;
    closeup_last_head_ms_ = -1;
    closeup_last_face_cy_ = -1.f;
    closeup_last_face_ms_ = -1;
    closeup_anchor_kind_ = 0;
    last_real_obs_ms_ = -1;      // [N1] 换目标 → 无真实观测（盲时长从头计）
    last_body_observation_ms_ = -1;
    last_head_continuity_ms_ = -1;
    last_face_identity_ms_ = -1;
    last_body_face_identity_ms_ = -1;

    // 重置头部连续性几何先验与独立头部缓存（目标切换 → 几何失效）
    head_body_geom_valid_ = false;
    hb_h_ratio_  = 7.0f;
    hb_w_ratio_  = 2.5f;
    hb_dx_ratio_ = 0.0f;
    face_body_geom_valid_ = false;
    fb_h_ratio_  = 8.0f;
    fb_w_ratio_  = 3.2f;
    fb_dx_ratio_ = 0.0f;
    last_main_bw_ = -1.f; last_main_bh_ = -1.f;
    standalone_heads_.clear();
    standalone_head_scores_.clear();
    standalone_faces_.clear();
    standalone_face_scores_.clear();
    recovery_heads_.clear();
    recovery_head_scores_.clear();
    recovery_head_owner_person_.clear();
    recovery_head_owner_ambiguous_.clear();
    recovery_body_boxes_.clear();
    recovery_body_valid_.clear();
    recovery_body_identity_evidence_.clear();
    recovery_body_identity_reason_.clear();
    recovery_faces_.clear();
    recovery_face_scores_.clear();
    recovery_face_owner_person_.clear();
    recovery_face_owner_ambiguous_.clear();
    face_inference_cache_.clear();
    face_recovery_rotor_ = 0;
    face_recovery_fail_streak_ = 0;
    preserve_face_search_state_ = false;
    prefer_body_geometry_output_ = false;
    face_recovery_hypotheses_.clear();
    next_face_recovery_hyp_id_ = 1;
    last_confirmed_face_box_ = cv::Rect();
    last_confirmed_face_ms_ = -1;
    last_confirmed_face_frame_ = -1;
    face_global_pending_ = false;
    face_global_pending_cx_ = face_global_pending_cy_ = -1.f;
    face_global_pending_streak_ = 0;
    face_global_pending_ms_ = -1;
    head_reacq_pending_cx_ = head_reacq_pending_cy_ = -1.f;
    head_reacq_pending_streak_ = 0;
    head_reacq_pending_ms_ = -1;

    // 重置头部维持计时 + 扫描人脸暂定标记（B7/B1）+ 墙钟 TTL 状态（B6）
    head_only_since_ms_ = -1;
    pending_from_sweep_ = false;
    pending_sweep_frame_ = -1;
    last_face_lock_ms_  = -999999;
    alert_start_ms_     = 0;
    occ_start_ms_       = 0;

    // 重置朝向 + 肩部几何（目标切换 → 几何/朝向先验失效）
    frontalness_ = -1.f;
    shoulder_geom_valid_ = false;
    sb_dx_ratio_ = 0.0f;
    sb_dy_ratio_ = -0.30f;
    sb_w_ratio_  = 0.60f;
    orient_low_since_ms_ = -1;
    prev_incumbent_anchor_ = -1.f;

    // 重置长盲 coast 后的重捕身份复核计数（C-identity）+ 暂定假设缓冲
    reacq_defer_count_ = 0;
    pending_cx_ = pending_cy_ = -1.f;
    pending_active_ = false;
    pending_src_ = pending_idx_ = -1;   // [N4] 暂定假设对应检测

    // 重置可见度状态
    visibility_state_            = VisibilityState::FULL;
    visibility_pending_          = VisibilityState::FULL;
    visible_ratio_ema_           = 1.0f;
    visibility_pending_since_ms_ = 0;   // [N7]

    // 重置遮挡状态机
    occlusion_state_ = OcclusionState::CLEAR;
    occlusion_start_frame_ = 0;
    recovery_start_ms_ = 0;   // [N7]
    separation_streak_ = 0;
    person_identity_context_ = PersonIdentityAmbiguityContext{};
    pre_occ_emb_ = cv::Mat();
    occluder_tracker_id_ = -1;
    pre_occ_velocity_ = cv::Mat();
    occ_kf_clean_ = true;

    // 重置质量监控
    quality_history_.clear();
    baseline_reid_ = 0.0f;
    baseline_anchor_sim_ = 0.0f;
    stable_frame_count_ = 0;
    suspect_streak_ = 0;
    id_switch_alert_ = false;
    alert_frame_start_ = 0;

    // 重置候选级 Pose cache / budget / committed 状态。
    pose_cache_.clear();
    pose_budget_used_ = 0;
    last_committed_pose_frame_ = -1;
    last_committed_pose_ms_ = -1;
    committed_pose_box_ = cv::Mat();

    // 重置延迟人脸注册（含模板质量锁）+ 人脸扫描轮询 + 框补全保持值
    face_registered_ = false;
    last_face_register_ms_ = -999999;   // [N7] 注册间隔改墙钟
    face_template_quality_ = 0.f;
    face_sweep_rotor_ = 0;
    main_h_hold_ = 0.f;
    main_h_hold_ms_ = 0;

    // 重置人脸硬锚定状态（#1）
    face_locked_ = false;
    last_face_lock_frame_ = -999;
    face_lock_box_ = cv::Mat();

    // [N10] 重置自运动前馈（未闭合误差/激活/采样锚点全清；β 是相机属性 → 跨 reset 保留）
    ego_ex_ = ego_ey_ = 0.f;
    ego_active_ = false;
    ego_out_ms_ = 0;
    ego_sx_ = ego_sy_ = 0.f;
    ego_shift_mag_ = 0.f;
    head_ego_dx_ = head_ego_dy_ = 0.f;
    prev_match_cx_ = prev_match_cy_ = 0.f;
    prev_match_ex_ = prev_match_ey_ = 0.f;
    prev_match_ms_ = -1;
    prev_match_valid_ = false;

    // [N11] 重置头锚定瞄准点
    aim_x_ = aim_y_ = -1.f;
    aim_valid_ = false;

    // [N9] 重置决策 trace 环形缓冲
    trace_head_ = 0;
    trace_count_ = 0;
    trace_last_dump_frame_ = -999;
    trace_prev_occ_ = OcclusionState::CLEAR;
    trace_prev_alert_ = false;
    trace_event_pending_ = false;
    if (kMatchTrace) {
        trace_push("[TRACKER_RESET] f=0 all tracking state cleared");
        trace_flush("reset", true);
    }

    // [N12] 退出过载降级模式
    overload_mode_ = false;
    overload_hi_ = 0;
    overload_lo_ = 0;
    face_model_budget_ = kFaceBudgetNormal;
    face_priority_frame_ = false;
    face_priority_streak_ = 0;
    frame_allow_secondary_reid_ = false;

    // 重置 GMC 相机运动补偿状态（#3b）
    prev_gray_ = cv::Mat();
    gmc_scale_ = 1.0f;
}

// ============================================================
// 延迟/重新人脸注册
// 在主目标稳定跟踪时尝试注册或更新人脸模板
// ============================================================
void LightTracker::try_deferred_face_register(const cv::Mat& img,
                                              const DetectionGroups& det_groups,
                                              const MainMatchResult& match_result,
                                              int close_det_count) {
    // 状态位必须与真实模板库一致。旧注册路径在特征提取失败时仍把 flag 置 true，
    // 会让后续恢复误以为可识别，并把重试间隔拉到 12 秒。
    bool actual_template = face_recognizer.has_face_template();
    if (face_registered_ != actual_template) {
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_REG] f=%d result=repair_flag old=%d template=%d",
                frame_count, face_registered_ ? 1 : 0, actual_template ? 1 : 0);
            trace_push(line);
            trace_event_pending_ = true;
        }
        face_registered_ = actual_template;
    }

    // ── 身份安全闸（防"把冒充者的脸注册进模板"）──
    // 旧实现仅看 hit_streak：跟错人 10 帧后就会 reset() 清掉正确模板并写入
    // 冒充者的脸 → 此后人脸识别永久反向失效（确认错误目标、拉不回真目标）。
    // 现要求：本帧匹配成立 + 身份存疑期之外 + 无邻人。已有模板时，必须是
    // 本帧真人脸确认才允许替换；anchor 只能用于“尚无模板”的首次延迟注册，
    // 不能在跟错后拿另一个人的脸覆盖仍可用于拉回主目标的旧模板。
    if (id_switch_alert_ || suspect_streak_ > 0) return;
    if (!match_result.matched) return;
    if (close_det_count > 1) return;
    if (match_result.face_from_sweep) return;  // 暂定扫脸尚未完成近场身份闭环
    if (actual_template && !match_result.from_face) return;
    if (!actual_template
        && !match_result.from_face
        && match_result.anchor_sim < 0.70f)
        return;

    // ── 模板质量锁：已有足够好的正脸模板 → 永久停止重注册 ──
    // 用户实测：后期场景多/曾跟错时，周期重注册会把早期正确的好模板覆盖掉。
    // 质量达标后模板冻结（仅 reset() 解锁），彻底关闭这条污染通道。
    if (face_registered_ && face_template_quality_ >= kFaceTemplateGoodEnough)
        return;

    // 找到主目标 tracker
    int main_idx = -1;
    for (int i = 0; i < (int)trackers.size(); ++i) {
        if (trackers[i]->get_is_main()) { main_idx = i; break; }
    }
    if (main_idx < 0) return;
    auto& trk = trackers[main_idx];

    // 必须正在跟踪（非丢失），且稳定 hit 一段时间
    if (trk->get_time_since_update() > 0) return;
    if (trk->get_hit_streak() < 10) return;

    // [N7] 频率控制改墙钟（原帧计数：15f 慢帧下≈5.6s、快帧下≈0.4s，语义漂移 14×）
    int64_t reg_now = now_ms();
    int64_t interval_ms = face_registered_ ? kFaceReregisterMs : kFaceRegisterRetryMs;
    if (reg_now - last_face_register_ms_ < interval_ms) return;
    last_face_register_ms_ = reg_now;   // 消费本次尝试槽（即便后续质量不达标也不每帧重试）

    // 获取主目标最近观测框
    cv::Mat last_obs = trk->get_last_observation();
    if (last_obs.empty()) return;

    cv::Rect box(static_cast<int>(last_obs.at<float>(0, 0)),
                 static_cast<int>(last_obs.at<float>(0, 1)),
                 static_cast<int>(last_obs.at<float>(0, 2) - last_obs.at<float>(0, 0)),
                 static_cast<int>(last_obs.at<float>(0, 3) - last_obs.at<float>(0, 1)));
    cv::Rect region = box & cv::Rect(0, 0, img.cols, img.rows);
    if (region.area() <= 0) return;

    cv::Mat rect_img = img(region).clone();

    // ── 复用 update() 开头已关联好的人脸框（matchPersonFaces 把脸绑到本人体）──
    //   免去对主框裁剪区再跑一次检测器（省一次 YOLO 推理）；且"关联"比旧的
    //   "裁剪区几何包含"更严格地保证这张脸属于主目标本人（身后重叠者的脸不会混入）。
    //   face_src[match_result.index] = 该人体的人脸框集合（[k,5]：xyxy + score，全帧坐标）；
    //   空 → 本帧主目标没拍到脸（背身/侧脸）→ 跳过（与旧 face_cnt!=1 的"无脸即跳过"一致）。
    const std::vector<cv::Mat>& face_src =
        (match_result.source == 0) ? det_groups.dets_one_face
                                   : det_groups.dets_second_face;
    if (match_result.index < 0 || match_result.index >= (int)face_src.size()) return;
    const cv::Mat& assoc_faces = face_src[match_result.index];
    if (assoc_faces.empty() || assoc_faces.rows < 1) return;
    // 关联通常仅一张；多张时取置信度最高的一张
    int   best_fi = -1; float best_fs = -1.f;
    for (int fi = 0; fi < assoc_faces.rows; ++fi) {
        float s = assoc_faces.at<float>(fi, 4);
        if (s > best_fs) { best_fs = s; best_fi = fi; }
    }
    if (best_fi < 0) return;
    // 全帧 xyxy → 裁剪区局部坐标（减去裁剪原点），仍按全工程"xyxy 塞 Rect"约定填入
    // face_box（.width/.height 存 x2/y2），与原检测路径下游用法逐字节一致。
    float fx1 = assoc_faces.at<float>(best_fi, 0) - (float)region.x;
    float fy1 = assoc_faces.at<float>(best_fi, 1) - (float)region.y;
    float fx2 = assoc_faces.at<float>(best_fi, 2) - (float)region.x;
    float fy2 = assoc_faces.at<float>(best_fi, 3) - (float)region.y;
    if (fx2 - fx1 <= 0.f || fy2 - fy1 <= 0.f) return;     // 退化框 → 跳过
    cv::Rect face_box((int)fx1, (int)fy1, (int)fx2, (int)fy2);   // xyxy 塞 Rect

    // 关键点检测 + 模板质量评估（[N6] 复用抽出的 evaluate_face_quality；含最小脸尺寸门）
    if (!take_face_model_slot()) return;
    FaceKeypointResult fk_result = detector_fk.run(rect_img, face_box);
    std::vector<float> kps_10 = get_kps10(fk_result.points);
    if (kps_10.size() < 10) return;

    // [N6] 注册最小脸尺寸门（当前与识别门统一为 14px）：脸高 = y2−y（face_box 为 xyxy 塞 Rect）
    if ((float)(face_box.height - face_box.y) < kFaceRegisterMinFacePx) {
        null_sink << "[FACE_REG] skip: face too small (h="
                     << (face_box.height - face_box.y)
                     << " < " << kFaceRegisterMinFacePx << ")" << std::endl;
        return;
    }
    float new_q = evaluate_face_quality(kps_10, face_box, rect_img.cols, rect_img.rows);
    if (new_q < 0.f) return;   // 五官退化（eye_d<1）→ 放弃本次

    // 已有模板时：仅当新样本质量显著更高才替换（防侧脸/小脸样本反向降质）
    if (face_registered_
        && new_q <= face_template_quality_ + kFaceTemplateUpgradeMargin) {
        null_sink << "[FACE_REG] skip: new_q=" << new_q
                  << " <= stored_q=" << face_template_quality_ << std::endl;
        return;
    }

    // 注册/升级模板，并记录质量；升级必须先成功提取新特征再原子替换，失败时
    // 保留旧模板，不能 reset() 后留下“flag=true、模板库为空”的永久失效状态。
    bool replacing = face_recognizer.has_face_template();
    std::string reg_result = replacing
                           ? face_recognizer.replace_face(rect_img, kps_10, "bro")
                           : face_recognizer.register_face(rect_img, kps_10, "bro");
    if (reg_result != "success" || !face_recognizer.has_face_template()) {
        face_registered_ = face_recognizer.has_face_template();
        if (kMatchTrace) {
            char line[kTraceLineLen];
            std::snprintf(line, sizeof(line),
                "[FACE_REG] f=%d result=reject mode=%s reason=%s template=%d",
                frame_count, replacing ? "upgrade" : "deferred",
                reg_result.c_str(), face_registered_ ? 1 : 0);
            trace_push(line);
            trace_event_pending_ = true;
        }
        return;
    }
    face_registered_ = true;
    face_template_quality_ = new_q;
    last_confirmed_face_box_ = cv::Rect(
        (int)std::lround(assoc_faces.at<float>(best_fi, 0)),
        (int)std::lround(assoc_faces.at<float>(best_fi, 1)),
        (int)std::lround(assoc_faces.at<float>(best_fi, 2)),
        (int)std::lround(assoc_faces.at<float>(best_fi, 3)));
    last_confirmed_face_ms_ = now_ms();
    last_confirmed_face_frame_ = frame_count;
    if (kMatchTrace) {
        char line[kTraceLineLen];
        std::snprintf(line, sizeof(line),
            "[FACE_REG] f=%d result=accept mode=%s q=%.2f",
            frame_count, replacing ? "upgrade" : "deferred", new_q);
        trace_push(line);
    }
    if (!replacing) {
        null_sink << "[FACE_REG] initial face registered (q=" << new_q
                  << ") at frame " << frame_count << std::endl;
    } else {
        null_sink << "[FACE_REG] template upgraded (q=" << new_q
                  << (new_q >= kFaceTemplateGoodEnough ? ", LOCKED" : "")
                  << ") at frame " << frame_count << std::endl;
    }
}

// ============================================================
// 清理过期 tracker（非主目标，超过存活期未更新的）
// [N7] 存活期改墙钟 kTrackMaxAgeMs：帧计数 max_age 在 27~374ms 帧间隔波动下语义漂移
//   严重（共存否决轨迹本该按真实时间存活）。有盖戳(last_update_ms_≥0)按墙钟判定；
//   从未盖戳（新建未命中）回退帧计数 max_age 作兜底。
// ============================================================
void LightTracker::cleanup_expired_trackers() {
    const int64_t clean_now = now_ms();
    std::vector<int> removed_ids;
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
            [this, clean_now, &removed_ids](const std::shared_ptr<KalmanBoxTracker>& trk) {
                // 主目标永不清理（由 reset 统一管理）
                if (trk->get_is_main()) return false;
                bool expired = trk->last_update_ms_ >= 0
                    ? (clean_now - trk->last_update_ms_) > kTrackMaxAgeMs
                    : trk->get_time_since_update() > max_age;   // 无盖戳 → 帧计数兜底
                if (expired) removed_ids.push_back(trk->get_id());
                return expired;
            }),
        trackers.end());
    for (int id : removed_ids) {
        secondary_frame_observations_.erase(id);
        relative_motion_history_.erase(id);
        if (occluder_tracker_id_ == id) {
            occluder_tracker_id_ = -1;
            emergence_valid_ = false;
            emergence_update_ms_ = -1;
        }
    }
}

// ============================================================
// GMC 相机运动补偿（#3b）
//
// 适配跳帧 + 不固定间隔的关键点：
//   只用"上一处理帧"和"当前处理帧"两张实际图像做稀疏特征匹配，
//   直接测量背景的真实仿射变换，不依赖任何时间外推 →
//   无论两帧间隔多长、是否跳帧，估计都是"已经发生的位移"，对变间隔免疫。
//
// 安全降级：任何一步失败（首帧/特征不足/内点不足/变换异常）都返回空矩阵，
//   调用方退回恒等变换（即不补偿），绝不会因为坏估计污染预测框。
// ============================================================
cv::Mat LightTracker::estimate_camera_motion(const cv::Mat& img, const cv::Mat& trks) {
    if (!gmc_enabled_ || img.empty()) {
        return cv::Mat();
    }
    fxprof::ScopedGmc _gtot(fxprof::GmcStage::Total);  // 整段墙钟（CPU 前段 / IVE 各算子 / RANSAC）

#ifdef USE_HISI_IVE
    // ── IVE 硬件路径：灰度下采样/金字塔/角点/LK 全在 IVE，仿射拟合留 CPU ──
    {
        if (!ive_gmc_.ready() && img.cols > 0) {
            int ww = kGmcWorkWidth;
            int wh = (int)std::lround((double)kGmcWorkWidth * img.rows / img.cols);
            ive_gmc_.init(ww, wh);
        }
        // 前景框（tracker 预测框，xyxy 塞 cv::Rect，与全工程约定一致）→ 排除其上的角点
        std::vector<cv::Rect> boxes;
        boxes.reserve(trks.rows);
        for (int i = 0; i < trks.rows; ++i) {
            const float* r = trks.ptr<float>(i);
            boxes.push_back(cv::Rect((int)r[0], (int)r[1], (int)r[2], (int)r[3]));
        }
        std::vector<cv::Point2f> pp, cp; float wscale = 1.f;
        // Y 平面零拷贝快路：app 传入 NV12/YUV420SP 的 Y 平面且尺寸与 img 一致时，省掉
        //   match_bgr 的 CPU 转灰度+缩放+上传，直接 IVE 缩放 Y→work；否则回退 BGR 路径。
        bool y_ok = cur_y_valid_ && cur_y_w_ == img.cols && cur_y_h_ == img.rows;
        bool got  = y_ok
            ? ive_gmc_.match_yuv420sp(cur_y_phys_, cur_y_virt_, cur_y_w_, cur_y_h_, cur_y_stride_,
                                      boxes, pp, cp, wscale)
            : ive_gmc_.match_bgr(img, boxes, pp, cp, wscale);
        if (!got)
            return cv::Mat();                       // 首帧/特征不足/失败 → 不补偿
        if ((int)pp.size() < kGmcMinInliers) return cv::Mat();

        // 与 OpenCV 路径完全一致的 CPU 尾段：拟合 + 校验 + 平移还原
        std::vector<uchar> inliers;
        cv::Mat M;
        {
            fxprof::ScopedGmc _gr(fxprof::GmcStage::Ransac);
            M = cv::estimateAffinePartial2D(pp, cp, inliers, cv::RANSAC, 3.0);
        }
        if (M.empty() || !cv::checkRange(M, true)
            || !std::isfinite(wscale) || wscale <= 1.0e-6f) {
            return cv::Mat();
        }
        int inlier_count = cv::countNonZero(inliers);
        if (inlier_count < kGmcMinInliers) {
            null_sink << "[GMC-IVE] too few inliers: " << inlier_count << std::endl;
            return cv::Mat();
        }
        double a = M.at<double>(0, 0), b = M.at<double>(0, 1);
        double est_scale = std::sqrt(a * a + b * b);
        if (std::abs(est_scale - 1.0) > kGmcMaxScaleDev) {
            null_sink << "[GMC-IVE] abnormal scale: " << est_scale << std::endl;
            return cv::Mat();
        }
        M.at<double>(0, 2) /= wscale;               // work→orig（平移分量）
        M.at<double>(1, 2) /= wscale;
        double tx = M.at<double>(0, 2), ty = M.at<double>(1, 2);
        if (std::abs(tx) > img.cols * 0.5 || std::abs(ty) > img.rows * 0.5) {
            null_sink << "[GMC-IVE] abnormal translation: (" << tx << "," << ty << ")" << std::endl;
            return cv::Mat();
        }
        null_sink << "[GMC-IVE] scale=" << est_scale << " t=(" << tx << "," << ty
                  << ") inliers=" << inlier_count << "/" << pp.size() << std::endl;
        return M;
    }
#else

    // 1. 转灰度
    cv::Mat gray;
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::SwGray);   // cvtColor
        if (img.channels() == 3)
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4)
            cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
        else
            gray = img.clone();
    }

    // 2. 缩放到工作分辨率（降低计算量；相似变换的旋转/缩放分量与图像尺度无关，
    //    平移分量稍后按尺度还原）
    float scale = 1.0f;
    if (gray.cols > kGmcWorkWidth) {
        scale = (float)kGmcWorkWidth / (float)gray.cols;
        fxprof::ScopedGmc _g(fxprof::GmcStage::SwResize);   // resize
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    gmc_scale_ = scale;

    // 3. 首帧（或尺寸变化）：仅缓存灰度图，无法估计 → 返回空（不补偿）
    if (prev_gray_.empty() || prev_gray_.size() != gray.size()) {
        prev_gray_ = gray.clone();
        return cv::Mat();
    }

    // 4. 前景掩膜：把 tracker 预测框（缩放到工作尺度并外扩）置 0，
    //    避免在运动目标上取特征点污染背景估计
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    if (!trks.empty()) {
        const int pad = 8;
        for (int i = 0; i < trks.rows; ++i) {
            // trks 行语义为 [x1,y1,x2,y2]
            float bx1 = trks.at<float>(i, 0) * scale;
            float by1 = trks.at<float>(i, 1) * scale;
            float bx2 = trks.at<float>(i, 2) * scale;
            float by2 = trks.at<float>(i, 3) * scale;
            int rx = std::max(0, (int)std::floor(std::min(bx1, bx2)) - pad);
            int ry = std::max(0, (int)std::floor(std::min(by1, by2)) - pad);
            int rw = (int)std::ceil(std::abs(bx2 - bx1)) + 2 * pad;
            int rh = (int)std::ceil(std::abs(by2 - by1)) + 2 * pad;
            rw = std::min(rw, gray.cols - rx);
            rh = std::min(rh, gray.rows - ry);
            if (rw > 0 && rh > 0 && rx < gray.cols && ry < gray.rows)
                mask(cv::Rect(rx, ry, rw, rh)) = 0;
        }
    }

    // 5. 在背景区检测角点
    std::vector<cv::Point2f> prev_pts;
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::SwCorner);   // goodFeaturesToTrack
        cv::goodFeaturesToTrack(prev_gray_, prev_pts, 400, 0.01, 8, mask);
    }
    if (prev_pts.size() < (size_t)kGmcMinInliers) {
        prev_gray_ = gray.clone();
        return cv::Mat();
    }

    // 6. LK 光流跟踪到当前帧
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::SwLK);   // calcOpticalFlowPyrLK
        cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts, curr_pts, status, err,
                                 cv::Size(21, 21), 3);
    }

    std::vector<cv::Point2f> good_prev, good_curr;
    good_prev.reserve(prev_pts.size());
    good_curr.reserve(prev_pts.size());
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            good_prev.push_back(prev_pts[i]);
            good_curr.push_back(curr_pts[i]);
        }
    }

    // 无论本次是否成功，都推进 prev_gray_ 供下一帧使用
    cv::Mat curr_gray_copy = gray.clone();

    if (good_prev.size() < (size_t)kGmcMinInliers) {
        prev_gray_ = curr_gray_copy;
        return cv::Mat();
    }

    // 7. RANSAC 估计相似变换（旋转+缩放+平移，4 自由度，比全仿射更稳健）
    std::vector<uchar> inliers;
    cv::Mat M;
    {
        fxprof::ScopedGmc _gr(fxprof::GmcStage::Ransac);
        M = cv::estimateAffinePartial2D(good_prev, good_curr, inliers, cv::RANSAC, 3.0);
    }
    prev_gray_ = curr_gray_copy;

    if (M.empty() || !cv::checkRange(M, true)
        || !std::isfinite(scale) || scale <= 1.0e-6f) {
        return cv::Mat();
    }

    int inlier_count = cv::countNonZero(inliers);
    if (inlier_count < kGmcMinInliers) {
        null_sink << "[GMC] too few inliers: " << inlier_count << " -> identity" << std::endl;
        return cv::Mat();
    }

    // 8. 变换合理性检查（防止异常仿射污染预测框）
    double a = M.at<double>(0, 0);
    double b = M.at<double>(0, 1);
    double est_scale = std::sqrt(a * a + b * b);
    if (std::abs(est_scale - 1.0) > kGmcMaxScaleDev) {
        null_sink << "[GMC] abnormal scale: " << est_scale << " -> identity" << std::endl;
        return cv::Mat();
    }

    // 平移分量从工作尺度还原到原图尺度（旋转/缩放分量无需改动）
    M.at<double>(0, 2) /= scale;
    M.at<double>(1, 2) /= scale;

    double tx = M.at<double>(0, 2);
    double ty = M.at<double>(1, 2);
    if (std::abs(tx) > img.cols * 0.5 || std::abs(ty) > img.rows * 0.5) {
        null_sink << "[GMC] abnormal translation: (" << tx << "," << ty
                  << ") -> identity" << std::endl;
        return cv::Mat();
    }

    null_sink << "[GMC] scale=" << est_scale << " t=(" << tx << "," << ty
              << ") inliers=" << inlier_count << "/" << good_prev.size() << std::endl;
    return M;
#endif /* USE_HISI_IVE */
}

cv::Point2f LightTracker::warp_point(float x, float y, const cv::Mat& M) {
    // M: 2x3 CV_64F 仿射矩阵
    double nx = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2);
    double ny = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2);
    return cv::Point2f((float)nx, (float)ny);
}

void LightTracker::warp_box_inplace(cv::Mat& box4, const cv::Mat& M) {
    if (M.empty() || box4.empty() || box4.cols < 4) return;
    // box4: [x1,y1,x2,y2]，变换 4 个角点后取轴对齐外接框
    float x1 = box4.at<float>(0, 0);
    float y1 = box4.at<float>(0, 1);
    float x2 = box4.at<float>(0, 2);
    float y2 = box4.at<float>(0, 3);

    cv::Point2f p0 = warp_point(x1, y1, M);
    cv::Point2f p1 = warp_point(x2, y1, M);
    cv::Point2f p2 = warp_point(x1, y2, M);
    cv::Point2f p3 = warp_point(x2, y2, M);

    box4.at<float>(0, 0) = std::min({p0.x, p1.x, p2.x, p3.x});
    box4.at<float>(0, 1) = std::min({p0.y, p1.y, p2.y, p3.y});
    box4.at<float>(0, 2) = std::max({p0.x, p1.x, p2.x, p3.x});
    box4.at<float>(0, 3) = std::max({p0.y, p1.y, p2.y, p3.y});
}
