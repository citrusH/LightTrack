#ifndef FX_MODEL_PROFILER_H
#define FX_MODEL_PROFILER_H

// ════════════════════════════════════════════════════════════════════════
// 模型耗时统计（每帧）
// ------------------------------------------------------------------------
// 统计五个推理模型在「一帧算法」内的：预处理 / 推理 / 后处理 耗时，
// 以及本帧每个模型被调用的次数。在 LightTracker::update 结束时打印汇总。
//
// 用法：
//   - 帧边界：在 LightTracker::update 顶部放一个 fxprof::ScopedFrame，
//     构造时清零、析构时打印（覆盖所有 return 分支）。
//   - 阶段计时：在各模型 run()/get_feat() 里用 fxprof::ScopedPhase 包住
//     preProcess/inference/postProcess，并在入口处 fxprof::add_call(...)。
//
// 纯头文件、无新增 .cpp（不影响 c_api/README.md 里的手动交叉编译命令）。
// 关闭统计：把下面的 kEnabled 改成 false 即可（探针退化为近似空操作）。
// ════════════════════════════════════════════════════════════════════════

#include <chrono>
#include <cstdio>
#include <array>

namespace fxprof {

// 总开关：false 时所有探针不累加、不打印。
constexpr bool kEnabled = false;

enum class Model { Detector = 0, ReID, Pose, FaceKps, FaceReco, COUNT };
enum class Phase { Pre = 0, Infer, Post };

// GMC（相机运动补偿）分阶段计时：与模型计时共用同一帧边界(ScopedFrame)与开关(kEnabled)，
//   用于定位 estimate_camera_motion 的 75~160ms 究竟耗在 CPU 前段 / IVE 各算子 / RANSAC。
//   Front  : CPU 前段（BGR→灰度+缩放+上传，match_bgr）；Y 平面零拷贝路径无此段。
//   Resize : IVE resize（full→L0 与金字塔逐层）合计；calls 即本帧 IVE resize 次数。
//   StCand : IVE st_cand_corner；StCorner: st_corner（同步）。
//   CornerRead: CPU 读角点+前景过滤+写定点；LK: IVE 金字塔光流。
//   Ransac : estimateAffinePartial2D + 合理性校验；Total: estimate_camera_motion 整段墙钟。
//
// 两条实现路径各自的算子都单独计时（report 只打印本帧 calls>0 的项）：
//   [IVE 路] Front(CPU 前段) / Resize(金字塔) / StCand+StCorner(角点) / CornerRead / LK(光流)
//   [软件路 #else/OpenCV] SwGray(cvtColor) / SwResize / SwCorner(goodFeaturesToTrack) /
//                         SwLK(calcOpticalFlowPyrLK)
//   [两路共用] Ransac(estimateAffinePartial2D) / Total
// Total 必须排在 Ransac 之后、COUNT 之前（report 的“分段和”按 != Total 累加）。
enum class GmcStage {
    Front = 0, Resize, StCand, StCorner, CornerRead, LK,   // IVE 路
    SwGray, SwResize, SwCorner, SwLK,                      // 软件(OpenCV)路
    Ransac, Total, COUNT                                   // 共用
};

inline const char* model_name(Model m) {
    switch (m) {
        case Model::Detector: return "Detector";
        case Model::ReID:     return "ReID    ";
        case Model::Pose:     return "Pose    ";
        case Model::FaceKps:  return "FaceKps ";
        case Model::FaceReco: return "FaceReco";
        default:              return "?       ";
    }
}

inline const char* gmc_stage_name(GmcStage s) {
    switch (s) {
        // —— IVE 路 ——
        case GmcStage::Front:      return "Front(ive-cpu) ";  // cvtColor+resize+上传
        case GmcStage::Resize:     return "Resize(ive)    ";  // 金字塔下采样
        case GmcStage::StCand:     return "Corner-c(ive)  ";  // st_cand_corner（角点候选）
        case GmcStage::StCorner:   return "Corner-p(ive)  ";  // st_corner（角点提取）
        case GmcStage::CornerRead: return "CornerRead(cpu)";  // 读角点+前景过滤
        case GmcStage::LK:         return "OptFlow(ive)   ";  // LK 金字塔光流
        // —— 软件(OpenCV)路 ——
        case GmcStage::SwGray:     return "Gray(sw)       ";  // cvtColor
        case GmcStage::SwResize:   return "Resize(sw)     ";  // resize
        case GmcStage::SwCorner:   return "Corner(sw)     ";  // goodFeaturesToTrack
        case GmcStage::SwLK:       return "OptFlow(sw)    ";  // calcOpticalFlowPyrLK
        // —— 共用 ——
        case GmcStage::Ransac:     return "AffineFit(cpu) ";  // estimateAffinePartial2D
        case GmcStage::Total:      return "TOTAL          ";
        default:                   return "?              ";
    }
}

struct Stat {
    int    calls    = 0;
    double pre_ms   = 0.0;   // 本帧该模型预处理总耗时
    double infer_ms = 0.0;   // 本帧该模型推理总耗时
    double post_ms  = 0.0;   // 本帧该模型后处理总耗时
    void reset() { calls = 0; pre_ms = infer_ms = post_ms = 0.0; }
};

struct GStat {
    int    calls = 0;
    double ms    = 0.0;   // 本帧该 GMC 阶段累计耗时
    void reset() { calls = 0; ms = 0.0; }
};

class Profiler {
public:
    static Profiler& instance() { static Profiler p; return p; }

    void add_call(Model m) { stats_[(int)m].calls++; }

    void add(Model m, Phase ph, double ms) {
        Stat& s = stats_[(int)m];
        if      (ph == Phase::Pre)   s.pre_ms   += ms;
        else if (ph == Phase::Infer) s.infer_ms += ms;
        else                         s.post_ms  += ms;
    }

    void add_gmc(GmcStage s, double ms) { GStat& g = gmc_[(int)s]; g.calls++; g.ms += ms; }

    void reset_frame() {
        for (auto& s : stats_) s.reset();
        for (auto& g : gmc_)   g.reset();
    }

    void report() {
        double grand = 0.0;
        std::printf("[ModelProfiler] ===== frame model summary =====\n");
        std::printf("[ModelProfiler]   model     calls      pre      infer       post      total\n");
        for (int i = 0; i < (int)Model::COUNT; ++i) {
            const Stat& s = stats_[i];
            if (s.calls == 0) continue;          // 本帧未用到的模型不打印
            double tot = s.pre_ms + s.infer_ms + s.post_ms;
            grand += tot;
            std::printf("[ModelProfiler]   %s  %4d  %7.2fms %8.2fms %8.2fms %8.2fms\n",
                        model_name((Model)i), s.calls,
                        s.pre_ms, s.infer_ms, s.post_ms, tot);
        }
        std::printf("[ModelProfiler]   ----- frame model time total: %.2f ms -----\n", grand);

        // GMC 分阶段（仅在本帧跑过 GMC 时打印）
        bool any_gmc = false;
        for (int i = 0; i < (int)GmcStage::COUNT; ++i) if (gmc_[i].calls) { any_gmc = true; break; }
        if (any_gmc) {
            std::printf("[ModelProfiler] ----- GMC (camera motion comp) stages -----\n");
            std::printf("[ModelProfiler]   stage          calls       time\n");
            double gsum = 0.0;
            for (int i = 0; i < (int)GmcStage::COUNT; ++i) {
                const GStat& g = gmc_[i];
                if (g.calls == 0) continue;
                if ((GmcStage)i != GmcStage::Total) gsum += g.ms;   // Total 是墙钟，不计入分段和
                std::printf("[ModelProfiler]   %s  %4d  %8.2fms\n",
                            gmc_stage_name((GmcStage)i), g.calls, g.ms);
            }
            std::printf("[ModelProfiler]   ----- GMC stage sum (excl TOTAL): %.2f ms -----\n", gsum);
        }
    }

private:
    std::array<Stat, (int)Model::COUNT> stats_{};
    std::array<GStat, (int)GmcStage::COUNT> gmc_{};
};

inline void add_call(Model m) { if (kEnabled) Profiler::instance().add_call(m); }

// 计时单个 GMC 阶段，析构时累加到对应阶段（calls++ / ms+=）。
class ScopedGmc {
public:
    explicit ScopedGmc(GmcStage s) : s_(s), t0_(std::chrono::steady_clock::now()) {}
    ~ScopedGmc() {
        if (!kEnabled) return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        Profiler::instance().add_gmc(s_, ms);
    }
    ScopedGmc(const ScopedGmc&) = delete;
    ScopedGmc& operator=(const ScopedGmc&) = delete;
private:
    GmcStage s_;
    std::chrono::steady_clock::time_point t0_;
};

// 计时单个阶段（pre/infer/post），析构时把耗时累加到对应模型的对应阶段。
class ScopedPhase {
public:
    ScopedPhase(Model m, Phase ph)
        : m_(m), ph_(ph), t0_(std::chrono::steady_clock::now()) {}
    ~ScopedPhase() {
        if (!kEnabled) return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
        Profiler::instance().add(m_, ph_, ms);
    }
    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;
private:
    Model m_;
    Phase ph_;
    std::chrono::steady_clock::time_point t0_;
};

// 帧边界：构造时清零本帧统计，析构时打印汇总（覆盖 update 的所有 return）。
class ScopedFrame {
public:
    ScopedFrame()  { if (kEnabled) Profiler::instance().reset_frame(); }
    ~ScopedFrame() { if (kEnabled) Profiler::instance().report(); }
    ScopedFrame(const ScopedFrame&) = delete;
    ScopedFrame& operator=(const ScopedFrame&) = delete;
};

} // namespace fxprof

#endif // FX_MODEL_PROFILER_H
