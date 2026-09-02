/* ════════════════════════════════════════════════════════════════════
 * fx_tracker.cpp — C API 桥接层（以 C++ 编译，对外导出纯 C 符号）
 *
 * 设计：薄桥接、零算法逻辑。所有行为由 LightTracker 决定，调用路径与
 * C++ 直接使用完全一致（同一份编译代码 → 字节级同效）。
 * 直接桥接 LightTracker（旧 Track 类只是无状态转发层，跳过不改变行为）。
 * 异常绝不跨越 C 边界：全部捕获并转错误码。
 * ════════════════════════════════════════════════════════════════════ */
#include "fx_tracker.h"

#include "../LightTracker.h"

#include <opencv2/opencv.hpp>
#include <tuple>
#include <iostream>

/* 不透明句柄的真身 */
struct FxTrackerHandle {
    LightTracker tracker;   /* 默认 LightTrackerConfig，与旧 Track 类一致 */
};

extern "C" {

const char* fx_tracker_version(void) {
    return FX_TRACKER_API_VERSION;
}

FxTrackerHandle* fx_tracker_create(void) {
    try {
        std::cout << "====== fx_tracker_create  v" << FX_TRACKER_API_VERSION
                  << " ======" << std::endl;
        return new FxTrackerHandle();
    } catch (...) {
        return NULL;
    }
}

void fx_tracker_destroy(FxTrackerHandle* handle) {
    /* delete NULL 安全 */
    delete handle;
}

int fx_tracker_init(FxTrackerHandle* handle) {
    if (!handle) return FX_ERR_NULL_HANDLE;
    try {
        return (handle->tracker.init() == 0) ? FX_OK : FX_ERR_INIT_FAILED;
    } catch (...) {
        return FX_ERR_EXCEPTION;
    }
}

int fx_tracker_run(FxTrackerHandle* handle,
                   const FxFrame* frame,
                   const FxRect* main_target,
                   FxTrackResult* out) {
    if (!handle)                 return FX_ERR_NULL_HANDLE;
    if (!frame || !out)          return FX_ERR_NULL_ARG;

    /* 先清零输出，任何错误路径下 out 都是确定状态 */
    out->count = 0;
    out->total_count = 0;
    for (int i = 0; i < FX_MAX_PERSONS; ++i) {
        out->persons[i].id = 0;
        out->persons[i].score = 0.f;
        out->persons[i].x = out->persons[i].y = 0;
        out->persons[i].w = out->persons[i].h = 0;
    }

    if (!frame->bgr_data || frame->width <= 0 || frame->height <= 0)
        return FX_ERR_BAD_FRAME;

    try {
        /* 零拷贝包装外部 BGR 缓冲（与旧 wrapper 一致）。
         * LightTracker 只读输入并在需要处自行 clone，调用返回后缓冲可复用。 */
        cv::Mat image(frame->height, frame->width, CV_8UC3,
                      const_cast<unsigned char*>(frame->bgr_data));

        /* 主目标框：xyxy 塞入 cv::Rect（全工程约定）。
         * NULL → (0,0,0,0)，area()==0 → 核心按"正常跟踪帧"处理。
         * （修复旧 wrapper 对"可选"指针的无条件解引用崩溃。） */
        cv::Rect main_rect(0, 0, 0, 0);
        if (main_target)
            main_rect = cv::Rect(main_target->x1, main_target->y1,
                                 main_target->x2, main_target->y2);

        /* 可选 Y 平面：传给核心用于 GMC 零拷贝硬件加速（IVE）。
         * y_phys==0 时核心自动回退 BGR 路径，行为不变。每帧调用以避免沿用旧帧地址。 */
        handle->tracker.set_frame_yplane(frame->y_phys, frame->y_virt,
                                         frame->width, frame->height, frame->y_stride);

        cv::Mat rows;
        int person_cnt = 0;
        std::tie(rows, person_cnt) = handle->tracker.update(image, main_rect);
        (void)person_cnt;   /* 当前 C API 结果结构未单独暴露人体检测数 */

        /* 行格式 [x1,y1,x2,y2,id] → C 结构（w/h 转换与旧 wrapper 一致） */
        out->total_count = rows.rows;
        int n = (rows.rows < FX_MAX_PERSONS) ? rows.rows : FX_MAX_PERSONS;
        for (int i = 0; i < n; ++i) {
            int x1 = (int)rows.at<float>(i, 0);
            int y1 = (int)rows.at<float>(i, 1);
            int x2 = (int)rows.at<float>(i, 2);
            int y2 = (int)rows.at<float>(i, 3);
            int id = (int)rows.at<float>(i, 4);

            out->persons[i].id = id;
            out->persons[i].x  = x1;
            out->persons[i].y  = y1;
            out->persons[i].w  = x2 - x1;
            out->persons[i].h  = y2 - y1;
            /* 主目标行携带云台控制权重：真实身体/严格接受的头脸观测为 1.0，
             * 有限短时预测为 (0,1)，无输出为 0；其他行恒 0。 */
            out->persons[i].score =
                (id == FX_MAIN_ID) ? handle->tracker.get_coast_weight() : 0.f;
        }
        out->count = n;
        return FX_OK;
    } catch (const TrackerRuntimeError& e) {
        std::cerr << "fx_tracker_run model error: " << e.what() << std::endl;
        return FX_ERR_MODEL_RUN;
    } catch (...) {
        return FX_ERR_EXCEPTION;
    }
}

void fx_tracker_reset(FxTrackerHandle* handle) {
    if (!handle) return;
    try { handle->tracker.reset(); } catch (...) {}
}

float fx_tracker_get_coast_weight(FxTrackerHandle* handle) {
    if (!handle) return 0.f;
    return handle->tracker.get_coast_weight();
}

int fx_tracker_get_reset_flag(FxTrackerHandle* handle) {
    if (!handle) return 0;
    return handle->tracker.get_reset_flag() ? 1 : 0;
}

void fx_tracker_clear_reset_flag(FxTrackerHandle* handle) {
    if (!handle) return;
    handle->tracker.set_reset_flag(false);
}

} /* extern "C" */
